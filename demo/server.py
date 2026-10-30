"""Loopback-only demo server. Serves web assets and one explicitly bound video, never the repo."""
from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.artifacts import build_reference, digest, validate, validate_media  # noqa: E402 (script entry point)


def byte_range(header: str, size: int) -> tuple[int, int]:
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or size <= 0:
        raise ValueError("Invalid range")
    first, last = match.groups()
    if not first:
        if not last or int(last) <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - int(last)), size - 1
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start > end or start >= size:
        raise ValueError("Unsatisfiable range")
    return start, end


def attach_annotation(session: dict, annotation: dict, sha256: str) -> dict:
    """Human ground truth written before inference. It must agree with the run's own labels."""
    questions = annotation.get("questions", {})
    for row in session["records"]:
        label = questions.get(row["question_id"])
        if label is None or label["answer"] != row["hierarchical"]["ground_truth"]:
            raise ValueError("Annotation does not match the run's questions and ground truth")
        for a, b in label.get("evidence_stimulus_seconds", []):
            if not 0 <= a < b <= session["clock_end"] + 3:
                raise ValueError("Annotated evidence outside the source clock")
    session = copy.deepcopy(session)
    session["annotation"], session["annotation_sha256"] = annotation, sha256
    return session


def make_server(session: dict, *, port: int = 8765, media: Path | None = None,
                rights: dict | None = None, web: Path | None = None,
                extra_sessions: list[dict] | None = None) -> ThreadingHTTPServer:
    """Serve one or more recorded runs. Every run must be bound to the same source video."""
    bodies, runs = [], []
    for index, run in enumerate([session, *(extra_sessions or [])]):
        validate(run)
        data = copy.deepcopy(run)
        if media is not None:
            validate_media(data, media, rights or {})
            data["media"].update(available=True, status="cleared", url="/media/source", rights=rights)
        else:
            data["media"]["available"] = False
            # A trace can be inspected before rights clearance, without exposing its images.
            for node in data["nodes"].values():
                node.pop("thumbnail", None)
        bodies.append(json.dumps(data, ensure_ascii=False, allow_nan=False).encode())
        runs.append({"index": index, "mode": data["mode"], "config": data.get("config"), "model": data.get("model")})
    body = bodies[0]
    runs_body = json.dumps(runs).encode()
    media_type = {".webm": "video/webm", ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime"}.get(
        media.suffix.lower(), "application/octet-stream") if media is not None else None
    web_root = (web or ROOT / "demo/web").resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def do_HEAD(self) -> None:
            self.send_resource(head=True)

        def do_GET(self) -> None:
            self.send_resource(head=False)

        def response_headers(self, status: int, size: int, mime: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")

        def send_resource(self, *, head: bool) -> None:
            path = unquote(urlsplit(self.path).path)
            run = re.fullmatch(r"/api/session(?:/(\d+))?", path)
            if run or path == "/api/runs":
                if path == "/api/runs":
                    payload = runs_body
                elif run.group(1) is None:
                    payload = body
                elif int(run.group(1)) < len(bodies):
                    payload = bodies[int(run.group(1))]
                else:
                    self.send_error(404)
                    return
                self.response_headers(200, len(payload), "application/json; charset=utf-8")
                self.end_headers()
                if not head:
                    self.wfile.write(payload)
                return
            if path == "/media/source" and media is not None:
                size = media.stat().st_size
                return self.serve_media(media, size, head)
            target = (web_root / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if not target.is_relative_to(web_root) or not target.is_file() or target.suffix not in (".html", ".css", ".mjs", ".json", ".svg"):
                self.send_error(404)
                return
            content = target.read_bytes()
            mime = "text/javascript" if target.suffix == ".mjs" else mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.response_headers(200, len(content), mime)
            self.end_headers()
            if not head:
                self.wfile.write(content)

        def serve_media(self, path: Path, size: int, head: bool) -> None:
            requested = self.headers.get("Range")
            try:
                start, end = byte_range(requested, size) if requested else (0, size - 1)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.response_headers(206 if requested else 200, end - start + 1, media_type)
            self.send_header("Accept-Ranges", "bytes")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head:
                return
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(256 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--trace", type=Path, action="append", help="recorded run; repeat to offer several runs of one source")
    parser.add_argument("--media", type=Path)
    parser.add_argument("--rights", type=Path)
    parser.add_argument("--annotation", type=Path, help="human ground-truth file written before the run")
    parser.add_argument("--tour-question", help="question_id the guided tour features (presentation only; the trace is not changed)")
    parser.add_argument("--showcase", action="store_true",
                        help="serve the recorded Big Buck Bunny runs in demo/showcase (run demo/fetch_showcase_media.py first)")
    args = parser.parse_args()
    if args.showcase:
        show = ROOT / "demo/showcase"
        media = show / "media/bbb-excerpt-0033-0459.webm"
        if not media.is_file():
            parser.error("Showcase clip missing. Run: python3 demo/fetch_showcase_media.py")
        args.trace = [show / "trace-notebook.json", show / "trace-short_clip_demo.json"]
        args.media, args.rights, args.annotation = media, show / "rights.json", show / "annotation.json"
        args.tour_question = args.tour_question or "bbb_q2_apple"
    sessions = [json.loads(t.read_text()) for t in args.trace] if args.trace else [build_reference()]
    if args.annotation:
        annotation = json.loads(args.annotation.read_text())
        try:
            sessions = [attach_annotation(s, annotation, digest(args.annotation)) for s in sessions]
        except ValueError as error:
            parser.error(str(error))
    rights = json.loads(args.rights.read_text()) if args.rights else None
    if args.tour_question:
        for s in sessions:
            if args.tour_question not in {r["question_id"] for r in s["records"]}:
                parser.error("--tour-question is not a question of every run")
            s["tour_question_id"] = args.tour_question
    try:
        server = make_server(sessions[0], port=args.port, media=args.media, rights=rights, extra_sessions=sessions[1:])
    except ValueError as error:
        parser.error(str(error))
    print(f"Local demo: http://127.0.0.1:{server.server_port} (recorded playback; no inference)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
