#!/usr/bin/env python3
"""Download a YouTube video as mp4."""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from pathlib import Path

import yt_dlp

LOG = logging.getLogger(__name__)

YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)


def parse_video_id(url_or_id: str) -> str:
    m = YOUTUBE_ID_RE.search(url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    raise ValueError(f"not a YouTube URL or 11-char video id: {url_or_id!r}")


def ensure_js_runtime() -> None:
    if shutil.which("node"):
        return
    for prefix in ("/usr/local/bin", "/usr/bin"):
        node = Path(prefix) / "node"
        if node.is_file():
            os.environ["PATH"] = f"{prefix}:{os.environ.get('PATH', '')}"
            break


def build_ydl_opts(
    out_path: Path,
    *,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
) -> dict:
    opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(out_path.with_suffix(".%(ext)s")),
        "merge_output_format": "mp4",
        "remux_video": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if cookies:
        opts["cookiefile"] = str(cookies)
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def download(
    url_or_id: str,
    out_path: Path,
    *,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
) -> Path:
    video_id = parse_video_id(url_or_id)
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file():
        LOG.info("already exists, skipping: %s", out_path)
        return out_path

    ensure_js_runtime()
    ydl_opts = build_ydl_opts(
        out_path,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
    )

    LOG.info("downloading %s ...", video_id)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    if not out_path.is_file():
        raise FileNotFoundError(f"download finished but missing: {out_path}")

    LOG.info("saved → %s (%.1f MiB)", out_path, out_path.stat().st_size / (1024 * 1024))
    return out_path


def add_cookie_args(parser: argparse.ArgumentParser, cookies_default: Path | None = None) -> None:
    parser.add_argument(
        "--cookies",
        type=Path,
        default=cookies_default,
        help="Netscape cookies.txt exported from a logged-in browser",
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="read cookies from local browser (e.g. chrome, firefox)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url_or_id", help="YouTube URL or 11-char video id")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="output mp4 path (default: data/lvbench/<id>/video.mp4)",
    )
    add_cookie_args(parser)
    args = parser.parse_args()

    video_id = parse_video_id(args.url_or_id)
    out = args.output or Path("data/lvbench") / video_id / "video.mp4"
    download(
        args.url_or_id,
        out,
        cookies=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
