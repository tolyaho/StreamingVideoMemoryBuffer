#!/usr/bin/env python3
# Download shard_1_50 of StreamingBench Real_Time_Visual_Understanding and
# write qas.json next to each video. --annotations-only skips the zip.
import argparse
import ast
import json
import logging
import re
import shutil
import time
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

SUBSET = "Real_Time_Visual_Understanding"
ZIP_NAME = "Real-Time Visual Understanding_1-50.zip"
OUTPUT_DIR = Path("data/streamingbench") / SUBSET / "shard_1_50"

LOG = logging.getLogger(__name__)


def parse_options(raw):
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return [raw]
    return [str(x) for x in parsed] if isinstance(parsed, list) else [str(parsed)]


def sample_id_from_row(row):
    qid = str(row.get("question_id", ""))
    suffix = qid.split("_sample_", 1)[1]
    return int(suffix.split("_", 1)[0])


def find_video_in_zip(zf, sample_id):
    videos = [
        m for m in zf.namelist()
        if m.lower().endswith(".mp4") and not m.endswith("/") and "__MACOSX" not in m
    ]
    for m in videos:
        for part in Path(m).parts:
            if re.match(rf"^sample_0*{sample_id}$", part, re.IGNORECASE):
                return m
        if re.search(rf"sample_0*{sample_id}\.mp4$", m, re.IGNORECASE):
            return m
    return None


def extract_video(zf, member, dest_dir):
    tmp = Path(zf.extract(member, path=dest_dir))
    final = dest_dir / "video.mp4"
    if tmp != final:
        shutil.move(str(tmp), str(final))
        parent = tmp.parent
        while parent != dest_dir and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return final


def find_local_video(videos_root: Path, sid: int) -> Path | None:
    sample_dir = videos_root / f"sample_{sid}"
    if not sample_dir.is_dir():
        return None
    preferred = sample_dir / "video.mp4"
    if preferred.is_file():
        return preferred.resolve()
    alt = sample_dir / f"sample_{sid}.mp4"
    if alt.is_file():
        return alt.resolve()
    mp4s = [p for p in sample_dir.iterdir() if p.suffix.lower() == ".mp4" and p.is_file()]
    if len(mp4s) == 1:
        return mp4s[0].resolve()
    return None


def load_grouped_annotations():
    LOG.info("loading annotations: ngqtrung/StreamingBench split=%r (first run can take a while)...", SUBSET)
    t_ds = time.perf_counter()
    ds = load_dataset("ngqtrung/StreamingBench", split=SUBSET)
    LOG.info("dataset object ready in %.1fs; scanning rows for sample ids 1-50...", time.perf_counter() - t_ds)

    grouped = {}
    row_count = 0
    for row in ds:
        row_count += 1
        if row_count % 5000 == 0:
            LOG.info("  ... scanned %d rows, matched %d distinct samples so far", row_count, len(grouped))
        sid = sample_id_from_row(row)
        if 1 <= sid <= 50:
            grouped.setdefault(sid, []).append(dict(row))

    LOG.info("annotation scan done: %d rows in %.1fs -> %d samples in range 1-50",
             row_count, time.perf_counter() - t_ds, len(grouped))
    return grouped


def qas_payload(rows, sid: int, video_path: Path):
    qas = [{
        "question_id": r.get("question_id"),
        "task_type": r.get("task_type"),
        "question": r.get("question"),
        "time_stamp": r.get("time_stamp"),
        "answer": r.get("answer"),
        "options": parse_options(r.get("options", [])),
    } for r in rows]
    return {"sample_id": sid, "video": str(video_path), "qas": qas}


def run_from_local_videos(grouped: dict, videos_root: Path) -> tuple[int, int]:
    root = videos_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"videos root not found: {root}")

    LOG.info("annotations-only: matching QAs to videos under %s", root)
    n_done = 0
    n_skip = 0
    for sid in sorted(grouped):
        rows = grouped[sid]
        video_path = find_local_video(root, sid)
        if video_path is None:
            LOG.warning("sample_%d: no video under %s, skipping", sid, root / f"sample_{sid}")
            n_skip += 1
            continue
        sample_dir = video_path.parent
        out = sample_dir / "qas.json"
        out.write_text(
            json.dumps(qas_payload(rows, sid, video_path), indent=2),
            encoding="utf-8",
        )
        n_done += 1
        LOG.info("sample_%d: wrote %s (%d QAs)", sid, out.name, len(rows))
    return n_done, n_skip


def run_from_zip(grouped: dict) -> tuple[int, int]:
    LOG.info("downloading zip from Hugging Face: mjuicem/StreamingBench / %s (this is often the slow step)...", ZIP_NAME)
    t_zip = time.perf_counter()
    zip_path = Path(hf_hub_download(
        repo_id="mjuicem/StreamingBench",
        repo_type="dataset",
        filename=ZIP_NAME,
    ))
    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    LOG.info("zip ready in %.1fs -> %s (%.1f MiB)", time.perf_counter() - t_zip, zip_path, zip_mb)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.info("extracting videos and writing qas.json under %s", OUTPUT_DIR.resolve())

    n_done = 0
    n_skip = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        n_mp4 = sum(
            1 for zi in infos
            if zi.filename.lower().endswith(".mp4") and "__MACOSX" not in zi.filename
        )
        LOG.info("opened zip: %d members, %d mp4 paths", len(infos), n_mp4)

        for sid in sorted(grouped):
            rows = grouped[sid]
            sample_dir = OUTPUT_DIR / f"sample_{sid}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            member = find_video_in_zip(zf, sid)
            if member is None:
                LOG.warning("sample_%d: no matching mp4 in zip, skipping", sid)
                n_skip += 1
                continue

            LOG.info("sample_%d: extracting %s ...", sid, member)
            t_ex = time.perf_counter()
            video_path = extract_video(zf, member, sample_dir)

            (sample_dir / "qas.json").write_text(
                json.dumps(qas_payload(rows, sid, video_path), indent=2),
                encoding="utf-8",
            )
            n_done += 1
            LOG.info("sample_%d: wrote %s + qas.json (%d QAs) in %.2fs",
                     sid, video_path.name, len(rows), time.perf_counter() - t_ex)

    return n_done, n_skip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="skip zip: load HF annotations only and write qas.json next to existing videos",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=Path("data/Real-Time%20Visual%20Understanding_1-50"),
        help="directory containing sample_<id>/video.mp4 (used with --annotations-only)",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.annotations_only:
        LOG.info("mode: annotations-only (no zip)")
    else:
        LOG.info("output directory: %s", OUTPUT_DIR.resolve())

    grouped = load_grouped_annotations()

    if args.annotations_only:
        n_done, n_skip = run_from_local_videos(grouped, args.videos_root)
    else:
        n_done, n_skip = run_from_zip(grouped)

    LOG.info("finished: %d ok, %d skipped, total elapsed %.1fs", n_done, n_skip, time.perf_counter() - t0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
