#!/usr/bin/env python3
"""Copy LVBench videos from a local HF archive (extracted dir or zip parts)."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger(__name__)

HF_REPO = "AIWinter/LVBench"
ZIP_PARTS = [f"all_videos_split.zip.{i:03d}" for i in range(1, 15)]
DEFAULT_ARCHIVE = ROOT / "data/lvbench_hf_archive"
DEFAULT_MANIFEST = ROOT / "data/lvbench_eval_manifest.json"


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_in_tree(root: Path, video_key: str) -> Path | None:
    names = {f"{video_key}.mp4", f"{video_key}.MP4"}
    for pattern in (f"**/{video_key}.mp4", f"**/{video_key}.MP4"):
        for path in root.glob(pattern):
            if path.is_file() and path.name in names:
                return path
    return None


def copy_video(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        LOG.info("exists: %s", dest)
        return
    shutil.copy2(src, dest)
    LOG.info("copied %s → %s (%.1f MiB)", src.name, dest, dest.stat().st_size / (1024 * 1024))


def ensure_concat_zip(archive_dir: Path) -> Path:
    joined = archive_dir / "all_videos.zip"
    if joined.is_file():
        return joined
    parts = [archive_dir / name for name in ZIP_PARTS]
    missing = [p.name for p in parts if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing zip parts in {archive_dir}: {missing[:3]}"
            f"{'...' if len(missing) > 3 else ''}"
        )
    LOG.info("joining %d zip parts → %s", len(parts), joined.name)
    with joined.open("wb") as out:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, out)
    return joined


def extract_from_zip(zip_path: Path, video_key: str, dest: Path) -> bool:
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        member = next(
            (n for n in zf.namelist() if n.rstrip("/").endswith(f"{video_key}.mp4")),
            None,
        )
        if member is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)
    LOG.info("extracted %s → %s (%.1f MiB)", video_key, dest, dest.stat().st_size / (1024 * 1024))
    return True


def download_zip_parts(archive_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in ZIP_PARTS:
        target = archive_dir / name
        if target.is_file():
            LOG.info("have %s", name)
            continue
        LOG.info("downloading %s from %s (large, one-time)...", name, HF_REPO)
        path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=name)
        shutil.copy2(path, target)


def fetch_one(video_key: str, dest: Path, archive_dir: Path) -> bool:
    if dest.is_file():
        LOG.info("%s: already at destination", video_key)
        return True

    hit = find_in_tree(archive_dir, video_key)
    if hit is not None:
        copy_video(hit, dest)
        return True

    extracted = archive_dir / "all_videos"
    if extracted.is_dir():
        hit = find_in_tree(extracted, video_key)
        if hit is not None:
            copy_video(hit, dest)
            return True

    try:
        zip_path = ensure_concat_zip(archive_dir)
    except FileNotFoundError:
        return False

    return extract_from_zip(zip_path, video_key, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--keys",
        type=str,
        default="",
        help="comma-separated YouTube ids (default: all in manifest)",
    )
    parser.add_argument(
        "--download-parts",
        action="store_true",
        help="download HF zip parts into archive-dir first (~62 GB one-time)",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    wanted = None
    if args.keys:
        wanted = {k.strip() for k in args.keys.split(",") if k.strip()}
    entries = [e for e in manifest if wanted is None or e["video_key"] in wanted]
    if not entries:
        raise SystemExit("no manifest entries")

    if args.download_parts:
        download_zip_parts(args.archive_dir)

    ok, missing = [], []
    for entry in entries:
        key = entry["video_key"]
        dest = Path(entry["video"])
        if fetch_one(key, dest, args.archive_dir):
            ok.append(key)
        else:
            missing.append(key)

    LOG.info("done: %d ok, %d missing", len(ok), len(missing))
    if missing:
        raise SystemExit(
            f"missing in archive: {', '.join(missing)}. "
            f"Put extracted mp4s under {args.archive_dir}/all_videos/ "
            f"or run with --download-parts."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
