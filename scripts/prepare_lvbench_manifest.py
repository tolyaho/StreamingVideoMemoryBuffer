#!/usr/bin/env python3
"""Sample random LVBench clips, write qas.json bundles, and optionally download videos."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_youtube import add_cookie_args, download
from scripts.fetch_lvbench_videos import fetch_one
from scripts.lvbench_utils import load_meta, sample_videos, write_clip_bundle

LOG = logging.getLogger(__name__)

DEFAULT_META = (
    ROOT / "data/streamingbench/Real_Time_Visual_Understanding/shard_1_50/video_info.meta.jsonl"
)
DEFAULT_DATA = ROOT / "data/lvbench_eval"
DEFAULT_MANIFEST = ROOT / "data/lvbench_eval_manifest.json"
DEFAULT_COOKIES = ROOT / "data/.youtube_cookies.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("-n", "--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch missing video.mp4 files from YouTube",
    )
    parser.add_argument(
        "--download-hf",
        action="store_true",
        help="fetch missing videos from LVBench HF archive (recommended)",
    )
    parser.add_argument(
        "--hf-archive-dir",
        type=Path,
        default=ROOT / "data/lvbench_hf_archive",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default="",
        help="comma-separated YouTube ids (overrides random sample)",
    )
    add_cookie_args(parser, cookies_default=DEFAULT_COOKIES)
    args = parser.parse_args()

    if not args.meta.is_file():
        raise SystemExit(f"meta file not found: {args.meta}")

    if args.keys:
        wanted = {k.strip() for k in args.keys.split(",") if k.strip()}
        by_key = {r["key"]: r for r in load_meta(args.meta)}
        missing = wanted - by_key.keys()
        if missing:
            raise SystemExit(f"keys not in meta: {sorted(missing)}")
        rows = [by_key[k] for k in sorted(wanted)]
    else:
        rows = sample_videos(args.meta, args.count, args.seed)

    args.data_root.mkdir(parents=True, exist_ok=True)
    manifest = [write_clip_bundle(row, args.data_root) for row in rows]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    LOG.info("manifest → %s (%d clips, %d QAs)",
             args.manifest, len(manifest), sum(e["n_qas"] for e in manifest))

    if args.download_hf:
        missing = []
        for entry in manifest:
            dest = Path(entry["video"])
            if dest.is_file():
                LOG.info("%s: video exists", entry["video_key"])
                continue
            if not fetch_one(entry["video_key"], dest, args.hf_archive_dir):
                missing.append(entry["video_key"])
        if missing:
            raise SystemExit(
                f"{len(missing)} not in HF archive cache: {', '.join(missing)}. "
                "Run scripts/fetch_lvbench_videos.py --download-parts once, "
                "or place mp4s under data/lvbench_hf_archive/all_videos/."
            )

    if args.download:
        cookies = args.cookies if args.cookies and args.cookies.is_file() else None
        if cookies is None:
            LOG.warning("no cookies file at %s — YouTube may block downloads", args.cookies)
        failed = []
        for entry in manifest:
            video_path = Path(entry["video"])
            if video_path.is_file():
                LOG.info("%s: video exists", entry["video_key"])
                continue
            try:
                download(
                    entry["video_key"],
                    video_path,
                    cookies=cookies,
                    cookies_from_browser=args.cookies_from_browser,
                )
            except Exception as exc:
                LOG.error("%s: download failed: %s", entry["video_key"], exc)
                failed.append(entry["video_key"])
        if failed:
            raise SystemExit(
                f"{len(failed)} download(s) failed: {', '.join(failed)}. "
                "Try --cookies / --cookies-from-browser, or --download-hf."
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
