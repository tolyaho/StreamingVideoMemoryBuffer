#!/usr/bin/env python3
"""Batch MCQ eval: hierarchical memory vs recent-window baseline on a manifest of videos."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (
    HierarchicalRetriever,
    LLMReasoner,
    PerceptionEncoder,
    ReasonerInputFormatter,
    SummaryBuilder,
)

from scripts.eval_common import (
    NOTEBOOK_CONFIG,
    print_final_summary,
    refresh_summary,
    run_video,
    save_video_results,
)

DEFAULT_MANIFEST = ROOT / "data/eval_manifest_50.json"
DEFAULT_OUT = ROOT / "outputs/eval_batch"
RECORDS_GLOB = "sample_*.jsonl"


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--samples",
        type=str,
        default="",
        help="comma-separated sample ids (default: all in manifest)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="shorthand for --samples 1,36")
    parser.add_argument(
        "--reasoner-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model id for text-only MCQ reasoner",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.pilot:
        wanted = {1, 36}
    elif args.samples:
        wanted = {int(x.strip()) for x in args.samples.split(",") if x.strip()}
    else:
        wanted = None

    entries = [e for e in manifest if wanted is None or e["sample_id"] in wanted]
    if not entries:
        raise SystemExit("no manifest entries to run")

    out_dir = args.output_dir
    per_video_dir = out_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    cfg = NOTEBOOK_CONFIG
    print(f"[eval] manifest : {args.manifest}")
    print(f"[eval] videos   : {len(entries)}")
    print(f"[eval] output   : {out_dir}")
    print(f"[eval] reasoner : {args.reasoner_model}")
    print("[eval] loading models (once)...")

    encoder = PerceptionEncoder()
    summary_builder = SummaryBuilder(use_model=True, use_vlm=True, use_moondream=True)
    retriever = HierarchicalRetriever(tau_fraction=cfg.tau_fraction)
    formatter = ReasonerInputFormatter()
    reasoner = LLMReasoner(model_name=args.reasoner_model)
    print("[eval] models ready.\n")

    for i, entry in enumerate(entries, start=1):
        sid = entry["sample_id"]
        out_path = per_video_dir / f"sample_{sid:02d}.jsonl"
        if args.resume and out_path.is_file():
            print(f"[{i}/{len(entries)}] sample_{sid} — skip (resume)")
            continue

        video_path = resolve(entry["video"])
        qas_path = resolve(entry["qas"])
        print(f"[{i}/{len(entries)}] sample_{sid} — streaming {video_path.name} ...", flush=True)

        records, meta = run_video(
            video_path,
            qas_path,
            clip_id=str(sid),
            encoder=encoder,
            summary_builder=summary_builder,
            retriever=retriever,
            formatter=formatter,
            reasoner=reasoner,
            cfg=cfg,
            extra_meta={"sample_id": sid},
        )
        _, h_ok, b_ok = save_video_results(
            out_dir,
            per_video_dir,
            out_path,
            records,
            meta,
            records_glob=RECORDS_GLOB,
        )
        print(
            f"         done in {meta['elapsed_s']}s · "
            f"{meta['n_windows']} windows · "
            f"hier {h_ok}/{len(records)} · base {b_ok}/{len(records)}"
        )

    summary = refresh_summary(out_dir, per_video_dir, records_glob=RECORDS_GLOB)
    print_final_summary(summary, out_dir / "summary.md")


if __name__ == "__main__":
    main()
