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
    PerceptionEncoder,
    ReasonerInputFormatter,
    SummaryBuilder,
)

from src.qwen_vl_io import TEXT_EVAL_EVENT_VLM

from scripts.eval_common import (
    EVAL_CONFIGS,
    STREAMINGBENCH_TUNED_CONFIG,
    VLM_FULL_CONFIG,
    print_final_summary,
    refresh_summary,
    run_video,
    save_video_results,
)
from scripts.reasoner_factory import ReasonerType, build_reasoner, default_reasoner_model

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
        "--reasoner-type",
        choices=("text", "vlm"),
        default="text",
        help="text: captions only; vlm: retrieved frames + text memory",
    )
    parser.add_argument(
        "--reasoner-model",
        default="",
        help="HF model id (default: 3B-Instruct for text, Qwen3-VL-8B for vlm)",
    )
    parser.add_argument(
        "--no-share-event-vlm",
        action="store_true",
        help="load a separate VLM for reasoning instead of reusing the event VLM",
    )
    parser.add_argument(
        "--config",
        choices=sorted(EVAL_CONFIGS),
        default="",
        help="hyperparameter preset (default: streamingbench_tuned for text, vlm_full for vlm)",
    )
    args = parser.parse_args()

    reasoner_type: ReasonerType = args.reasoner_type
    reasoner_model = args.reasoner_model or default_reasoner_model(reasoner_type)
    if args.config:
        cfg = EVAL_CONFIGS[args.config]
    elif reasoner_type == "vlm":
        cfg = VLM_FULL_CONFIG
    else:
        cfg = STREAMINGBENCH_TUNED_CONFIG
    config_name = args.config or (
        "vlm_full" if reasoner_type == "vlm" else "streamingbench_tuned"
    )

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

    print(f"[eval] manifest : {args.manifest}")
    print(f"[eval] config   : {config_name}")
    print(f"[eval] videos   : {len(entries)}")
    print(f"[eval] output   : {out_dir}")
    print(f"[eval] reasoner : {reasoner_type} ({reasoner_model})")
    print("[eval] loading models (once)...")

    encoder = PerceptionEncoder()
    event_vlm_name = reasoner_model if reasoner_type == "vlm" else TEXT_EVAL_EVENT_VLM
    summary_builder = SummaryBuilder(
        use_model=True,
        use_vlm=True,
        use_moondream=True,
        vlm_model_name=event_vlm_name,
    )
    retriever = HierarchicalRetriever(tau_fraction=cfg.tau_fraction)
    formatter = ReasonerInputFormatter()
    reasoner = build_reasoner(
        reasoner_type,
        reasoner_model,
        cfg,
        summary_builder,
        share_event_vlm=not args.no_share_event_vlm,
    )
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
            extra_meta={
                "sample_id": sid,
                "reasoner_type": reasoner_type,
                "reasoner_model": reasoner_model,
                "config": config_name,
            },
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
