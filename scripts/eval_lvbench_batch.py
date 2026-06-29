#!/usr/bin/env python3
"""Batch MCQ eval on LVBench clips: hierarchical memory vs recent-window baseline."""
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
    EvalConfig,
    NOTEBOOK_CONFIG,
    VLM_FULL_CONFIG,
    VLM_STREAMING_CONFIG,
    print_final_summary,
    refresh_summary,
    run_video,
    save_video_results,
)
from scripts.reasoner_factory import ReasonerType, build_reasoner, default_reasoner_model

DEFAULT_MANIFEST = ROOT / "data/lvbench_eval_manifest.json"
DEFAULT_OUT = ROOT / "outputs/eval_lvbench"
RECORDS_GLOB = "*.jsonl"

LVBENCH_TUNED_CONFIG = EvalConfig(
    recent_capacity=30,
    baseline_recent_capacity=20,
    episodic_capacity=150,
    novelty_threshold=0.02,
    episode_max_gap=6.0,
    event_max_gap=30.0,
    episodic_merge_batch=15,
    event_min_episode_sim=0.62,
    top_m=2,
    top_k=8,
    baseline_top_k=5,
    tau_fraction=0.08,
    neighbor_radius=2,
    recent_episodes=8,
)

LVBENCH_EVAL_CONFIGS = {
    "notebook": NOTEBOOK_CONFIG,
    "lvbench_tuned": LVBENCH_TUNED_CONFIG,
    "vlm_full": VLM_FULL_CONFIG,
    "vlm_streaming": VLM_STREAMING_CONFIG,
}


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--keys",
        type=str,
        default="",
        help="comma-separated YouTube ids (default: all in manifest)",
    )
    parser.add_argument("--resume", action="store_true")
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
        choices=sorted(LVBENCH_EVAL_CONFIGS),
        default="",
        help="hyperparameter preset (default: lvbench_tuned for text, vlm_full for vlm)",
    )
    args = parser.parse_args()

    reasoner_type: ReasonerType = args.reasoner_type
    reasoner_model = args.reasoner_model or default_reasoner_model(reasoner_type)
    if args.config:
        cfg = LVBENCH_EVAL_CONFIGS[args.config]
    elif reasoner_type == "vlm":
        cfg = VLM_FULL_CONFIG
    else:
        cfg = LVBENCH_TUNED_CONFIG
    config_name = args.config or (
        "vlm_full" if reasoner_type == "vlm" else "lvbench_tuned"
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = None
    if args.keys:
        wanted = {k.strip() for k in args.keys.split(",") if k.strip()}
    entries = [e for e in manifest if wanted is None or e["video_key"] in wanted]
    if not entries:
        raise SystemExit("no manifest entries to run")

    out_dir = args.output_dir
    per_video_dir = out_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lvbench] manifest : {args.manifest}")
    print(f"[lvbench] config   : {config_name}")
    print(f"[lvbench] reasoner : {reasoner_type} ({reasoner_model})")
    print(f"[lvbench] videos   : {len(entries)}")
    print(f"[lvbench] output   : {out_dir}")
    print("[lvbench] loading models (once)...")

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
    print("[lvbench] models ready.\n")

    for i, entry in enumerate(entries, start=1):
        video_key = entry["video_key"]
        out_path = per_video_dir / f"{video_key}.jsonl"
        if args.resume and out_path.is_file():
            print(f"[{i}/{len(entries)}] {video_key} — skip (resume)")
            continue

        video_path = resolve(entry["video"])
        qas_path = resolve(entry["qas"])
        if not video_path.is_file():
            print(f"[{i}/{len(entries)}] {video_key} — missing video")
            continue

        print(
            f"[{i}/{len(entries)}] {video_key} — streaming "
            f"({entry.get('n_qas', '?')} QAs) ...",
            flush=True,
        )
        records, meta = run_video(
            video_path,
            qas_path,
            clip_id=video_key,
            encoder=encoder,
            summary_builder=summary_builder,
            retriever=retriever,
            formatter=formatter,
            reasoner=reasoner,
            cfg=cfg,
            extra_meta={
                "video_key": video_key,
                "type": entry.get("type"),
                "duration_minutes": entry.get("duration_minutes"),
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
            summary_title="LVBench eval summary",
        )
        print(
            f"         done in {meta['elapsed_s']}s · "
            f"{meta['n_windows']} windows · "
            f"hier {h_ok}/{len(records)} · base {b_ok}/{len(records)}"
        )

    summary = refresh_summary(
        out_dir,
        per_video_dir,
        records_glob=RECORDS_GLOB,
        title="LVBench eval summary",
    )
    if summary.get("n_qas", 0) == 0:
        raise SystemExit("no QAs evaluated — download videos first?")

    print_final_summary(summary, out_dir / "summary.md")


if __name__ == "__main__":
    main()
