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
    LLMReasoner,
    PerceptionEncoder,
    ReasonerInputFormatter,
    SummaryBuilder,
)

from scripts.eval_common import (
    EvalConfig,
    NOTEBOOK_CONFIG,
    print_final_summary,
    refresh_summary,
    run_video,
    save_video_results,
)

DEFAULT_MANIFEST = ROOT / "data/lvbench_eval_manifest.json"
DEFAULT_OUT = ROOT / "outputs/eval_lvbench"

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

EVAL_CONFIGS = {
    "notebook": NOTEBOOK_CONFIG,
    "lvbench_tuned": LVBENCH_TUNED_CONFIG,
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
        "--config",
        choices=sorted(EVAL_CONFIGS),
        default="notebook",
    )
    parser.add_argument(
        "--reasoner-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model id for text-only MCQ reasoner",
    )
    args = parser.parse_args()

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

    cfg = EVAL_CONFIGS[args.config]
    print(f"[lvbench] manifest : {args.manifest}")
    print(f"[lvbench] config   : {args.config}")
    print(f"[lvbench] reasoner : {args.reasoner_model}")
    print(f"[lvbench] videos   : {len(entries)}")
    print(f"[lvbench] output   : {out_dir}")
    print("[lvbench] loading models (once)...")

    encoder = PerceptionEncoder()
    summary_builder = SummaryBuilder(use_model=True, use_vlm=True, use_moondream=True)
    retriever = HierarchicalRetriever(tau_fraction=cfg.tau_fraction)
    formatter = ReasonerInputFormatter()
    reasoner = LLMReasoner(model_name=args.reasoner_model)
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
                "config": args.config,
            },
        )
        _, h_ok, b_ok = save_video_results(
            out_dir,
            per_video_dir,
            out_path,
            records,
            meta,
            summary_title="LVBench eval summary",
        )
        print(
            f"         done in {meta['elapsed_s']}s · "
            f"{meta['n_windows']} windows · "
            f"hier {h_ok}/{len(records)} · base {b_ok}/{len(records)}"
        )

    summary = refresh_summary(
        out_dir, per_video_dir, title="LVBench eval summary"
    )
    if summary.get("n_qas", 0) == 0:
        raise SystemExit("no QAs evaluated — download videos first?")

    print_final_summary(summary, out_dir / "summary.md")


if __name__ == "__main__":
    main()
