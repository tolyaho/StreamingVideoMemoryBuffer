"""Capture the ACTUAL pipeline on a source video. Requires the normal model environment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.artifacts import digest, validate_media  # noqa: E402 (script entry point)
from demo.trace import TraceRecorder  # noqa: E402 (script entry point)

# Demo-only preset, never used for the published StreamingBench numbers. Identical to the
# notebook preset except episodic_capacity: the benchmark's 100 episodes only overflow on long
# videos (9 of 50 StreamingBench videos), so on a few-minute clip the long-term tier would never be
# written while streaming. 8 lets the same consolidation code run inside a short clip.
SHORT_CLIP_DEMO = {"base": "notebook", "overrides": {"episodic_capacity": 8}}

CONFIG_CHOICES = ("notebook", "streamingbench_tuned", "vlm_full", "vlm_streaming", "short_clip_demo")


def resolve_config(name: str):
    from scripts.eval_common import EVAL_CONFIGS
    if name == "short_clip_demo":
        return replace(EVAL_CONFIGS[SHORT_CLIP_DEMO["base"]], **SHORT_CLIP_DEMO["overrides"])
    return EVAL_CONFIGS[name]


def cached_revision(repo: str) -> str | None:
    """Commit the local Hugging Face cache resolves for `repo` (what from_pretrained will load)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    ref = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--qas", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "demo/local/run.json")
    parser.add_argument("--config", choices=CONFIG_CHOICES, default="notebook")
    parser.add_argument("--reasoner-type", choices=("text", "vlm"), default="text")
    parser.add_argument("--rights", type=Path, help="Required to include representative frames in an export")
    parser.add_argument("--focus-question", help="question_id the walkthrough features (chosen before the run)")
    parser.add_argument("--pins", type=Path, help="JSON {repo: commit}; refuse to run if the cache resolves elsewhere")
    args = parser.parse_args()
    for path in (args.video, args.qas):
        if not path.is_file():
            parser.error(f"Required source missing: {path.name}")
    missing = [m for m in ("torch", "transformers", "cv2", "numpy", "PIL") if importlib.util.find_spec(m) is None]
    if missing:
        parser.error("Model environment unavailable (" + ", ".join(missing) + "). Install the repository's normal dependencies; no fake run was created.")
    import torch
    import transformers
    from src import HierarchicalRetriever, PerceptionEncoder, ReasonerInputFormatter, SummaryBuilder
    from src.qwen_vl_io import TEXT_EVAL_EVENT_VLM
    from scripts.eval_common import run_video
    from scripts.reasoner_factory import build_reasoner, default_reasoner_model

    model = default_reasoner_model(args.reasoner_type)
    event_vlm = model if args.reasoner_type == "vlm" else TEXT_EVAL_EVENT_VLM
    repos = ["microsoft/xclip-base-patch32", "microsoft/Florence-2-base", "vikhyatk/moondream2", event_vlm, model]
    revisions = {repo: cached_revision(repo) for repo in dict.fromkeys(repos)}
    if args.pins:
        pins = json.loads(args.pins.read_text())
        wrong = {r: c for r, c in revisions.items() if pins.get(r) != c}
        if wrong:
            parser.error(f"Cached model revisions differ from the pins: {sorted(wrong)}")

    recorder = TraceRecorder(args.video, args.qas, model, args.config)
    if args.rights:
        rights = json.loads(args.rights.read_text())
        validate_media(recorder.session, args.video, rights)
        recorder.include_frames = True
        recorder.session["provenance"]["rights"] = rights
    cfg = resolve_config(args.config)
    prov = recorder.session["provenance"]
    prov["pipeline_config"] = asdict(cfg)
    prov["config_note"] = (f"Demo-only preset: notebook settings with {SHORT_CLIP_DEMO['overrides']} "
                           "so long-term consolidation can occur within a short clip. Not a benchmark setting."
                           if args.config == "short_clip_demo" else "Unmodified repository preset.")
    prov["models"] = {"window_captioner": "microsoft/Florence-2-base", "episode_captioner": "vikhyatk/moondream2",
                      "event_summarizer": event_vlm, "embeddings": "microsoft/xclip-base-patch32",
                      "reasoner": model, "revisions": revisions}
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    prov["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                       "transformers": transformers.__version__, "device": device, "machine": platform.machine(),
                       "platform": platform.platform()}
    prov["code_files"] = [{"path": p, "sha256": digest(ROOT / p)} for p in
        ("scripts/eval_common.py", "src/memory_writer.py", "src/retriever.py", "src/formatter.py",
         "src/summary_builder.py", "src/prompts.py", "src/perception_encoder.py", "src/stream_reader.py",
         "demo/trace.py", "demo/capture.py")]
    encoder = PerceptionEncoder()
    summary = SummaryBuilder(use_model=True, use_vlm=True, use_moondream=True, vlm_model_name=event_vlm,
                             moondream_revision=revisions.get("vikhyatk/moondream2"))
    reasoner = build_reasoner(args.reasoner_type, model, cfg, summary)
    run_video(args.video, args.qas, clip_id=recorder.session["case_id"], encoder=encoder,
              summary_builder=summary, retriever=HierarchicalRetriever(tau_fraction=cfg.tau_fraction),
              formatter=ReasonerInputFormatter(), reasoner=reasoner, cfg=cfg, observer=recorder)
    recorder.write(args.output, focus_question_id=args.focus_question)
    print(f"Captured actual run: {args.output}")


if __name__ == "__main__":
    main()
