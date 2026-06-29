# Streaming Video Memory Buffer

This repo is a research prototype for long video QA. It stores a video stream as a hierarchy of recent windows, episodic spans, and long-term events, then retrieves a small evidence set for each question instead of feeding the whole video history to a model.

The main claim is narrow: for a text-only reasoner, hierarchical memory recovers useful past evidence better than a flat recent-window buffer. On StreamingBench Real-Time Visual Understanding, the current text run gets 129/250 with hierarchical memory vs 112/250 for the recent baseline. With a stronger VLM reasoner, both systems are much closer, which is an important caveat rather than a failure to hide.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
```

The full model runs need local videos plus Hugging Face model downloads. The main config-driven entry point is:

```bash
streamvmb experiment=streamingbench_text
```

Useful overrides:

```bash
streamvmb experiment=streamingbench_text dataset.limit=1 output.dir=outputs/smoke_text
streamvmb experiment=streamingbench_vlm dataset.samples='[36]' output.dir=outputs/sample36_vlm
streamvmb experiment=lvbench_text dataset.keys='[16Z-XQh9jhk]' output.dir=outputs/lvbench_one
```

Older scripts under `scripts/` still work and are kept for reproducibility, but new experiments should use configs under `configs/`.

## What Is Implemented

- Online video windowing with OpenCV.
- X-CLIP embeddings for video windows and text queries.
- Three-tier memory: recent windows, episodic spans, long-term events.
- Novelty-based promotion, self-centrality episode pooling, and event consolidation.
- Tiered captioning: Florence-2 for windows, Moondream2 for episodes, Qwen-VL for event fusion.
- Coarse-to-fine retrieval with time decay and representative-window grounding.
- Text and VLM MCQ reasoners.
- Recent-window baseline for head-to-head comparisons.
- Config-driven experiments with Hydra.
- Unit tests for the pure memory/retrieval/config pieces.

## Repo Layout

- `src/` — core implementation.
- `configs/` — Hydra experiment presets and config groups.
- `scripts/` — older launchers and data helpers.
- `tests/` — fast tests that do not download models or videos.
- `notebooks/demo.ipynb` — original notebook walkthrough and result narrative.
- `results/` — committed StreamingBench eval outputs.
- `docs/` — architecture and experiment notes for working with the repo.
- `data/` — local manifests and QA annotations; videos are fetched locally.

## Results

StreamingBench, 50 videos, 250 MCQs:

| Run | Reasoner | Hierarchical | Baseline | Delta |
| --- | --- | ---: | ---: | ---: |
| text tuned | Qwen2.5-3B-Instruct | 129/250 | 112/250 | +6.8 pp |
| text notebook | Qwen2.5-3B-Instruct | 128/250 | 109/250 | +7.6 pp |
| VLM early config | Qwen3-VL-8B-Instruct | 198/250 | 200/250 | -0.8 pp |

The text result is the cleanest evidence for the memory design. The VLM result says something different: when the final model can inspect retrieved frames directly, the simple recent baseline is already very strong.

## Known Weak Spots

- Full runs are expensive and depend on local video files.
- The captioners are often the bottleneck, especially for colors, exact counts, and fine object identity.
- SQLite persistence is write-side only; there is no restore/resume path yet.
- The VLM result in `results/` used the older stripped `vlm_streaming` config. The new default VLM preset is `vlm_full`.
- There is no model-free end-to-end smoke runner yet; current tests cover pure logic only.

## More Detail

- `docs/architecture.md` explains the memory and retrieval design.
- `docs/experiments.md` explains how to run experiments and what artifacts are written.
- `results/RESULTS.md` keeps the committed eval summary.
