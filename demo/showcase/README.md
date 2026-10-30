# Showcase run records

These are the recorded runs behind the [showcase video](https://github.com/tolyaho/StreamingVideoMemoryBuffer/releases/tag/showcase-v1). The browser viewer in [`demo/`](../README.md) replays them.

| File | What it is |
| --- | --- |
| `trace-notebook.json` | Run A: the unchanged `notebook` preset (the showcase run) |
| `trace-short_clip_demo.json` | Run B: a demo-only preset, `notebook` with `episodic_capacity=8` instead of 100, so long-term consolidation happens inside a 266 s clip |
| `qas.json`, `annotation.json` | The four questions, options, answers and evidence intervals, written from frame inspection before inference |
| `FROZEN_BEFORE_INFERENCE.sha256` | Hashes of those files and of the clip, recorded before any model ran |
| `RUN_PLAN.md` | The run plan written before inference, with a deviation log |
| `compare-*.txt` | Repeat vs first attempt of each run |
| `model-pins.json` | Exact Hugging Face commits used |
| `rights.json` | Licence record for the source clip; its SHA-256 binds the clip to the traces |

**What the traces contain.** Each trace was written by `demo/trace.py` while the real pipeline ran (`demo/capture.py` → `scripts/eval_common.run_video`) on an Apple M4 Pro (MPS). The models were X-CLIP base/32, Florence-2-base, Moondream2, Qwen2.5-VL-3B-Instruct and Qwen2.5-3B-Instruct. Each trace holds:
- every window: its frame thumbnail and Florence-2 caption;
- each promote/discard decision with its similarity;
- every episode and event, with its Moondream2 or Qwen2.5-VL text;
- what the retriever returned for each question, with scores;
- the text given to both reasoners;
- both answers, verbatim.

**Results.**
- Run A: hierarchical memory 2/4, recent-only baseline 1/4.
- Run B: 1/4 each.

The featured question (Q2, the apple) was chosen after the results were known. The plan had named Q3; the reason for the change is logged in `RUN_PLAN.md`. Both runs were repeated after a recorder fix. The first attempts, which `RUN_PLAN.md` refers to as `trace-*.first-attempt.json`, are attached to the release as `svmb-showcase-first-attempts.zip`.

**Reproducibility note.** Moondream2's `query()` samples by default (temperature 0.5, top-p 0.3), and `src/summary_builder.py` does not pass settings, so episode texts differ between runs. Window captions, writer decisions, memory statistics and all Run A answer letters were identical across the two attempts.

To re-capture a run, you need the full model environment and about 18 GB of weights:

```bash
HF_HUB_OFFLINE=1 python demo/capture.py --video demo/showcase/media/bbb-excerpt-0033-0459.webm \
  --qas demo/showcase/qas.json --config notebook --reasoner-type text \
  --rights demo/showcase/rights.json --focus-question bbb_q3_butterfly \
  --pins demo/showcase/model-pins.json --output demo/local/trace-notebook.json
```

`model-pins.json` lists the five models. Moondream2 also loads its tokenizer from `moondream/starmie-v1` (commit `35192e10a54e36eabe0a7cc57a2c1aab371cafc5`).

## Footage

[*Big Buck Bunny*](https://peach.blender.org/) © 2008 Blender Foundation / www.bigbuckbunny.org, licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

The source clip, the trace thumbnails and the showcase video use an **edited excerpt**: film time 0:33–4:59, audio removed, re-encoded to VP9/WebM, 1280×720 at 24 fps. The excerpt starts after the title card. The Blender Foundation does not endorse this project.
