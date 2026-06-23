# StreamingBench evaluation results

Canonical copies of batch eval outputs for the year report. Live runs write to `outputs/` (gitignored); this folder is committed.

**Benchmark:** StreamingBench Real-Time Visual Understanding, 50 videos, 250 MCQs  
**Comparison:** hierarchical memory vs recent-window baseline (same stream, same reasoner)

## Summary table

| Run | Reasoner | Config | Hier | Base | Δ |
|-----|----------|--------|------|------|---|
| [text 3B notebook](streamingbench_text_3b_notebook/) | Qwen2.5-3B-Instruct | `notebook` | **128/250 (51.2%)** | 109/250 (43.6%) | **+7.6 pp** |
| [text 3B tuned](streamingbench_text_3b_tuned/) | Qwen2.5-3B-Instruct | `streamingbench_tuned` | **129/250 (51.6%)** | 112/250 (44.8%) | **+6.8 pp** |
| [VLM 8B](streamingbench_vlm_8b/) | Qwen3-VL-8B-Instruct | `vlm_streaming` | 198/250 (79.2%) | 200/250 (80.0%) | −0.8 pp |

## Headline (text reasoner)

Hierarchical streaming memory improves MCQ accuracy by **~7 percentage points** over a recent-window baseline on StreamingBench with a text reasoner. Largest gains are on **Object Perception** and **Event Understanding** (see per-run `summary.md`).

With a VLM reasoner (~80% accuracy), both systems perform similarly — retrieved frames from the recent buffer are sufficient and episodic memory adds little.

## Folder layout (each run)

| File | Description |
|------|-------------|
| `summary.md` | Human-readable totals + **by task type** |
| `summary.json` | Same stats as JSON |
| `run_config.json` | Model, config name, hyperparameters, reproduce command |
| `run_log.jsonl` | Per-video rollup (windows, time, hier/base correct) |
| `per_video/sample_XX.jsonl` | Per-question predictions (`hierarchical` / `baseline`, `task_type`, `correct`) |

## Reproduce

```bash
source .venv/bin/activate

# Text 3B — notebook (original +7.6 pp run)
python scripts/eval_batch.py --config notebook --output-dir outputs/eval_batch --reasoner-type text

# Text 3B — tuned (+6.8 pp run)
scripts/run_streamingbench_text.sh

# VLM 8B
scripts/run_streamingbench_vlm.sh
```
