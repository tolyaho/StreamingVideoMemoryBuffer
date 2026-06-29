# Experiments

Experiments are configured with Hydra. The config files live under `configs/`.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Smoke Run

Use a single video first. This still loads the real models.

```bash
streamvmb experiment=streamingbench_text dataset.limit=1 output.dir=outputs/smoke_text
```

For a specific StreamingBench sample:

```bash
streamvmb experiment=streamingbench_text dataset.samples='[36]' output.dir=outputs/sample36_text
```

For a specific LVBench key:

```bash
streamvmb experiment=lvbench_text dataset.keys='[16Z-XQh9jhk]' output.dir=outputs/lvbench_16Z
```

## Full Runs

StreamingBench text:

```bash
streamvmb experiment=streamingbench_text output.dir=outputs/streamingbench_text
```

StreamingBench VLM:

```bash
streamvmb experiment=streamingbench_vlm output.dir=outputs/streamingbench_vlm
```

LVBench text:

```bash
streamvmb experiment=lvbench_text output.dir=outputs/lvbench_text
```

## Common Overrides

```bash
streamvmb experiment=streamingbench_text retrieval.top_k=8 retrieval.tau_fraction=0.05
streamvmb experiment=streamingbench_text memory.recent_capacity=40 output.resume=true
streamvmb experiment=streamingbench_text summary=template
```

`summary=template` is useful for plumbing checks, but it is not a meaningful quality run.

## Output Files

Each config-driven run writes:

- `run_config.json` — resolved config used for the run
- `manifest_snapshot.json` — exact manifest entries selected
- `per_video/*.jsonl` — per-question predictions
- `run_log.jsonl` — per-video runtime and score summary
- `summary.json` — aggregate metrics
- `summary.md` — readable aggregate summary
- `metrics.json` — flat metrics for plotting or dashboards

## Old Scripts

The older scripts still work:

```bash
python scripts/eval_batch.py --config streamingbench_tuned
python scripts/eval_lvbench_batch.py --config lvbench_tuned
scripts/run_streamingbench_text.sh
```

Keep them around for old result reproduction. New work should prefer `streamvmb` because it writes a resolved config and manifest snapshot every time.

## Result Caveats

The committed VLM result under `results/streamingbench_vlm_8b` used the early `vlm_streaming` retrieval preset. The config-driven VLM experiment now defaults to the fuller memory path. If you need to reproduce the old run, override:

```bash
streamvmb experiment=streamingbench_vlm retrieval=vlm_streaming
```
