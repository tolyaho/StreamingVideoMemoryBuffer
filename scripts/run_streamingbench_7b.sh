#!/usr/bin/env bash
# StreamingBench batch eval — same hyperparams as the 3B run, Qwen2.5-7B reasoner.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-outputs/eval_batch_7b}"
REASONER="${REASONER:-Qwen/Qwen2.5-7B-Instruct}"
MANIFEST="${MANIFEST:-data/eval_manifest_50.json}"
EXTRA_ARGS=("$@")

mkdir -p "$OUT_DIR"

source .venv/bin/activate

echo "manifest : $MANIFEST"
echo "output   : $OUT_DIR"
echo "reasoner : $REASONER"
echo "extra    : ${EXTRA_ARGS[*]:-<none>}"

nohup python scripts/eval_batch.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUT_DIR" \
  --reasoner-model "$REASONER" \
  "${EXTRA_ARGS[@]}" \
  > "$OUT_DIR/eval.log" 2>&1 &

echo $! > "$OUT_DIR/eval.pid"
echo "started pid $(cat "$OUT_DIR/eval.pid")"
echo "log → $OUT_DIR/eval.log"
echo "tail -f $OUT_DIR/eval.log"
