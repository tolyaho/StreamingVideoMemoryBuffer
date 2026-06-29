#!/usr/bin/env bash
# StreamingBench text MCQ eval — hierarchical vs baseline (Qwen2.5-3B default).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-outputs/eval_batch_tuned}"
REASONER="${REASONER:-Qwen/Qwen2.5-3B-Instruct}"
MANIFEST="${MANIFEST:-data/eval_manifest_50.json}"
CONFIG="${CONFIG:-streamingbench_tuned}"
EXTRA_ARGS=("$@")

mkdir -p "$OUT_DIR"
source .venv/bin/activate

echo "manifest : $MANIFEST"
echo "output   : $OUT_DIR"
echo "config   : $CONFIG"
echo "reasoner : text ($REASONER)"

nohup python scripts/eval_batch.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUT_DIR" \
  --reasoner-type text \
  --config "$CONFIG" \
  --reasoner-model "$REASONER" \
  "${EXTRA_ARGS[@]}" \
  > "$OUT_DIR/eval.log" 2>&1 &

echo $! > "$OUT_DIR/eval.pid"
echo "pid $(cat "$OUT_DIR/eval.pid") → $OUT_DIR/eval.log"
