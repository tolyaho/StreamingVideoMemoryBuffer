#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-outputs/eval_batch_vlm}"
REASONER_MODEL="${REASONER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
MANIFEST="${MANIFEST:-data/eval_manifest_50.json}"
EXTRA_ARGS=("$@")

mkdir -p "$OUT_DIR"
source .venv/bin/activate

echo "manifest : $MANIFEST"
echo "output   : $OUT_DIR"
echo "reasoner : vlm ($REASONER_MODEL)"

nohup python scripts/eval_batch.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUT_DIR" \
  --reasoner-type vlm \
  --config vlm_full \
  --reasoner-model "$REASONER_MODEL" \
  "${EXTRA_ARGS[@]}" \
  > "$OUT_DIR/eval.log" 2>&1 &

echo $! > "$OUT_DIR/eval.pid"
echo "pid $(cat "$OUT_DIR/eval.pid") → $OUT_DIR/eval.log"
