#!/usr/bin/env bash
#
# Continue unified GRPO over the 324 hard maps, in 20-update chunks.
#
# Default: start after chunk1 (offset 20), using the chunk1 checkpoint and
# metrics. Each chunk saves a checkpoint and appends to the shared metrics
# JSON. No evaluation is run here. Run the final evaluation separately.
#
# Usage:
#   ./scripts/run_grpo_unified.sh
#   CUDA_VISIBLE_DEVICES=2 ./scripts/run_grpo_unified.sh
#   START_OFFSET=100 PREV_CHUNK=6 ./scripts/run_grpo_unified.sh
#
# If a chunk is interrupted, inspect /tmp/opencode/grpo_unified_chunk* and
# restart with START_OFFSET/PREV_CHUNK pointing at the last completed chunk.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/data/hjt1/anaconda3/envs/minimind/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Triton/ptxas and cache settings used by all prior successful runs.
# Without these, a stale per-user Triton cache can produce:
#   RuntimeError: Triton Error [CUDA]: device kernel image is invalid
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/data/hjt1/anaconda3/envs/cuda_learn/bin/ptxas}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/opencode/triton_cache_grpo}"
export TRITON_DUMP_DIR="${TRITON_DUMP_DIR:-/tmp/opencode/triton_dump}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/opencode/cache}"
export TORCH_HOME="${TORCH_HOME:-/tmp/opencode/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_DUMP_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME"

POOL="${POOL:-data/grpo/l1_l4_pool.json}"
POOL_SIZE="${POOL_SIZE:-324}"
CHUNK_UPDATES="${CHUNK_UPDATES:-20}"
G="${G:-8}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
LR="${LR:-1e-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
HISTORY_WINDOW="${HISTORY_WINDOW:-12}"

START_OFFSET="${START_OFFSET:-20}"
PREV_CHUNK="${PREV_CHUNK:-1}"
PREV_DIR="${PREV_DIR:-/tmp/opencode/grpo_unified_chunk${PREV_CHUNK}}"
METRICS="${METRICS:-/tmp/opencode/grpo_unified_progress.json}"
OUT_BASE="${OUT_BASE:-/tmp/opencode}"

if [[ ! -f "$POOL" ]]; then
  echo "error: missing pool: $POOL" >&2
  exit 1
fi
if [[ ! -f "$PREV_DIR/grpo_agent.pt" ]]; then
  echo "error: missing previous checkpoint: $PREV_DIR/grpo_agent.pt" >&2
  exit 1
fi
if [[ ! -f "$METRICS" ]]; then
  SRC_METRICS="${SRC_METRICS:-/tmp/opencode/grpo_unified_chunk1.json}"
  if [[ ! -f "$SRC_METRICS" ]]; then
    echo "error: missing initial metrics: $SRC_METRICS" >&2
    exit 1
  fi
  cp "$SRC_METRICS" "$METRICS"
  echo "metrics -> $METRICS (copied from chunk1)"
fi

echo "============================================================"
echo "unified GRPO continuation"
echo "  python:        $PYTHON"
echo "  GPU:           $CUDA_VISIBLE_DEVICES"
echo "  pool:          $POOL ($POOL_SIZE maps)"
echo "  start offset:  $START_OFFSET"
echo "  prev dir:      $PREV_DIR"
echo "  metrics:       $METRICS"
echo "  chunk updates: $CHUNK_UPDATES"
echo "============================================================"

PREV="$PREV_DIR"

for ((offset = START_OFFSET; offset < POOL_SIZE; offset += CHUNK_UPDATES)); do
  chunk=$((offset / CHUNK_UPDATES + 1))
  remain=$((POOL_SIZE - offset))
  updates=$((remain < CHUNK_UPDATES ? remain : CHUNK_UPDATES))
  out="${OUT_BASE}/grpo_unified_chunk${chunk}"
  mkdir -p "$out"

  echo
  echo "------------------------------------------------------------"
  echo "chunk ${chunk}: offset=${offset} updates=${updates}"
  echo "init: $PREV/grpo_agent.pt"
  echo "out:  $out"
  echo "------------------------------------------------------------"

  "$PYTHON" scripts/train_grpo_agent.py \
    --init-adapter "$PREV/grpo_agent.pt" \
    --pool "$POOL" \
    --offset "$offset" \
    --updates "$updates" \
    --g "$G" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --lr "$LR" \
    --history-window "$HISTORY_WINDOW" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --no-eval \
    --save-dir "$out" \
    --json "$METRICS" \
    --resume-metrics \
    2>&1 | tee "${out}/train.log"

  PREV="$out"
done

FINAL_CHUNK=$(( (POOL_SIZE - 1) / CHUNK_UPDATES + 1 ))
FINAL_CKPT="${OUT_BASE}/grpo_unified_chunk${FINAL_CHUNK}/grpo_agent.pt"

echo
echo "============================================================"
echo "unified GRPO complete"
echo "final checkpoint: $FINAL_CKPT"
echo "metrics:          $METRICS"
echo "============================================================"
echo
echo "Run the final evaluation separately:"
echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \\"
echo "  $PYTHON scripts/sft_agent.py \\"
echo "    --eval-only \\"
echo "    --init-adapter $FINAL_CKPT \\"
echo "    --levels L0,L1,L2,L3,L4 \\"
echo "    --val-limit 100 --test-limit 100 \\"
echo "    --eval-batch-size 4 \\"
echo "    --json /tmp/opencode/grpo_unified_full_eval.json"
