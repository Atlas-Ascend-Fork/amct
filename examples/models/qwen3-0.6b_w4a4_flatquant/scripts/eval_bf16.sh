#!/usr/bin/env bash
# Step 1: BF16 baseline evaluation (WikiText2 PPL)
# Usage: bash scripts/eval_bf16.sh <MODEL_DIR> [NPU_ID] [SEQ_LEN]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(dirname "${SCRIPT_DIR}")"

MODEL_DIR=${1:?Usage: bash scripts/eval_bf16.sh <MODEL_DIR> [NPU_ID] [SEQ_LEN]}
NPU_ID=${2:-0}
SEQ_LEN=${3:-4096}
BIT_CONFIG="${SAMPLE_DIR}/configs/w4a4_int_flatquant.yaml"

echo "==================== Run Configuration ===================="
echo "  step        : BF16 baseline evaluation"
echo "  MODEL_DIR   : ${MODEL_DIR}"
echo "  model_name  : qwen3"
echo "  device      : npu:${NPU_ID}"
echo "  seq_len     : ${SEQ_LEN}"
echo "  granularity : block"
echo "  eval_mode   : bf16"
echo "  bit_config  : ${BIT_CONFIG}"
echo "============================================================"

python3 -m amct_pytorch.eval \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --eval_mode bf16 \
  --bit_config "${BIT_CONFIG}"
