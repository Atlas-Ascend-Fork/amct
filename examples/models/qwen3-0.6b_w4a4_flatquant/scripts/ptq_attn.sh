#!/usr/bin/env bash
# Step 3a: PTQ training for Attention linear layers (flatquant).
# Usage: bash scripts/ptq_attn.sh <MODEL_DIR> <int|mxfp> [NPU_ID] [SEQ_LEN] [DATA_DIR] [OUTPUT_DIR]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(dirname "${SCRIPT_DIR}")"

MODEL_DIR=${1:?Usage: bash scripts/ptq_attn.sh <MODEL_DIR> <int|mxfp> [NPU_ID] [SEQ_LEN] [DATA_DIR] [OUTPUT_DIR]}
QUANT_DTYPE=${2:?Usage: bash scripts/ptq_attn.sh <MODEL_DIR> <int|mxfp> [NPU_ID] [SEQ_LEN] [DATA_DIR] [OUTPUT_DIR]}
NPU_ID=${3:-0}
SEQ_LEN=${4:-4096}
DATA_DIR=${5:-ptq_data/qwen3/attn-linear}
OUTPUT_DIR=${6:-ptq_result/${QUANT_DTYPE}}
BIT_CONFIG="${SAMPLE_DIR}/configs/w4a4_${QUANT_DTYPE}_flatquant.yaml"

echo "==================== Run Configuration ===================="
echo "  step          : PTQ training (attention linear)"
echo "  MODEL_DIR     : ${MODEL_DIR}"
echo "  model_name    : qwen3"
echo "  device        : npu:${NPU_ID}"
echo "  seq_len       : ${SEQ_LEN}"
echo "  granularity   : block"
echo "  quant_dtype   : ${QUANT_DTYPE}"
echo "  algos         : flatquant"
echo "  quant_target  : attn-linear"
echo "  data_dir      : ${DATA_DIR}"
echo "  bit_config    : ${BIT_CONFIG}"
echo "  output_dir    : ${OUTPUT_DIR}"
echo "============================================================"

python3 -m amct_pytorch.ptq \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --data_dir "${DATA_DIR}" \
  --quant_dtype "${QUANT_DTYPE}" \
  --algos flatquant \
  --bit_config "${BIT_CONFIG}" \
  --quant_target attn-linear \
  --output_dir "${OUTPUT_DIR}"
