#!/usr/bin/env bash
# Step 4: Evaluate W4A4 fake-quant accuracy (loads attn + mlp PTQ params together).
# Usage: bash scripts/eval_quant.sh <MODEL_DIR> <int|mxfp> [NPU_ID] [SEQ_LEN] [OUTPUT_DIR]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(dirname "${SCRIPT_DIR}")"

MODEL_DIR=${1:?Usage: bash scripts/eval_quant.sh <MODEL_DIR> <int|mxfp> [NPU_ID] [SEQ_LEN] [OUTPUT_DIR]}
QUANT_DTYPE=${2:?Usage: bash scripts/eval_quant.sh <MODEL_DIR> <int|mxfp> [NPU_ID] [SEQ_LEN] [OUTPUT_DIR]}
NPU_ID=${3:-0}
SEQ_LEN=${4:-4096}
OUTPUT_DIR=${5:-ptq_result/${QUANT_DTYPE}}
BIT_CONFIG="${SAMPLE_DIR}/configs/w4a4_${QUANT_DTYPE}_flatquant.yaml"
EVAL_SET="WikiText2 (Salesforce/wikitext, wikitext-2-raw-v1, test)"

echo "==================== Run Configuration ===================="
echo "  step              : quantized evaluation (fake quant)"
echo "  MODEL_DIR         : ${MODEL_DIR}"
echo "  model_name        : qwen3"
echo "  device            : npu:${NPU_ID}"
echo "  seq_len           : ${SEQ_LEN}"
echo "  granularity       : block"
echo "  eval_mode         : quant"
echo "  quant_dtype       : ${QUANT_DTYPE}"
echo "  algos             : flatquant"
echo "  quant_target      : attn-linear mlp"
echo "  eval set          : ${EVAL_SET}"
echo "  bit_config        : ${BIT_CONFIG}"
echo "  attn_param_dir    : ${OUTPUT_DIR}/ptq_params/qwen3/attn-linear"
echo "  mlp_param_dir     : ${OUTPUT_DIR}/ptq_params/qwen3/mlp"
echo "============================================================"

python3 -m amct_pytorch.eval \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --eval_mode quant \
  --quant_target attn-linear mlp \
  --quant_dtype "${QUANT_DTYPE}" \
  --bit_config "${BIT_CONFIG}" \
  --algos flatquant \
  --attn_linear_param_dir "${OUTPUT_DIR}/ptq_params/qwen3/attn-linear" \
  --moe_mlp_param_dir "${OUTPUT_DIR}/ptq_params/qwen3/mlp"
