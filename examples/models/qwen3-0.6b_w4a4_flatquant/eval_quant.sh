#!/usr/bin/env bash
# Step 4: Evaluate W4A4 fake-quant accuracy (loads attn + mlp PTQ params together)
# Usage: bash eval_quant.sh /path/to/Qwen3-0.6B <int|mxfp> [npu_id]
set -e

MODEL_DIR=${1:?Usage: bash eval_quant.sh /path/to/Qwen3-0.6B <int|mxfp> [npu_id]}
QUANT_DTYPE=${2:?Usage: bash eval_quant.sh /path/to/Qwen3-0.6B <int|mxfp> [npu_id]}
NPU_ID=${3:-0}
SEQ_LEN=4096

python3 -m amct_pytorch.eval \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --eval_mode quant \
  --quant_target attn-linear mlp \
  --quant_dtype "${QUANT_DTYPE}" \
  --bit_config amct_pytorch/configs/w4a4.yaml \
  --algos flatquant \
  --attn_linear_param_dir "ptq_result/${QUANT_DTYPE}/ptq_params/qwen3/attn-linear" \
  --moe_mlp_param_dir "ptq_result/${QUANT_DTYPE}/ptq_params/qwen3/mlp"
