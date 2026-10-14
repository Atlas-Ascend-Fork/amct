#!/usr/bin/env bash
# Step 3a: PTQ training for Attention linear layers (flatquant)
# Usage: bash ptq_attn.sh /path/to/Qwen3-0.6B <int|mxfp> [npu_id]
set -e

MODEL_DIR=${1:?Usage: bash ptq_attn.sh /path/to/Qwen3-0.6B <int|mxfp> [npu_id]}
QUANT_DTYPE=${2:?Usage: bash ptq_attn.sh /path/to/Qwen3-0.6B <int|mxfp> [npu_id]}
NPU_ID=${3:-0}
SEQ_LEN=4096

python3 -m amct_pytorch.ptq \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --data_dir ptq_data/qwen3/attn-linear \
  --quant_dtype "${QUANT_DTYPE}" \
  --algos flatquant \
  --bit_config amct_pytorch/configs/w4a4.yaml \
  --quant_target attn-linear \
  --output_dir "ptq_result/${QUANT_DTYPE}"
