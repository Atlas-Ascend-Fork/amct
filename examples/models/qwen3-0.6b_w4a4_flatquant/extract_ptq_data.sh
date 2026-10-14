#!/usr/bin/env bash
# Step 2: Extract PTQ calibration data (blockwise, saved under ptq_data/)
# Usage: bash extract_ptq_data.sh /path/to/Qwen3-0.6B [npu_id]
# Note: quant_target is processed one at a time; this script extracts both
#       attn-linear and mlp sequentially. The extracted data is independent
#       of the PTQ algorithm / quantization format and can be reused.
set -e

MODEL_DIR=${1:?Usage: bash extract_ptq_data.sh /path/to/Qwen3-0.6B [npu_id]}
NPU_ID=${2:-0}
SEQ_LEN=4096

# Attention linear layers
python3 -m amct_pytorch.extract_ptq_data \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --data_dir ptq_data/qwen3/attn-linear \
  --quant_target attn-linear

# Dense MLP
python3 -m amct_pytorch.extract_ptq_data \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --data_dir ptq_data/qwen3/mlp \
  --quant_target mlp

echo "PTQ data extracted. Expected file counts (each 28 for Qwen3-0.6B):"
find ptq_data/qwen3/attn-linear -maxdepth 1 -name '*_attn_in.pkl' | wc -l
find ptq_data/qwen3/mlp -maxdepth 1 -name '*_mlp_in.pkl' | wc -l
