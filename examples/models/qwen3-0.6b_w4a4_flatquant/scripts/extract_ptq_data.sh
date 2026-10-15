#!/usr/bin/env bash
# Step 2: Extract PTQ calibration data (Pileval, blockwise).
# Extracts attention-linear and MLP targets sequentially.
# Usage: bash scripts/extract_ptq_data.sh <MODEL_DIR> [NPU_ID] [SEQ_LEN] [DATA_DIR]
set -e

MODEL_DIR=${1:?Usage: bash scripts/extract_ptq_data.sh <MODEL_DIR> [NPU_ID] [SEQ_LEN] [DATA_DIR]}
NPU_ID=${2:-0}
SEQ_LEN=${3:-4096}
DATA_DIR=${4:-ptq_data/qwen3}

echo "==================== Run Configuration ===================="
echo "  step        : extract PTQ calibration data"
echo "  MODEL_DIR   : ${MODEL_DIR}"
echo "  model_name  : qwen3"
echo "  device      : npu:${NPU_ID}"
echo "  seq_len     : ${SEQ_LEN}"
echo "  granularity : block"
echo "  calib set   : Pileval (mit-han-lab/pile-val-backup)"
echo "  data_dir    : ${DATA_DIR}/{attn-linear,mlp}"
echo "============================================================"

# Attention linear layers
python3 -m amct_pytorch.extract_ptq_data \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --data_dir "${DATA_DIR}/attn-linear" \
  --quant_target attn-linear

# Dense MLP
python3 -m amct_pytorch.extract_ptq_data \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --data_dir "${DATA_DIR}/mlp" \
  --quant_target mlp

echo "PTQ data extracted. Expected file counts (each 28 for Qwen3-0.6B):"
find "${DATA_DIR}/attn-linear" -maxdepth 1 -name '*_attn_in.pkl' | wc -l
find "${DATA_DIR}/mlp" -maxdepth 1 -name '*_mlp_in.pkl' | wc -l
