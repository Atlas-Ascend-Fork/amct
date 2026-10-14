#!/usr/bin/env bash
# Step 1: BF16 baseline evaluation (WikiText2 PPL)
# Usage: bash eval_bf16.sh /path/to/Qwen3-0.6B [npu_id]
set -e

MODEL_DIR=${1:?Usage: bash eval_bf16.sh /path/to/Qwen3-0.6B [npu_id]}
NPU_ID=${2:-0}
SEQ_LEN=4096

python3 -m amct_pytorch.eval \
  --model "${MODEL_DIR}" \
  --model_name qwen3 \
  --seq_len "${SEQ_LEN}" \
  --granularity block \
  --device "npu:${NPU_ID}" \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
