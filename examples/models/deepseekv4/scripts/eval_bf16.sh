#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# DeepSeek-V4-Flash W4A4 (lwc + lac) - step 1/4: BF16 baseline eval.

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/path/to/deepseek-v4-flash}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/dsv4_flash_w4a4}
BIT_CONFIG=${BIT_CONFIG:-amct_pytorch/configs/bf16.yaml}
SEQ_LEN=${SEQ_LEN:-4096}
DEVICE=${DEVICE:-npu:0}

echo "=== eval_bf16.sh ==="
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BIT_CONFIG=${BIT_CONFIG}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "DEVICE=${DEVICE}"

python -m amct_pytorch.eval \
    --model "${MODEL_PATH}" \
    --model_name deepseek_v4 \
    --seq_len "${SEQ_LEN}" \
    --granularity block \
    --device "${DEVICE}" \
    --eval_mode bf16 \
    --bit_config "${BIT_CONFIG}" \
    --output_dir "${OUTPUT_DIR}/01_baseline_eval"
