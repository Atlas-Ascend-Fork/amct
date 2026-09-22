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
# DeepSeek-V4-Flash W4A4 (lwc + lac) - step 2/4: PTQ offline data extraction.
#
# extract_ptq_data only supports a single --quant_target per invocation, so run
# this script once per target (QUANT_TARGET=attn-linear, then QUANT_TARGET=moe).
# Each run produces ~22 GiB of calibration data for the default NSAMPLES=16.

set -euo pipefail

# Anchor paths to the sample dir so the script runs from any cwd and the four
# workflow steps keep sharing one OUTPUT_DIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-/path/to/deepseek-v4-flash}
OUTPUT_DIR=${OUTPUT_DIR:-${SAMPLE_DIR}/outputs/dsv4_flash_w4a4}
QUANT_TARGET=${QUANT_TARGET:-attn-linear}   # attn-linear | moe
DATA_DIR=${DATA_DIR:-${OUTPUT_DIR}/ptq_data/${QUANT_TARGET}}
NSAMPLES=${NSAMPLES:-16}
SEQ_LEN=${SEQ_LEN:-4096}
DEVICE=${DEVICE:-npu:0}

echo "=== extract_ptq_data.sh ==="
echo "MODEL_PATH=${MODEL_PATH}"
echo "QUANT_TARGET=${QUANT_TARGET}"
echo "DATA_DIR=${DATA_DIR}"
echo "NSAMPLES=${NSAMPLES}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "DEVICE=${DEVICE}"

python -m amct_pytorch.extract_ptq_data \
    --trust_remote_code \
    --model "${MODEL_PATH}" \
    --model_name deepseek_v4 \
    --seq_len "${SEQ_LEN}" \
    --granularity block \
    --device "${DEVICE}" \
    --quant_target "${QUANT_TARGET}" \
    --nsamples "${NSAMPLES}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}/02_extract_${QUANT_TARGET//-/}"
