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
# DeepSeek-V4-Flash W4A4 (lwc + lac) - step 3/4a: PTQ for the attention target.
#
# The bit-width config is the repo-level amct_pytorch/configs/w4a4.yaml;
# the format (int/mxfp) is selected via QUANT_DTYPE.
# DeepSeek-V4 is a MoE model: the MLP/MoE side is covered by ptq_moe.sh
# (the CLI rejects quant_target=mlp for this model).

set -euo pipefail

# Anchor paths to the repo root / sample dir so the script runs from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMCT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-/path/to/deepseek-v4-flash}
OUTPUT_DIR=${OUTPUT_DIR:-${SAMPLE_DIR}/outputs/dsv4_flash_w4a4}
QUANT_DTYPE=${QUANT_DTYPE:-int}             # int | mxfp
BIT_CONFIG=${BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/w4a4.yaml}
DATA_DIR=${DATA_DIR:-${OUTPUT_DIR}/ptq_data/attn-linear}
PARAM_DIR=${PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${QUANT_DTYPE}/attn-linear}
EPOCHS=${EPOCHS:-15}
BASE_LR=${BASE_LR:-1e-3}
SEQ_LEN=${SEQ_LEN:-4096}
CALI_BSZ=${CALI_BSZ:-1}
DEVICE=${DEVICE:-npu:0}

echo "=== ptq_attn.sh ==="
echo "MODEL_PATH=${MODEL_PATH}"
echo "QUANT_DTYPE=${QUANT_DTYPE}"
echo "BIT_CONFIG=${BIT_CONFIG}"
echo "DATA_DIR=${DATA_DIR}"
echo "PARAM_DIR=${PARAM_DIR}"
echo "EPOCHS=${EPOCHS}"
echo "BASE_LR=${BASE_LR}"
echo "CALI_BSZ=${CALI_BSZ}"

python -m amct_pytorch.ptq \
    --trust_remote_code \
    --model "${MODEL_PATH}" \
    --model_name deepseek_v4 \
    --seq_len "${SEQ_LEN}" \
    --granularity block \
    --device "${DEVICE}" \
    --quant_target attn-linear \
    --quant_dtype "${QUANT_DTYPE}" \
    --bit_config "${BIT_CONFIG}" \
    --algos lwc lac \
    --base_lr "${BASE_LR}" \
    --data_dir "${DATA_DIR}" \
    --cali_bsz "${CALI_BSZ}" \
    --epochs "${EPOCHS}" \
    --attn_linear_param_dir "${PARAM_DIR}" \
    --output_dir "${OUTPUT_DIR}/03_ptq_attn_${QUANT_DTYPE}"
