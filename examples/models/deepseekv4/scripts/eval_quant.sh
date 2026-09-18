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
# DeepSeek-V4-Flash W4A4 (lwc + lac) - step 4/4: quantized eval with PTQ params.
#
# Loads the attention and MoE PTQ params produced by ptq_attn.sh + ptq_moe.sh
# and evaluates the quantized model on WikiText2.
# The bit-width config is the repo-level amct_pytorch/configs/w4a4.yaml;
# the format (int/mxfp) is selected via QUANT_DTYPE.

set -euo pipefail

# Anchor paths to the repo root / sample dir so the script runs from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMCT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-/path/to/deepseek-v4-flash}
OUTPUT_DIR=${OUTPUT_DIR:-${SAMPLE_DIR}/outputs/dsv4_flash_w4a4}
QUANT_DTYPE=${QUANT_DTYPE:-int}             # int | mxfp
BIT_CONFIG=${BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/w4a4.yaml}
ATTN_PARAM_DIR=${ATTN_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${QUANT_DTYPE}/attn-linear}
MOE_PARAM_DIR=${MOE_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${QUANT_DTYPE}/moe}
SEQ_LEN=${SEQ_LEN:-4096}
DEVICE=${DEVICE:-npu:0}

echo "=== eval_quant.sh ==="
echo "MODEL_PATH=${MODEL_PATH}"
echo "QUANT_DTYPE=${QUANT_DTYPE}"
echo "BIT_CONFIG=${BIT_CONFIG}"
echo "ATTN_PARAM_DIR=${ATTN_PARAM_DIR}"
echo "MOE_PARAM_DIR=${MOE_PARAM_DIR}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "DEVICE=${DEVICE}"

python -m amct_pytorch.eval \
    --trust_remote_code \
    --model "${MODEL_PATH}" \
    --model_name deepseek_v4 \
    --seq_len "${SEQ_LEN}" \
    --granularity block \
    --device "${DEVICE}" \
    --eval_mode quant \
    --quant_target attn-linear moe \
    --quant_dtype "${QUANT_DTYPE}" \
    --bit_config "${BIT_CONFIG}" \
    --algos lwc lac \
    --attn_linear_param_dir "${ATTN_PARAM_DIR}" \
    --moe_mlp_param_dir "${MOE_PARAM_DIR}" \
    --output_dir "${OUTPUT_DIR}/05_eval_quant_${QUANT_DTYPE}"
