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

# Shared defaults for DeepSeek-V4-Flash w4a8 lwc+lac (Issue #182 task 25).
# Prepare CANN / Python / dataset mirrors in the shell before calling a script.
# Override any variable first, e.g. MODEL=/path/to/DeepSeek-V4-Flash

SAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMCT_ROOT="$(cd "${SAMPLE_DIR}/../../.." && pwd)"

MODEL="${MODEL:-./path/to/DeepSeek-V4-Flash}"
MODEL_NAME="${MODEL_NAME:-deepseek_v4}"
DEVICE="${DEVICE:-npu:0}"
SEQ_LEN="${SEQ_LEN:-4096}"
GRANULARITY="${GRANULARITY:-block}"
NSAMPLES="${NSAMPLES:-128}"
QUANT_DTYPE="${QUANT_DTYPE:-int}"
ALGOS="${ALGOS:-lwc lac}"
START_BLOCK_IDX="${START_BLOCK_IDX:-0}"
# Exclusive upper bound; clamped to model.num_layers inside PTQ.
END_BLOCK_IDX="${END_BLOCK_IDX:-43}"
EPOCHS="${EPOCHS:-15}"
BASE_LR="${BASE_LR:-1e-3}"
# 64GB HBM. Default cali_bsz=4 OOMs once fake-quant experts are resident.
CALI_BSZ="${CALI_BSZ:-1}"

BF16_BIT_CONFIG="${BF16_BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/bf16.yaml}"
# Reuse in-repo bit-width config; int vs mxfp is selected by --quant_dtype.
BIT_CONFIG="${BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/w4a8.yaml}"
if [ "${QUANT_DTYPE}" = "mxfp" ]; then
  OUT_ROOT="${OUT_ROOT:-${SAMPLE_DIR}/outputs/deepseek_v4_flash_w4a8_mxfp}"
else
  OUT_ROOT="${OUT_ROOT:-${SAMPLE_DIR}/outputs/deepseek_v4_flash_w4a8_int}"
fi

DATA_DIR="${DATA_DIR:-${SAMPLE_DIR}/outputs/deepseek_v4_flash/ptq_data}"
ATTN_DATA_DIR="${ATTN_DATA_DIR:-${DATA_DIR}/attn-linear}"
MOE_DATA_DIR="${MOE_DATA_DIR:-${DATA_DIR}/moe}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}}"
ATTN_PARAM_DIR="${ATTN_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${MODEL_NAME}/attn-linear}"
MOE_PARAM_DIR="${MOE_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${MODEL_NAME}/moe}"

AMCT_CLI=(python3 -m)

print_runtime_args() {
  local step="$1"
  echo "[deepseek-v4-flash-w4a8.lwc_lac.${step}] runtime args:"
  cat <<EOF
  AMCT_ROOT=${AMCT_ROOT}
  SAMPLE_DIR=${SAMPLE_DIR}
  MODEL=${MODEL}
  MODEL_NAME=${MODEL_NAME}
  DEVICE=${DEVICE}
  SEQ_LEN=${SEQ_LEN}
  GRANULARITY=${GRANULARITY}
  NSAMPLES=${NSAMPLES}
  QUANT_DTYPE=${QUANT_DTYPE}
  ALGOS=${ALGOS}
  BIT_CONFIG=${BIT_CONFIG}
  BF16_BIT_CONFIG=${BF16_BIT_CONFIG}
  DATA_DIR=${DATA_DIR}
  ATTN_DATA_DIR=${ATTN_DATA_DIR}
  MOE_DATA_DIR=${MOE_DATA_DIR}
  OUTPUT_DIR=${OUTPUT_DIR}
  ATTN_PARAM_DIR=${ATTN_PARAM_DIR}
  MOE_PARAM_DIR=${MOE_PARAM_DIR}
  START_BLOCK_IDX=${START_BLOCK_IDX}
  END_BLOCK_IDX=${END_BLOCK_IDX}
  EPOCHS=${EPOCHS}
  BASE_LR=${BASE_LR}
  CALI_BSZ=${CALI_BSZ}
  RUN_MOE_PTQ=${RUN_MOE_PTQ:-0}
EOF
}
