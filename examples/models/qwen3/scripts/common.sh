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

# Shared defaults for the Qwen3-0.6B w4a8 lwc+lac sample.
# Prepare CANN / Python / dataset mirrors in the shell before calling a script.
# Override any variable first, e.g. MODEL=/path/to/Qwen3-0.6B

Qwen3_SAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMCT_ROOT="$(cd "${Qwen3_SAMPLE_DIR}/../../.." && pwd)"

MODEL="${MODEL:-./path/to/Qwen3-0.6B}"
MODEL_NAME="${MODEL_NAME:-qwen3}"
DEVICE="${DEVICE:-npu:0}"
SEQ_LEN="${SEQ_LEN:-4096}"
GRANULARITY="${GRANULARITY:-block}"
NSAMPLES="${NSAMPLES:-128}"
QUANT_DTYPE="${QUANT_DTYPE:-int}"
ALGOS="${ALGOS:-lwc lac}"
START_BLOCK_IDX="${START_BLOCK_IDX:-0}"
END_BLOCK_IDX="${END_BLOCK_IDX:-28}"
EPOCHS="${EPOCHS:-15}"
BASE_LR="${BASE_LR:-1e-5}"

# Same in-repo W4A8 policy for int and mxfp; format is selected by --quant_dtype.
BIT_CONFIG="${BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/w4a8.yaml}"
BF16_BIT_CONFIG="${BF16_BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/bf16.yaml}"
if [ "${QUANT_DTYPE}" = "mxfp" ]; then
  OUT_ROOT="${OUT_ROOT:-${Qwen3_SAMPLE_DIR}/outputs/qwen3_0_6b_w4a8_mxfp}"
else
  OUT_ROOT="${OUT_ROOT:-${Qwen3_SAMPLE_DIR}/outputs/qwen3_0_6b_w4a8_int}"
fi

DATA_DIR="${DATA_DIR:-${Qwen3_SAMPLE_DIR}/outputs/qwen3_0_6b/ptq_data}"
ATTN_DATA_DIR="${ATTN_DATA_DIR:-${DATA_DIR}/attn-linear}"
MLP_DATA_DIR="${MLP_DATA_DIR:-${DATA_DIR}/mlp}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}}"
ATTN_PARAM_DIR="${ATTN_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${MODEL_NAME}/attn-linear}"
MLP_PARAM_DIR="${MLP_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${MODEL_NAME}/mlp}"

AMCT_CLI=(python3 -m)

print_runtime_args() {
  local step="$1"
  echo "[qwen3.0.6b.w4a8.lwc_lac.${step}] runtime args:"
  cat <<EOF
  AMCT_ROOT=${AMCT_ROOT}
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
  MLP_DATA_DIR=${MLP_DATA_DIR}
  OUTPUT_DIR=${OUTPUT_DIR}
  ATTN_PARAM_DIR=${ATTN_PARAM_DIR}
  MLP_PARAM_DIR=${MLP_PARAM_DIR}
  START_BLOCK_IDX=${START_BLOCK_IDX}
  END_BLOCK_IDX=${END_BLOCK_IDX}
  EPOCHS=${EPOCHS}
  BASE_LR=${BASE_LR}
EOF
}
