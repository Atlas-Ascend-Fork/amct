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

# Shared defaults for Qwen3.6-35B-A3B w4a4 autoround (Issue #182 task 24).
# Prepare CANN / Python / dataset mirrors in the shell before calling a script.
# Override any variable first, e.g. MODEL=/path/to/Qwen3.6-35B-A3B

SAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMCT_ROOT="$(cd "${SAMPLE_DIR}/../../.." && pwd)"

MODEL="${MODEL:-./path/to/Qwen3.6-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-qwen3_6_moe}"
DEVICE="${DEVICE:-npu:0}"
SEQ_LEN="${SEQ_LEN:-4096}"
GRANULARITY="${GRANULARITY:-block}"
NSAMPLES="${NSAMPLES:-128}"
QUANT_DTYPE="${QUANT_DTYPE:-int}"
ALGOS="${ALGOS:-autoround}"
START_BLOCK_IDX="${START_BLOCK_IDX:-0}"
# Exclusive upper bound; clamped to model.num_layers inside PTQ.
END_BLOCK_IDX="${END_BLOCK_IDX:-40}"
EPOCHS="${EPOCHS:-10}"
BASE_LR="${BASE_LR:-1e-3}"

BF16_BIT_CONFIG="${BF16_BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/bf16.yaml}"
# Reuse in-repo bit-width config; int vs mxfp is selected by --quant_dtype.
BIT_CONFIG="${BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/w4a4.yaml}"
if [ "${QUANT_DTYPE}" = "mxfp" ]; then
  OUT_ROOT="${OUT_ROOT:-${SAMPLE_DIR}/outputs/qwen3_6_35b_a3b_w4a4_mxfp}"
else
  OUT_ROOT="${OUT_ROOT:-${SAMPLE_DIR}/outputs/qwen3_6_35b_a3b_w4a4_int}"
fi

DATA_DIR="${DATA_DIR:-${SAMPLE_DIR}/outputs/qwen3_6_35b_a3b/ptq_data}"
ATTN_DATA_DIR="${ATTN_DATA_DIR:-${DATA_DIR}/attn-linear}"
# MoE model: use quant_target=moe (mlp is rejected by the adapter).
MOE_DATA_DIR="${MOE_DATA_DIR:-${DATA_DIR}/moe}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}}"
ATTN_PARAM_DIR="${ATTN_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${MODEL_NAME}/attn-linear}"
MOE_PARAM_DIR="${MOE_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${MODEL_NAME}/moe}"

AMCT_CLI=(python3 -m)

print_runtime_args() {
  local step="$1"
  echo "[qwen3.6-35b-a3b-w4a4.autoround.${step}] runtime args:"
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
  RUN_MOE_PTQ=${RUN_MOE_PTQ:-0}
EOF
}
