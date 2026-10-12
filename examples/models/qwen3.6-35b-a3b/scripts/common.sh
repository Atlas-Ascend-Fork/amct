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

# Shared defaults for the Qwen3.6-35B-A3B w4a8 autoround sample (issue #182 task 20).
# Prepare CANN / Python / dataset mirrors in the shell before calling a script.
# Override any variable first, e.g. MODEL=/path/to/Qwen3.6-35B-A3B

Qwen36_SAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMCT_ROOT="$(cd "${Qwen36_SAMPLE_DIR}/../../.." && pwd)"

MODEL="${MODEL:-./path/to/Qwen3.6-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-qwen3_6_moe}"
DEVICE="${DEVICE:-npu:0}"
SEQ_LEN="${SEQ_LEN:-4096}"
GRANULARITY="${GRANULARITY:-block}"
NSAMPLES="${NSAMPLES:-128}"
CALI_BSZ="${CALI_BSZ:-4}"
QUANT_DTYPE="${QUANT_DTYPE:-int}"
ALGOS="${ALGOS:-autoround}"
START_BLOCK_IDX="${START_BLOCK_IDX:-0}"
END_BLOCK_IDX="${END_BLOCK_IDX:-40}"
EPOCHS="${EPOCHS:-10}"
BASE_LR="${BASE_LR:-1e-3}"
# MoE overrides: expert-wise autoround walks ~40x256 units, so the MoE run
# accepts its own nsamples/epochs to bound card time; defaults keep the
# same protocol as Attention (128 / EPOCHS).
MOE_NSAMPLES="${MOE_NSAMPLES:-${NSAMPLES}}"
MOE_EPOCHS="${MOE_EPOCHS:-${EPOCHS}}"
MOE_BASE_LR="${MOE_BASE_LR:-${BASE_LR}}"
MOE_CALI_BSZ="${MOE_CALI_BSZ:-${CALI_BSZ}}"
# cosine T_max = epochs * (nsamples // cali_bsz); nsamples < cali_bsz makes T_max 0
# and CosineAnnealingLR raises ZeroDivisionError inside the PTQ loop.
if [ "${MOE_NSAMPLES}" -lt "${MOE_CALI_BSZ}" ]; then
  echo "[qwen3.6-35b-a3b.w4a8.autoround] ERROR: MOE_NSAMPLES=${MOE_NSAMPLES} < MOE_CALI_BSZ=${MOE_CALI_BSZ}." >&2
  echo "  The cosine scheduler needs MOE_NSAMPLES >= MOE_CALI_BSZ (T_max = epochs * (nsamples // cali_bsz))." >&2
  exit 1
fi

# Same in-repo W4A8 policy for int and mxfp; format is selected by --quant_dtype.
BIT_CONFIG="${BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/w4a8.yaml}"
BF16_BIT_CONFIG="${BF16_BIT_CONFIG:-${AMCT_ROOT}/amct_pytorch/configs/bf16.yaml}"
if [ "${QUANT_DTYPE}" = "mxfp" ]; then
  OUT_ROOT="${OUT_ROOT:-${Qwen36_SAMPLE_DIR}/outputs/qwen3_6_35b_a3b_w4a8_mxfp}"
else
  OUT_ROOT="${OUT_ROOT:-${Qwen36_SAMPLE_DIR}/outputs/qwen3_6_35b_a3b_w4a8_int}"
fi
# The BF16 baseline runs once and is shared by both formats, so its logs must not
# follow QUANT_DTYPE (otherwise the mxfp reader would look for it in the int dir).
BF16_OUTPUT_DIR="${BF16_OUTPUT_DIR:-${Qwen36_SAMPLE_DIR}/outputs/qwen3_6_35b_a3b_bf16}"

# PTQ offline data is shared by both formats; extract once, reuse twice.
DATA_DIR="${DATA_DIR:-${Qwen36_SAMPLE_DIR}/outputs/qwen3_6_35b_a3b/ptq_data}"
ATTN_DATA_DIR="${ATTN_DATA_DIR:-${DATA_DIR}/attn-linear}"
MOE_DATA_DIR="${MOE_DATA_DIR:-${DATA_DIR}/moe}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}}"
ATTN_PARAM_DIR="${ATTN_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${MODEL_NAME}/attn-linear}"
MOE_PARAM_DIR="${MOE_PARAM_DIR:-${OUTPUT_DIR}/ptq_params/${MODEL_NAME}/moe}"

AMCT_CLI=(python3 -m)

print_runtime_args() {
  local step="$1"
  echo "[qwen3.6-35b-a3b.w4a8.autoround.${step}] runtime args:"
  cat <<EOF
  AMCT_ROOT=${AMCT_ROOT}
  MODEL=${MODEL}
  MODEL_NAME=${MODEL_NAME}
  DEVICE=${DEVICE}
  SEQ_LEN=${SEQ_LEN}
  GRANULARITY=${GRANULARITY}
  NSAMPLES=${NSAMPLES}
  CALI_BSZ=${CALI_BSZ}
  QUANT_DTYPE=${QUANT_DTYPE}
  ALGOS=${ALGOS}
  BIT_CONFIG=${BIT_CONFIG}
  BF16_BIT_CONFIG=${BF16_BIT_CONFIG}
  DATA_DIR=${DATA_DIR}
  ATTN_DATA_DIR=${ATTN_DATA_DIR}
  MOE_DATA_DIR=${MOE_DATA_DIR}
  OUTPUT_DIR=${OUTPUT_DIR}
  BF16_OUTPUT_DIR=${BF16_OUTPUT_DIR}
  ATTN_PARAM_DIR=${ATTN_PARAM_DIR}
  MOE_PARAM_DIR=${MOE_PARAM_DIR}
  START_BLOCK_IDX=${START_BLOCK_IDX}
  END_BLOCK_IDX=${END_BLOCK_IDX}
  EPOCHS=${EPOCHS}
  BASE_LR=${BASE_LR}
  MOE_NSAMPLES=${MOE_NSAMPLES}
  MOE_EPOCHS=${MOE_EPOCHS}
  MOE_BASE_LR=${MOE_BASE_LR}
  MOE_CALI_BSZ=${MOE_CALI_BSZ}
  RUN_MOE_PTQ=${RUN_MOE_PTQ:-1}
EOF
}
