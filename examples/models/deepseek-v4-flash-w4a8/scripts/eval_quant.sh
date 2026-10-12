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
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
print_runtime_args "eval_quant"

if [ ! -d "${ATTN_PARAM_DIR}" ] || [ -z "$(ls -A "${ATTN_PARAM_DIR}" 2>/dev/null || true)" ]; then
  echo "ERROR: attn PTQ params missing or empty: ${ATTN_PARAM_DIR}" >&2
  echo "Run: QUANT_DTYPE=${QUANT_DTYPE} bash ${SCRIPT_DIR}/ptq_attn.sh" >&2
  exit 1
fi

EVAL_EXTRA=()
if [ -d "${MOE_PARAM_DIR}" ] && [ -n "$(ls -A "${MOE_PARAM_DIR}" 2>/dev/null || true)" ]; then
  EVAL_EXTRA+=(--moe_mlp_param_dir "${MOE_PARAM_DIR}")
fi

# One 64GB NPU cannot hold a whole quant block (256 routed experts, each weight
# doubled by QuantLinear's cached_eval_weight). `--expert_offload auto` (the
# default) detects this and streams routed experts from host memory; see README
# and upstream Issue #248. Set EXPERT_OFFLOAD=on/off to override.
# shellcheck disable=SC2086
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --expert_offload "${EXPERT_OFFLOAD}" \
  --model "${MODEL}" \
  --model_name "${MODEL_NAME}" \
  --seq_len "${SEQ_LEN}" \
  --granularity "${GRANULARITY}" \
  --device "${DEVICE}" \
  --eval_mode quant \
  --quant_target attn-linear moe \
  --quant_dtype "${QUANT_DTYPE}" \
  --algos ${ALGOS} \
  --bit_config "${BIT_CONFIG}" \
  --attn_linear_param_dir "${ATTN_PARAM_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"}
