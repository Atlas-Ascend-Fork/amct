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
print_runtime_args "ptq_attn"

# shellcheck disable=SC2086
"${AMCT_CLI[@]}" amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name "${MODEL_NAME}" \
  --seq_len "${SEQ_LEN}" \
  --granularity "${GRANULARITY}" \
  --device "${DEVICE}" \
  --data_dir "${ATTN_DATA_DIR}" \
  --quant_dtype "${QUANT_DTYPE}" \
  --algos ${ALGOS} \
  --bit_config "${BIT_CONFIG}" \
  --quant_target attn-linear \
  --start_block_idx "${START_BLOCK_IDX}" \
  --end_block_idx "${END_BLOCK_IDX}" \
  --epochs "${EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --output_dir "${OUTPUT_DIR}"
