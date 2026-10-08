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
# Issue template keeps the name ptq_mlp.sh. DeepSeek-V4-Flash has no dense MLP.
# Do NOT run expert-wise MoE PTQ by default: 43 layers x 256 routed experts is
# far beyond a community-sample budget on one 64GB NPU. MoE is still covered
# at eval time via --quant_target moe (w4a8 fake-quant, no per-expert params).
#
# To force full MoE lwc+lac anyway: RUN_MOE_PTQ=1 bash ptq_mlp.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
print_runtime_args "ptq_moe_skip_or_run"

if [ "${RUN_MOE_PTQ:-0}" != "1" ]; then
  cat <<EOF
[deepseek-v4-flash-w4a8.lwc_lac.ptq_moe] SKIPPED by default.
  Reason: expert-wise lwc+lac is 43*256 PTQ units. One Attention unit
          already takes ~42 min; scaling that to every expert is weeks.
  Workflow: PTQ attn only (ptq_attn.sh), then eval_quant.sh with
            --quant_target attn-linear moe and --attn_linear_param_dir set.
  Override: RUN_MOE_PTQ=1 bash $0
EOF
  exit 0
fi

# shellcheck disable=SC2086
"${AMCT_CLI[@]}" amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name "${MODEL_NAME}" \
  --seq_len "${SEQ_LEN}" \
  --granularity "${GRANULARITY}" \
  --device "${DEVICE}" \
  --data_dir "${MOE_DATA_DIR}" \
  --quant_dtype "${QUANT_DTYPE}" \
  --algos ${ALGOS} \
  --bit_config "${BIT_CONFIG}" \
  --quant_target moe \
  --nsamples "${NSAMPLES}" \
  --cali_bsz "${CALI_BSZ}" \
  --start_block_idx "${START_BLOCK_IDX}" \
  --end_block_idx "${END_BLOCK_IDX}" \
  --epochs "${EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --output_dir "${OUTPUT_DIR}"
