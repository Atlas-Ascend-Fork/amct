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
print_runtime_args "ptq_mlp"

# This sample quantizes BOTH Attention and MoE by default (issue #182 requirement:
# 量化范围覆盖 Attention 和 MLP/MoE). Set RUN_MOE_PTQ=0 only for a quick smoke test;
# the result table in README assumes full Attention + MoE PTQ.
# MoE autoround walks ~40x256 expert units and dominates the wall time on 1 NPU;
# split blocks across NPUs as in examples/ptq_multi_npu.sh to shorten it.
if [ "${RUN_MOE_PTQ:-1}" != "1" ]; then
  echo "[qwen3.6-35b-a3b.w4a8.autoround.ptq_mlp] SKIPPED (RUN_MOE_PTQ=0)."
  echo "[qwen3.6-35b-a3b.w4a8.autoround.ptq_mlp] NOTE: smoke test only. eval_quant.sh still passes"
  echo "  --quant_target attn-linear moe and --moe_mlp_param_dir; with the MoE params missing the"
  echo "  evaluation falls back to direct quantization with a debug-level notice only, so drop the"
  echo "  moe target for smoke-test evaluation. The README result table assumes full MoE PTQ."
  exit 0
fi

echo "[qwen3.6-35b-a3b.w4a8.autoround.ptq_mlp] start: $(date '+%F %T')"
# shellcheck disable=SC2086
"${AMCT_CLI[@]}" amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name "${MODEL_NAME}" \
  --seq_len "${SEQ_LEN}" \
  --granularity "${GRANULARITY}" \
  --nsamples "${MOE_NSAMPLES}" \
  --cali_bsz ${MOE_CALI_BSZ} \
  --device "${DEVICE}" \
  --data_dir "${MOE_DATA_DIR}" \
  --quant_dtype "${QUANT_DTYPE}" \
  --algos ${ALGOS} \
  --bit_config "${BIT_CONFIG}" \
  --quant_target moe \
  --start_block_idx "${START_BLOCK_IDX}" \
  --end_block_idx "${END_BLOCK_IDX}" \
  --epochs "${MOE_EPOCHS}" \
  --base_lr "${MOE_BASE_LR}" \
  --output_dir "${OUTPUT_DIR}"
echo "[qwen3.6-35b-a3b.w4a8.autoround.ptq_mlp] end: $(date '+%F %T')"
