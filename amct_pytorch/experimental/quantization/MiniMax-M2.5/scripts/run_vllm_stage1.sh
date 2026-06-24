#!/bin/bash
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_BASENAME="0001-MiniMax-M2-adapt-Ascend-fp8-loading-and-qk-norm-path.patch"

MODEL_DIR="${MODEL_DIR:-/model/MiniMax/MiniMax-M2.5-bf16}"
CALIB_DATA="${CALIB_DATA:-/data.jsonl}"
RECORD_DIR="${RECORD_DIR:-${SCRIPT_DIR}/record_data_vllm}"
NUM_CALIB_DATA="${NUM_CALIB_DATA:-2048}"
SEQ_LEN="${SEQ_LEN:-32768}"
TP_SIZE="${TP_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_REPO_DIR="${VLLM_REPO_DIR:-/vllm-workspace/vllm}"
VLLM_PATCH_PATH="${VLLM_PATCH_PATH:-${REPO_DIR}/patches/${PATCH_BASENAME}}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
DEFAULT_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
VLLM_COMPILATION_CONFIG="${VLLM_COMPILATION_CONFIG:-$DEFAULT_COMPILATION_CONFIG}"

vllm_patch_is_functionally_present() {
  VLLM_REPO_DIR="${VLLM_REPO_DIR}" python3 - <<'PY'
from pathlib import Path
import os
import sys

repo_dir = Path(os.environ["VLLM_REPO_DIR"])
required_snippets = {
    repo_dir / "vllm/config/model.py": "Detected fp8 MiniMax-M2 checkpoint on NPU",
    repo_dir / "vllm/model_executor/layers/mamba/linear_attn.py": "torch.ops.npu.npu_rms_norm",
    repo_dir / "vllm/model_executor/models/minimax_m2.py": "_dequantize_fp8_block_weight",
}

for path, snippet in required_snippets.items():
    try:
        content = path.read_text()
    except FileNotFoundError:
        sys.exit(1)
    if snippet not in content:
        sys.exit(1)

sys.exit(0)
PY
}

mkdir -p "${RECORD_DIR}"

if [[ -d "${VLLM_REPO_DIR}/.git/rebase-apply" ]]; then
  echo "Detected an unfinished git am session in ${VLLM_REPO_DIR} (.git/rebase-apply exists)." >&2
  echo "To avoid disturbing any in-progress git am that may belong to you, this script will not touch it automatically." >&2
  echo "Please resolve it manually with 'git am --continue', 'git am --skip', or 'git am --abort' first, then re-run this script." >&2
  exit 1
fi

if vllm_patch_is_functionally_present; then
  echo "MiniMax Ascend patch functionality already present, skipping git am."
elif git -C "${VLLM_REPO_DIR}" apply --reverse --check "${VLLM_PATCH_PATH}" >/dev/null 2>&1; then
  echo "Patch already applied, skipping git am."
elif git -C "${VLLM_REPO_DIR}" apply --check "${VLLM_PATCH_PATH}" >/dev/null 2>&1; then
  git -C "${VLLM_REPO_DIR}" am "${VLLM_PATCH_PATH}"
else
  echo "Patch does not apply cleanly to the current VLLM checkout, and the required MiniMax Ascend support is not fully present." >&2
  echo "Please rebase ${VLLM_PATCH_PATH} onto $(git -C "${VLLM_REPO_DIR}" rev-parse --short HEAD)." >&2
  exit 1
fi

export PYTHONPATH="${VLLM_REPO_DIR}:${PYTHONPATH:-}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1}"
export VLLM_REPO_DIR
export VLLM_PATCH_PATH
export ENABLE_EXPERT_PARALLEL
export VLLM_MAX_NUM_SEQS
export VLLM_MAX_NUM_BATCHED_TOKENS
export VLLM_COMPILATION_CONFIG
export VLLM_ENFORCE_EAGER

echo "============================================"
echo "Stage 1: Record Activation Statistics (vLLM)"
echo "============================================"
echo "MODEL_DIR=${MODEL_DIR}"
echo "TP_SIZE=${TP_SIZE}"
echo "VLLM_REPO_DIR=${VLLM_REPO_DIR}"
echo "ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL}"
echo "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
echo "VLLM_COMPILATION_CONFIG=${VLLM_COMPILATION_CONFIG}"
torchrun --nproc_per_node="${TP_SIZE}" "${REPO_DIR}/src/stage1_record_activations_vllm.py" \
  --model_dir "${MODEL_DIR}" \
  --calib_data_path "${CALIB_DATA}" \
  --output_dir "${RECORD_DIR}" \
  --num_calib_data "${NUM_CALIB_DATA}" \
  --seq_len "${SEQ_LEN}" \
  --tensor_parallel_size "${TP_SIZE}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --vllm_repo_dir "${VLLM_REPO_DIR}" \
  --vllm_patch_path "${VLLM_PATCH_PATH}" \
  --enable_expert_parallel "${ENABLE_EXPERT_PARALLEL}" \
  --max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
  --max_num_batched_tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  --compilation_config "${VLLM_COMPILATION_CONFIG}" \
  --enforce_eager "${VLLM_ENFORCE_EAGER}" \
  --flashcomm1 "${VLLM_ASCEND_ENABLE_FLASHCOMM1}"

echo
echo "Done."
