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

if ! python -c "import accelerate" 2>/dev/null; then
  echo "accelerate not found, installing..."
  pip install accelerate
fi

MODEL_DIR="${MODEL_DIR:-/model/MiniMax/MiniMax-M2.5-bf16}"
RECORD_DIR="${RECORD_DIR:-${SCRIPT_DIR}/record_data_vllm}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/exported_model_vllm}"
LOAD_DEVICE_MAP="${LOAD_DEVICE_MAP:-auto}"
DEVICE_MAP_FILE="${DEVICE_MAP_FILE:-}"
ALPHA="${ALPHA:-0.8}"

if [[ ! -d "${RECORD_DIR}" ]]; then
  echo "Missing record dir: ${RECORD_DIR}" >&2
  echo "Please run ascend/run_vllm_stage1.sh first or set RECORD_DIR." >&2
  exit 1
fi

if [[ -z "$(ls -A "${RECORD_DIR}")" ]]; then
  echo "Record dir is empty: ${RECORD_DIR}" >&2
  echo "Please generate activation records first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Stage 2: SmoothQuant + MXFP4 + Export (Ascend)"
echo "============================================"
echo "MODEL_DIR=${MODEL_DIR}"
echo "RECORD_DIR=${RECORD_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOAD_DEVICE_MAP=${LOAD_DEVICE_MAP}"
echo "ALPHA=${ALPHA}"

CMD=(
  python "${REPO_DIR}/src/stage2_smoothquant_and_export.py"
  --model_dir "${MODEL_DIR}"
  --record_dir "${RECORD_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --load_device_map "${LOAD_DEVICE_MAP}"
  --alpha "${ALPHA}"
)

if [[ -n "${DEVICE_MAP_FILE}" ]]; then
  CMD+=(--device_map_file "${DEVICE_MAP_FILE}")
fi

"${CMD[@]}"

echo
echo "Done."
