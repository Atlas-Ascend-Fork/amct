#!/bin/bash
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
#
# OSPlus SmoothQuant Stage 3：把 stage2 导出的（融合后）BF16 检查点用 RTN 转换为
# Quark 风格 MXFP4，导出可直接部署的 MXFP4 HuggingFace safetensors 检查点。
#
# 本步骤不使用任何校准数据，直接对可量化的 Linear 权重做 round-to-nearest MXFP4
# 量化；MoE 门控 / lm_head 等模块保持原 dtype。config.json 会带上 quark 风格的
# quantization_config。
#
# 输入（BF16_MODEL_DIR）默认指向 stage2 的输出目录，也可显式指向任意可加载的
# BF16 MiniMax-M2.7 检查点。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${SAMPLE_ROOT}/data}"

# stage3 的输入是「融合后的 BF16 模型」，默认取 stage2 的默认输出目录。
DEFAULT_BF16_MODEL_DIR="${DATA_ROOT}/exported/MiniMax-M2.7-osplus_sq_fused_bf16_hf"
BF16_MODEL_DIR="${BF16_MODEL_DIR:-${DEFAULT_BF16_MODEL_DIR}}"

DEFAULT_OUTPUT_DIR="${DATA_ROOT}/exported/MiniMax-M2.7-osplus_sq_mxfp4_hf"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

# 模型加载策略与量化设备。
LOAD_DEVICE_MAP="${LOAD_DEVICE_MAP:-cpu}"
DEVICE_MAP_FILE="${DEVICE_MAP_FILE:-}"
QUANT_DEVICE="${QUANT_DEVICE:-auto}"
MAX_INFLIGHT_JOBS="${MAX_INFLIGHT_JOBS:-32}"
ROW_CHUNK_SIZE="${ROW_CHUNK_SIZE:-512}"

LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/stage3_mxfp4_${TS}.log}"

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
if [[ ! -d "${BF16_MODEL_DIR}" ]]; then
  echo "[fatal] BF16_MODEL_DIR does not exist: ${BF16_MODEL_DIR}" >&2
  echo "        请先运行 run_stage2_bf16.sh 生成融合后的 BF16 检查点，" >&2
  echo "        或通过 BF16_MODEL_DIR=/path/to/bf16_model 显式指定输入。" >&2
  exit 1
fi
if [[ ! -f "${BF16_MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "[fatal] ${BF16_MODEL_DIR} is not a sharded HF checkpoint (missing model.safetensors.index.json)." >&2
  exit 1
fi
if [[ -d "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
  echo "[fatal] OUTPUT_DIR is not empty: ${OUTPUT_DIR}" >&2
  echo "        Choose a fresh directory or delete the existing one." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

DEVICE_MAP_ARGS=()
if [[ -n "${DEVICE_MAP_FILE}" ]]; then
  DEVICE_MAP_ARGS=(--device_map_file "${DEVICE_MAP_FILE}")
fi

echo "=========================================================="
echo "OS+SQ Stage 3 (RTN): fused BF16 -> MXFP4 (Quark-style) export"
echo "=========================================================="
echo "BF16_MODEL_DIR   = ${BF16_MODEL_DIR}"
echo "OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "LOAD_DEVICE_MAP  = ${LOAD_DEVICE_MAP}"
echo "DEVICE_MAP_FILE  = ${DEVICE_MAP_FILE:-<none>}"
echo "QUANT_DEVICE     = ${QUANT_DEVICE}"
echo "MAX_INFLIGHT_JOBS= ${MAX_INFLIGHT_JOBS}"
echo "ROW_CHUNK_SIZE   = ${ROW_CHUNK_SIZE}"
echo "LOG_FILE         = ${LOG_FILE}"
echo "=========================================================="

python3 "${SAMPLE_ROOT}/src/export_rtn_mxfp4.py" \
  --model_dir "${BF16_MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --load_device_map "${LOAD_DEVICE_MAP}" \
  --quant_device "${QUANT_DEVICE}" \
  --max_inflight_jobs "${MAX_INFLIGHT_JOBS}" \
  --row_chunk_size "${ROW_CHUNK_SIZE}" \
  "${DEVICE_MAP_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo
echo "Done. output_dir=${OUTPUT_DIR}"
echo "      log_file=${LOG_FILE}"
echo "      shards: $(ls "${OUTPUT_DIR}"/model-*-of-*.safetensors 2>/dev/null | wc -l)"
echo "      total size: $(du -sh "${OUTPUT_DIR}" 2>/dev/null | awk '{print $1}')"
