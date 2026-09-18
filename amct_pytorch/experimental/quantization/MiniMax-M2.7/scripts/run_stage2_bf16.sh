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
# OSPlus SmoothQuant Stage 2（BF16-only）：逐层应用 stage1 搜出的等价缩放 scale，
# 在 fp32 下完成融合后 cast 回 BF16，导出融合后的 BF16 HuggingFace 检查点
# （不做 MXFP4 量化）。
#
# 输出是一份不带任何 quantization_config 的纯 BF16 checkpoint，可作为普通 BF16
# 模型加载，用途：
#   - 部署前检查 / 评测平滑后的权重；
#   - 与原始 BF16 模型比对，校验 OSPlus SmoothQuant 等价变换的数学正确性
#     （输出应在 BF16 舍入误差范围内一致）；
#   - 作为后续量化（MXFP4 / W4A8 / W8A8 / INT4 ...）的标准 BF16 起点，无需重跑 stage1。
#
# 依赖：
#   * MODEL_DIR 下的 BF16 参考模型（MiniMax-M2.7-bf16 约 427 GB）；
#   * RECORD_DIR 下 stage1 产出的 scale（62 个 attn + 62 个 oproj .pt 文件）；
#   * 约 430 GB 空闲内存（融合期间整份 BF16 模型驻留 CPU 内存）；
#   * OUTPUT_DIR 下约 430 GB 空闲磁盘。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 记录用户显式设置的 OMP/MKL 线程数，后续若用户未指定则回退到自动探测的默认值。
_USER_OMP_OVERRIDE="${OMP_NUM_THREADS:-__unset__}"
_USER_MKL_OVERRIDE="${MKL_NUM_THREADS:-__unset__}"



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:?请通过环境变量指定 MiniMax-M2.7 BF16 模型目录，例如 MODEL_DIR=/model/MiniMax-M2.7-bf16}"

# Stage 1 output. Default mirrors run_stage1.sh's naming convention
# (data/record_data/hf_npu_calib_${NUM_CALIB_DATA}_${SEQ_LEN}_bs${BATCH_SIZE}).
# In practice users often point RECORD_DIR at one of the curated dirs under
# data/record_data/ (e.g. hf_npu_aligned), so override via env when needed.
NUM_CALIB_DATA="${NUM_CALIB_DATA:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATA_ROOT="${DATA_ROOT:-${SAMPLE_ROOT}/data}"
DEFAULT_RECORD_DIR="${DATA_ROOT}/record_data/hf_npu_calib_${NUM_CALIB_DATA}_${SEQ_LEN}_bs${BATCH_SIZE}"
RECORD_DIR="${RECORD_DIR:-${DEFAULT_RECORD_DIR}}"

DEFAULT_OUTPUT_DIR="${DATA_ROOT}/exported/MiniMax-M2.7-osplus_sq_fused_bf16_hf"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

# tokenizer / config / modeling 等随权重导出的辅助文件来源目录。
# 默认留空，此时 stage2 会直接从 MODEL_DIR（BF16 模型目录本身即含这些文件）复制；
# 仅当上述文件不在 MODEL_DIR 时，才需显式指向一个包含它们的目录。
MODEL_FILES_DIR="${MODEL_FILES_DIR:-${MODEL_DIR}}"

LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/stage2_export_bf16_${TS}.log}"

# ---------------------------------------------------------------------------
# CPU parallelism. BF16 export is mostly memcpy + dtype-cast (fp32 fusion
# arithmetic + bf16 clone), much less CPU-bound than the MXFP4 path, but a
# handful of threads still helps the fp32 fold and safetensors write. Cap
# at 16 (plateau observed on this host).
# ---------------------------------------------------------------------------
_HARDWARE_CPUS="$(getconf _NPROCESSORS_CONF 2>/dev/null || echo 1)"
_cg_cpus=0
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r _quota _period < /sys/fs/cgroup/cpu.max
  if [[ "${_quota}" != "max" && "${_period}" -gt 0 ]]; then
    _cg_cpus=$(( _quota / _period ))
  fi
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
  _quota="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
  _period="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
  if [[ "${_quota}" -gt 0 && "${_period}" -gt 0 ]]; then
    _cg_cpus=$(( _quota / _period ))
  fi
fi
if (( _cg_cpus > 0 && _cg_cpus < _HARDWARE_CPUS )); then
  _HARDWARE_CPUS=${_cg_cpus}
fi
_DEFAULT_THREADS=16
if (( _HARDWARE_CPUS < _DEFAULT_THREADS )); then
  _DEFAULT_THREADS=${_HARDWARE_CPUS}
fi
if [[ "${_USER_OMP_OVERRIDE}" != "__unset__" ]]; then
  export OMP_NUM_THREADS="${_USER_OMP_OVERRIDE}"
else
  export OMP_NUM_THREADS="${_DEFAULT_THREADS}"
fi
if [[ "${_USER_MKL_OVERRIDE}" != "__unset__" ]]; then
  export MKL_NUM_THREADS="${_USER_MKL_OVERRIDE}"
else
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[fatal] MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "[fatal] ${MODEL_DIR} is not a sharded HF checkpoint (missing model.safetensors.index.json)." >&2
  exit 1
fi
if [[ ! -d "${RECORD_DIR}" ]]; then
  echo "[fatal] RECORD_DIR does not exist: ${RECORD_DIR}" >&2
  echo "        Run run_stage1.sh (and run_search_parallel.sh if record-only) first," >&2
  echo "        or override RECORD_DIR=/path/to/scales when invoking this script." >&2
  exit 1
fi

SCALE_COUNT=$(ls "${RECORD_DIR}"/layer_*_attn_scale.pt 2>/dev/null | wc -l)
OPROJ_COUNT=$(ls "${RECORD_DIR}"/layer_*_oproj_scale.pt 2>/dev/null | wc -l)
if [[ "${SCALE_COUNT}" -lt 62 || "${OPROJ_COUNT}" -lt 62 ]]; then
  echo "[fatal] Stage 1 output incomplete: found ${SCALE_COUNT} attn / ${OPROJ_COUNT} oproj scales (need 62 each)." >&2
  echo "        RECORD_DIR=${RECORD_DIR}" >&2
  exit 1
fi

if [[ -d "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
  echo "[fatal] OUTPUT_DIR is not empty: ${OUTPUT_DIR}" >&2
  echo "        Choose a fresh directory or delete the existing one." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo "OS+SQ Stage 2 (HF / NPU port, BF16-only): apply scales -> export BF16"
echo "=========================================================="
echo "MODEL_DIR        = ${MODEL_DIR}"
echo "RECORD_DIR       = ${RECORD_DIR}    (${SCALE_COUNT} attn / ${OPROJ_COUNT} oproj scales)"
echo "OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "MODEL_FILES_DIR  = ${MODEL_FILES_DIR}"
echo "LOG_FILE         = ${LOG_FILE}"
echo "OMP_NUM_THREADS  = ${OMP_NUM_THREADS}    (host has ${_HARDWARE_CPUS} usable CPUs)"
echo "=========================================================="

python3 "${SAMPLE_ROOT}/src/stage2_export_bf16.py" \
  --model_dir "${MODEL_DIR}" \
  --record_dir "${RECORD_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_files_dir "${MODEL_FILES_DIR}" \
  2>&1 | tee "${LOG_FILE}"

echo
echo "Done. output_dir=${OUTPUT_DIR}"
echo "      log_file=${LOG_FILE}"
echo "      shards: $(ls "${OUTPUT_DIR}"/model-*-of-*.safetensors 2>/dev/null | wc -l)"
echo "      total size: $(du -sh "${OUTPUT_DIR}" 2>/dev/null | awk '{print $1}')"
