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

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:?请通过环境变量指定 MiniMax-M2.7 BF16 模型目录，例如 MODEL_DIR=/model/MiniMax-M2.7-bf16}"
CALIB_DATA="${CALIB_DATA:?请通过环境变量指定校准数据 jsonl 文件，例如 CALIB_DATA=/data/minimax/calib.json}"

NUM_CALIB_DATA="${NUM_CALIB_DATA:-512}"
SEQ_LEN="${SEQ_LEN:-32768}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-4096}"
LOAD_DEVICE_MAP="${LOAD_DEVICE_MAP:-auto}"

NUM_WORKERS="${NUM_WORKERS:-16}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-20}"
NUMA_NODES="${NUMA_NODES:-8}"
CORES_PER_NUMA="${CORES_PER_NUMA:-40}"
NUM_LAYERS="${NUM_LAYERS:-62}"

DATA_ROOT="${DATA_ROOT:-${SAMPLE_ROOT}/data}"
DEFAULT_RECORD_DIR="${DATA_ROOT}/record_data/hf_npu_calib_${NUM_CALIB_DATA}_${SEQ_LEN}_bs${BATCH_SIZE}"
RECORD_DIR="${RECORD_DIR:-${DEFAULT_RECORD_DIR}}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs}"

RECORD_ONLY="${RECORD_ONLY:-1}"
RESUME_FLAG=""
if [[ "${RESUME:-1}" == "1" ]]; then
  RESUME_FLAG="--resume"
fi

mkdir -p "${RECORD_DIR}" "${LOG_DIR}"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/stage1_hf_npu_${NUM_CALIB_DATA}_${SEQ_LEN}_bs${BATCH_SIZE}_${TS}.log}"

export ACCELERATE_TORCH_DEVICE="${ACCELERATE_TORCH_DEVICE:-npu}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"

echo "=========================================================="
echo "OS+SQ Stage 1 (record on NPU, scale search on CPU)"
echo "=========================================================="
echo "MODEL_DIR=${MODEL_DIR}"
echo "CALIB_DATA=${CALIB_DATA}"
echo "RECORD_DIR=${RECORD_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "NUM_CALIB_DATA=${NUM_CALIB_DATA}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_TOKENS_PER_LAYER=${MAX_TOKENS_PER_LAYER}"
echo "LOAD_DEVICE_MAP=${LOAD_DEVICE_MAP}"
echo "RECORD_ONLY=${RECORD_ONLY}"
echo "RESUME=${RESUME:-1}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "THREADS_PER_WORKER=${THREADS_PER_WORKER}"
echo "NUMA_NODES=${NUMA_NODES}"
echo "CORES_PER_NUMA=${CORES_PER_NUMA}"
echo "=========================================================="

python3 "${SAMPLE_ROOT}/src/stage1_calibrate.py" \
  --model_dir "${MODEL_DIR}" \
  --calib_data "${CALIB_DATA}" \
  --record_dir "${RECORD_DIR}" \
  --seq_len "${SEQ_LEN}" \
  --num_calib_data "${NUM_CALIB_DATA}" \
  --batch_size "${BATCH_SIZE}" \
  --max_tokens_per_layer "${MAX_TOKENS_PER_LAYER}" \
  --load_device_map "${LOAD_DEVICE_MAP}" \
  ${RESUME_FLAG} --record_only \
  2>&1 | tee "${LOG_FILE}"

if [[ "${RECORD_ONLY}" == "1" ]]; then
  echo
  echo "Done (record_only). record_dir=${RECORD_DIR}"
  echo "                    log_file=${LOG_FILE}"
  exit 0
fi

WORKERS_PER_NUMA=$(( NUM_WORKERS / NUMA_NODES ))
if (( WORKERS_PER_NUMA < 1 )); then WORKERS_PER_NUMA=1; fi
SEARCH_LOG_DIR="${RECORD_DIR}/search_logs"
mkdir -p "${SEARCH_LOG_DIR}"
DRIVER_LOG="${SEARCH_LOG_DIR}/driver_${TS}.log"

python3 - "${NUM_LAYERS}" "${NUM_WORKERS}" > "${SEARCH_LOG_DIR}/_layer_chunks.txt" <<'PY'
import sys
n_layers, n_workers = int(sys.argv[1]), int(sys.argv[2])
per = -(-n_layers // n_workers)
for w in range(n_workers):
    a = w * per
    b = min((w + 1) * per, n_layers) - 1
    if a > b:
        continue
    print(f"{w}\t{a}-{b}")
PY

echo "==========================================================" | tee -a "${DRIVER_LOG}"
echo "OS+SQ scale-search (CPU, ${NUM_WORKERS}-way parallel, ${THREADS_PER_WORKER} threads/worker)" | tee -a "${DRIVER_LOG}"
echo "==========================================================" | tee -a "${DRIVER_LOG}"

pids=()
while IFS=$'\t' read -r worker_id span; do
  [[ -z "${worker_id}" ]] && continue

  numa_node=$(( worker_id / WORKERS_PER_NUMA ))
  pos_in_node=$(( worker_id % WORKERS_PER_NUMA ))
  cpu_start=$(( numa_node * CORES_PER_NUMA + pos_in_node * THREADS_PER_WORKER ))
  cpu_end=$(( cpu_start + THREADS_PER_WORKER - 1 ))

  worker_log="${SEARCH_LOG_DIR}/worker_${worker_id}_layers_${span}.log"
  echo "[launch] worker ${worker_id}  numa=${numa_node}  cpus=${cpu_start}-${cpu_end}  layers=${span}  -> ${worker_log}" \
      | tee -a "${DRIVER_LOG}"

  ASCEND_RT_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=${THREADS_PER_WORKER} \
  MKL_NUM_THREADS=${THREADS_PER_WORKER} \
  OPENBLAS_NUM_THREADS=${THREADS_PER_WORKER} \
  numactl --physcpubind=${cpu_start}-${cpu_end} --membind=${numa_node} \
    nohup python3 "${SAMPLE_ROOT}/src/run_search_from_cache.py" \
      --model_dir "${MODEL_DIR}" \
      --record_dir "${RECORD_DIR}" \
      --search_device cpu \
      --resume 1 \
      --layers "${span}" \
      > "${worker_log}" 2>&1 &
  pids+=($!)
done < "${SEARCH_LOG_DIR}/_layer_chunks.txt"

echo "Spawned ${#pids[@]} workers: ${pids[*]}" | tee -a "${DRIVER_LOG}"
echo "Waiting for all to finish..."           | tee -a "${DRIVER_LOG}"

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "[ERROR] worker pid=${pid} exited non-zero" | tee -a "${DRIVER_LOG}"
    failed=1
  fi
done

if [[ "${failed}" == "0" ]]; then
  echo "All workers finished successfully." | tee -a "${DRIVER_LOG}"
else
  echo "Some workers failed; check per-worker logs under ${SEARCH_LOG_DIR}" | tee -a "${DRIVER_LOG}"
  exit 1
fi

echo
echo "Done. record_dir=${RECORD_DIR}"
echo "      log_file=${LOG_FILE}"
echo "      search_logs=${SEARCH_LOG_DIR}"
