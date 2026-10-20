#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

echo "start run test case, please wait ..."
cd ${WORKSPACE}

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 确定要测试的 ops 列表
# ==============================
declare -a ops
ops=("is_finite")
echo $ops

# ==============================
# 运行测试主循环
# ==============================
log "start run test case, please wait ..."

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

smoke_status=0
for op in "${ops[@]}"; do
  echo "Processing: $op"
  mode="eager"
  [ "$op" = "crop_and_resize" ] && mode="graph"
  source /usr/local/Ascend/cann/set_env.sh
  pip3 install onnx modelscope
  echo "bash tests/smoke/run_smoke.sh --model_source modelscope"
  bash tests/smoke/run_smoke.sh --model_source modelscope 2>&1 | tee -a "${WORKSPACE}/run_test.log" || smoke_status=$?
done

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
if [ "${smoke_status}" -eq 0 ] && grep -w -e "AMCT_EXAMPLE_ALL_PASS" "./run_test.log"; then
    echo "$date_time : run test case success"
else
  echo "$date_time : run test case failed"
  exit 1
fi
