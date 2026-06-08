#!/bin/bash
# 모델-zoo E2E 스윕 런처 (NPU 2 고정, resumable, 영속 로그). 하트비트가 이걸 호출.
cd /data/home/l00960396/amct_pruning || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
# 중복 실행 방지
if ps -eo cmd | grep -q "[e]2e_model_zoo_pipeline.py"; then
  echo "이미 스윕 실행 중 — 재가동 생략"; exit 0
fi
export ASCEND_RT_VISIBLE_DEVICES=2
export HF_HUB_DISABLE_XET=1
LOG=/data/home/l00960396/amct_pruning/examples/pruning/results/zoo_run.log
setsid .venv_npu/bin/python examples/pruning/e2e_model_zoo_pipeline.py >> "$LOG" 2>&1 < /dev/null &
echo $! > /tmp/zoo_pid.txt
echo "스윕 시작 (NPU 2) PID=$(cat /tmp/zoo_pid.txt)"
