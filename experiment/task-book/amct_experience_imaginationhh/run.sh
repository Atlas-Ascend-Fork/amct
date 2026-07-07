#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------
# HiFloat8 量化运行封装脚本。
#
# 作用：补齐 quantize.py 运行所需的 PYTHONPATH（仓库根、amct_ops staging、
# CPU 扩展目录、post4 torch_npu），降低使用门槛。直接把参数透传给 quantize.py。
#
# 用法：
#   bash run.sh --model_path /home/developer/models/Qwen2.5-3B-Instruct
#   bash run.sh --model_path <path> --backend cpu_sim --max_samples 5
#
# 可通过环境变量覆盖默认路径：
#   AMCT_REPO_ROOT   仓库根目录（默认按本脚本位置上溯三级）
#   TORCH_NPU_POST4  post4 torch_npu 的隔离安装目录（默认 /tmp/tnpu_post4）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${AMCT_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TORCH_NPU_POST4="${TORCH_NPU_POST4:-/tmp/tnpu_post4}"

CPU_EXT_DIR="$REPO_ROOT/amct_pytorch/experimental/hifloat8"
OPS_STAGING="$REPO_ROOT/amct_ops/staging"

# post4 torch_npu 优先（提供 hifloat8 dtype），其余依次加入
export PYTHONPATH="$TORCH_NPU_POST4:$SCRIPT_DIR:$REPO_ROOT:$OPS_STAGING:$CPU_EXT_DIR:$PYTHONPATH"

echo "[run.sh] REPO_ROOT      = $REPO_ROOT"
echo "[run.sh] TORCH_NPU_POST4= $TORCH_NPU_POST4"
echo "[run.sh] PYTHONPATH set, launching quantize.py ..."

python3 "$SCRIPT_DIR/quantize.py" "$@"
