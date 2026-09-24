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

# 使用方法:
#   bash tests/smoke/run_smoke.sh [选项]
#
# 参数说明:
#   --model               HuggingFace 模型名或本地路径（默认 Qwen/Qwen3-0.6B）
#                         传本地路径时跳过下载步骤
#   --model_dir           模型下载目标目录（默认 /tmp/amct_verify_model）
#   --device              运行设备（默认 npu:0）
#   --cases               逗号分隔的用例列表，可选 cast/minmax/smoothquant/awq
#                         （默认 cast,minmax,smoothquant,awq，即全部运行）
#   --model_source        模型下载来源：huggingface（默认）或 modelscope
#   --hf_endpoint         HuggingFace 镜像地址（可选，如 https://hf-mirror.com）
#
# 示例:
#   # 默认下载 Qwen3-0.6B 并运行全部用例
#   bash tests/smoke/run_smoke.sh
#
#   # 使用本地模型，只运行 cast 和 minmax
#   bash tests/smoke/run_smoke.sh --model /data/Qwen3-0.6B \
#       --cases cast,minmax --device npu:0
#
#   # 使用镜像加速下载
#   bash tests/smoke/run_smoke.sh --hf_endpoint https://hf-mirror.com
#
#   # 使用 ModelScope 下载（国内推荐）
#   bash tests/smoke/run_smoke.sh --model_source modelscope

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME_OR_PATH="Qwen/Qwen3-0.6B"
MODEL_DIR="/tmp/amct_verify_model"
DEVICE="npu:0"
CASES="cast,minmax,smoothquant,awq"
HF_ENDPOINT=""
MODEL_SOURCE="huggingface"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_NAME_OR_PATH="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --cases)
            CASES="$2"
            shift 2
            ;;
        --model_source)
            MODEL_SOURCE="$2"
            shift 2
            ;;
        --hf_endpoint)
            HF_ENDPOINT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "AMCT 快速验证"
echo "  模型: ${MODEL_NAME_OR_PATH}"
echo "  设备: ${DEVICE}"
echo "  用例: ${CASES}"
echo "=========================================="

# ── 模型下载 ──────────────────────────────────────────────────────────────────
# 本地路径直接使用，跳过下载
if [[ "${MODEL_NAME_OR_PATH}" == /* ]] || [[ "${MODEL_NAME_OR_PATH}" == ./* ]] || [[ "${MODEL_NAME_OR_PATH}" == ~* ]]; then
    LOCAL_MODEL_PATH="${MODEL_NAME_OR_PATH}"
    echo "[*] 使用本地模型: ${LOCAL_MODEL_PATH}"
else
    echo "[*] 下载模型 ${MODEL_NAME_OR_PATH} 到 ${MODEL_DIR} ..."
    mkdir -p "${MODEL_DIR}"

    if [[ -n "${HF_ENDPOINT}" ]]; then
        export HF_ENDPOINT="${HF_ENDPOINT}"
        echo "    HF_ENDPOINT=${HF_ENDPOINT}"
    fi

    LOCAL_MODEL_PATH=$(MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" MODEL_DIR="${MODEL_DIR}" MODEL_SOURCE="${MODEL_SOURCE}" TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 - <<'PYEOF'
import os, sys, time

model_id = os.environ["MODEL_NAME_OR_PATH"]
local_dir = os.environ["MODEL_DIR"]
source = os.environ.get("MODEL_SOURCE", "huggingface")

t0 = time.time()
if source == "modelscope":
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("[ERROR] modelscope 未安装，请执行: pip install modelscope", file=sys.stderr)
        sys.exit(1)
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        path = snapshot_download(model_id, local_dir=local_dir)
else:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub 未安装，请执行: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
print(f"    下载完成，耗时 {time.time() - t0:.1f}s", file=sys.stderr)
print(path)
PYEOF
)
    echo "[*] 模型已就绪: ${LOCAL_MODEL_PATH}"
fi

# ── 运行验证 ──────────────────────────────────────────────────────────────────
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 "${SCRIPT_DIR}/run_smoke.py" \
    --model "${LOCAL_MODEL_PATH}" \
    --device "${DEVICE}" \
    --cases "${CASES}"

exit $?
