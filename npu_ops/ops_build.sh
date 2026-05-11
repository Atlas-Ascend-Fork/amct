#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
# ----------------------------------------------------------------------------

# 构建所有 NPU 算子并打包为统一 wheel
#
# 用法：
#   bash ops_build.sh               # 构建全部 op，输出到 dist/
#   bash ops_build.sh hifloat8_cast # 仅构建指定 op
#
# 产物：dist/npu_ops-1.0.0-cp*-cp*-linux_aarch64.whl
#
# 新增算子：
#   1. 在 <op>/python/<pkg>/ 下放 __init__.py，<op>/CMakeLists.txt 构建 .so
#   2. 重新运行此脚本即可自动打包

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 加载 CANN 环境 ─────────────────────────────────────────────────────────────
echo "[1/4] 加载 CANN 环境..."
# shellcheck disable=SC1091
source "$ASCEND_HOME_PATH/set_env.sh"

# ── 编译各算子 ────────────────────────────────────────────────────────────────
echo "[2/4] 编译算子..."

build_op() {
    local op_name="$1"
    local op_dir="$SCRIPT_DIR/$op_name"

    echo "  编译 $op_name ..."
    rm -rf "$op_dir/build" && mkdir "$op_dir/build"
    cmake -S "$op_dir" -B "$op_dir/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build "$op_dir/build" -j"$(nproc)"
}

if [ -n "$1" ]; then
    build_op "$1"
else
    for op in */; do
        op="${op%/}"
        [ -f "$op/CMakeLists.txt" ] && build_op "$op"
    done
fi

# ── 汇集各包到 staging/ ───────────────────────────────────────────────────────
echo "[3/4] 汇集 Python 包到 staging/..."
rm -rf staging && mkdir -p staging/npu_ops
cp "$SCRIPT_DIR/npu_ops_init.py" staging/npu_ops/__init__.py

collect_op() {
    local op_name="$1"
    local op_dir="$SCRIPT_DIR/$op_name"
    local python_src="$op_dir/python"

    [ -d "$python_src" ] || return

    for pkg_dir in "$python_src"/*/; do
        [ -d "$pkg_dir" ] || continue
        local pkg_name
        pkg_name="$(basename "$pkg_dir")"
        [[ "$pkg_name" == _* || "$pkg_name" == .* ]] && continue

        local dst="staging/npu_ops/$pkg_name"
        mkdir -p "$dst"

        find "$pkg_dir" -maxdepth 1 -name "*.py" -exec cp {} "$dst/" \;
        find "$op_dir/build" -maxdepth 1 -name "*.so" -exec cp {} "$dst/" \;

        echo "    $op_name → staging/npu_ops/$pkg_name/ ($(ls "$dst"/*.so 2>/dev/null | wc -l) .so)"
    done
}

if [ -n "$1" ]; then
    # 仅重新收集指定 op，保留其他已有包
    collect_op "$1"
    # 补回其余已编译的包
    for op in */; do
        op="${op%/}"
        [ "$op" = "$1" ] && continue
        [ -d "$op/python" ] && collect_op "$op"
    done
else
    for op in */; do
        op="${op%/}"
        [ -d "$op/python" ] && collect_op "$op"
    done
fi

# ── 构建 wheel ────────────────────────────────────────────────────────────────
echo "[4/4] 构建 wheel..."
rm -rf dist npu_ops.egg-info build
mkdir -p dist
pip wheel . -w dist/ --no-deps --no-build-isolation -q

echo ""
echo "构建完成："
ls -lh dist/npu_ops-*.whl
echo ""
echo "内容："
python3 -m zipfile -l dist/npu_ops-*.whl | grep -v "dist-info"
echo ""
echo "安装方式：  pip install dist/npu_ops-*.whl"
