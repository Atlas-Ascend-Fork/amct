#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Apache License, Version 2.0
#
# 构建所有 NPU 算子并打包为统一 wheel
#
# 用法：
#   bash ops_build.sh [--soc <soc>] [<op>]
#
#   --soc <soc>   目标 SOC（可选，默认 ascend910b）：
#                   ascend910b    Ascend A2（910B1/B2/B3，dav-2201）[默认]
#                   ascend910_93  Ascend A3（910B4，dav-3101）
#                   ascend950     Ascend A5（dav-3510）
#   <op>          仅构建指定算子，省略则构建全部
#
# 示例：
#   bash ops_build.sh                            # 构建全部，默认 A2
#   bash ops_build.sh --soc ascend910_93         # 构建全部，A3
#   bash ops_build.sh hifloat8_cast              # 仅构建 hifloat8_cast，A2
#   bash ops_build.sh --soc ascend950 hifloat8_cast  # 仅构建 hifloat8_cast，A5
#
# 产物：dist/npu_ops-1.0.0-cp*-cp*-linux_aarch64.whl
#
# 新增算子：
#   1. 在 <op>/python/<pkg>/ 下放 __init__.py，<op>/CMakeLists.txt 构建 .so
#   2. 重新运行此脚本即可自动打包

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 解析参数 ──────────────────────────────────────────────────────────────────
SOC="ascend910b"
OP_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --soc)
            SOC="$2"
            shift 2
            ;;
        --soc=*)
            SOC="${1#*=}"
            shift
            ;;
        -*)
            echo "未知选项: $1" >&2
            echo "用法: bash ops_build.sh [--soc ascend910b|ascend910_93|ascend950] [<op>]" >&2
            exit 1
            ;;
        *)
            OP_NAME="$1"
            shift
            ;;
    esac
done

# SOC → NPU_ARCH
# ascend910b 和 ascend910_93 共用 dav-2201（ISA 相同，UB 大小由运行时平台 API 区分）
case "$SOC" in
    ascend910b)   NPU_ARCH="dav-2201" ;;
    ascend910_93) NPU_ARCH="dav-2201" ;;
    ascend950)    NPU_ARCH="dav-3510" ;;
    *)
        echo "不支持的 SOC: '$SOC'，可选值: ascend910b, ascend910_93, ascend950" >&2
        exit 1
        ;;
esac

# ── 加载 CANN 环境 ─────────────────────────────────────────────────────────────
echo "[1/4] 加载 CANN 环境..."
# shellcheck disable=SC1091
source "$ASCEND_HOME_PATH/set_env.sh"

# ── 编译各算子 ────────────────────────────────────────────────────────────────
echo "[2/4] 编译算子 (SOC=${SOC}, NPU_ARCH=${NPU_ARCH})..."

build_op() {
    local op_name="$1"
    local op_dir="$SCRIPT_DIR/$op_name"

    echo "  编译 $op_name ..."
    rm -rf "$op_dir/build" && mkdir "$op_dir/build"
    cmake -S "$op_dir" -B "$op_dir/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DNPU_ARCH="${NPU_ARCH}" > /dev/null
    cmake --build "$op_dir/build" -j"$(nproc)"
}

if [ -n "$OP_NAME" ]; then
    build_op "$OP_NAME"
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

if [ -n "$OP_NAME" ]; then
    # 仅重新收集指定 op，保留其他已有包
    collect_op "$OP_NAME"
    # 补回其余已编译的包
    for op in */; do
        op="${op%/}"
        [ "$op" = "$OP_NAME" ] && continue
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
