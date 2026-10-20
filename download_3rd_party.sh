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
# 一键离线三方件下载脚本
# 用法: bash download_3rd_party.sh [--3rd_party <目录>]
# 说明:
#   - 默认下载到仓库根目录 third_party, 压缩包直接存放于该目录,
#     patch 存放于其 patch 子目录 (即编译时 CANN_3RD_LIB_PATH 的布局要求);
#   - 已存在的同名文件先做 SHA256 校验: 一致则复用缓存不重复下载,
#     不一致则删除后重新下载, 下载完成后再复核完整性;
#   - 下载完成后, 编译时请将 build.sh 的 --cann_3rd_lib_path 指向相同目录:
#       bash build.sh --torch --cann_3rd_lib_path=third_party

set -euo pipefail

WORK_DIR="./third_party"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --3rd_party)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "错误: --3rd_party 需要指定一个目录。" >&2
                exit 1
            fi
            WORK_DIR="$2"
            shift 2
            ;;
        --3rd_party=*)
            WORK_DIR="${1#*=}"
            if [[ -z "${WORK_DIR}" ]]; then
                echo "错误: --3rd_party 需要指定一个目录。" >&2
                exit 1
            fi
            shift
            ;;
        -h|--help)
            echo "用法: bash download_3rd_party.sh [--3rd_party <目录>]"
            echo "默认: ./third_party; 所有 patch 保存于 <目录>/patch 子目录。"
            exit 0
            ;;
        *)
            echo "错误: 未知参数: $1" >&2
            exit 1
            ;;
    esac
done

OBS_URL="https://cann-3rd.obs.cn-north-4.myhuaweicloud.com"
# 下载列表与 cann/cmake master-059 的 third_party 定义保持一致:
#   https://gitcode.com/cann/cmake/tree/master-059/third_party (commit 3c4a9da4)
# amct 构建仅依赖 protobuf 及其传递依赖 abseil-cpp, 另需 cann-cmake 构建框架。
# 格式: "URL SHA256 相对 third_party 的存放路径"
DOWNLOAD_LIST=(
    "https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-059.tar.gz 9e59939b5c97c9498293969df22ab2dd40e450700c51f84f117d065880f40d25 cmake-master-059.tar.gz"
    "${OBS_URL}/abseil-cpp/abseil-cpp-20230802.1.tar.gz 987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed abseil-cpp-20230802.1.tar.gz"
    "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1-h0/backport-CVE-2025-0838.patch a4e4ef6fc4f4c1ed871a3472c5e650068aaadaad2adfef2c8ccb00918bb36d8c patch/backport-CVE-2025-0838.patch"
    "${OBS_URL}/protobuf/protobuf-25.1.tar.gz 9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42 protobuf-25.1.tar.gz"
)

TEMP_FILE=""
trap 'if [[ -n "${TEMP_FILE}" ]]; then rm -f -- "${TEMP_FILE}"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for TOOL in wget sha256sum; do
    if ! command -v "${TOOL}" >/dev/null 2>&1; then
        echo "错误: 缺少必要工具 ${TOOL}。" >&2
        exit 1
    fi
done

check_sha256() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --check --status
}

for ITEM in "${DOWNLOAD_LIST[@]}"; do
    read -r URL SHA256 FILE_NAME <<< "${ITEM}"
    FILE_PATH="${WORK_DIR}/${FILE_NAME}"

    if [[ -f "${FILE_PATH}" ]]; then
        if check_sha256 "${SHA256}" "${FILE_PATH}"; then
            echo "缓存校验通过, 跳过下载: ${FILE_NAME}"
            continue
        fi
        echo "缓存 SHA256 不一致, 重新下载: ${FILE_NAME}"
        rm -f -- "${FILE_PATH}"
    fi

    mkdir -p -- "$(dirname "${FILE_PATH}")"
    TEMP_FILE=$(mktemp -- "${FILE_PATH}.part.XXXXXX")
    echo "下载 ${URL} -> ${FILE_PATH}"
    if ! wget -q --show-progress --timeout=60 --tries=3 -O "${TEMP_FILE}" "${URL}"; then
        echo "错误: 下载失败: ${FILE_NAME}" >&2
        exit 1
    fi
    if ! check_sha256 "${SHA256}" "${TEMP_FILE}"; then
        echo "错误: SHA256 校验失败: ${FILE_NAME}" >&2
        exit 1
    fi
    mv -f -- "${TEMP_FILE}" "${FILE_PATH}"
    TEMP_FILE=""
    echo "下载并校验完成: ${FILE_NAME}"
done

echo "全部三方件下载完成, 存放于 '${WORK_DIR}'。"
echo "离线编译示例: bash build.sh --torch --cann_3rd_lib_path=${WORK_DIR}"
