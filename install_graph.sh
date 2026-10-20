#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
set -e
# 获取当前工作目录
WORKING_DIR=$(pwd)
echo "工作目录: $WORKING_DIR"

# Define default values
ARCH="$(uname -m)"
BASE_NAME="cann-amct-graph"
SOURCE_URL="https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260902_newest/${BASE_NAME}_9.2.0_linux-${ARCH}.tar.gz"
BUNDLE_DIR="${WORKING_DIR}/amctgraph"
OUTPUT_FILE="${BASE_NAME}_*_linux-${ARCH}.tar.gz"
AMCT_GRAPH_FILE="${BASE_NAME}_linux-${ARCH}.tar.gz"
AMCT_GRAPH_PATH="${WORKING_DIR}/build"
# Pinned sha256 checksums per architecture to verify downloaded package integrity.
# NOTE: when SOURCE_URL or the version changes, recompute with sha256sum and update below.
case "${ARCH}" in
  aarch64) AMCT_GRAPH_SHA256="da20354657c9d6e5369d74f2e39db694f505ceed36124c664a8585b8ec3b179e" ;;
  x86_64)  AMCT_GRAPH_SHA256="5bf9aae8b5964dbc5bff69524e659f08e48b5486132343240ac3e595b5613e97" ;;
  *)       echo "Error: unsupported ARCH=${ARCH}"; exit 1 ;;
esac

# Display current directory
echo "Current directory: $(pwd)"
echo "Building $BASE_NAME package for $ARCH"
echo ""

# Create necessary directories (clean stale bundle first to avoid residue)
echo "Creating directories..."
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

if find "${AMCT_GRAPH_PATH}" -name "${OUTPUT_FILE}" | grep -q '.'; then
    echo "${OUTPUT_FILE} exist, copy ${OUTPUT_FILE} to ${AMCT_GRAPH_FILE}."
    cp ${AMCT_GRAPH_PATH}/${OUTPUT_FILE} ${AMCT_GRAPH_FILE}
    echo "Using local package, skip sha256 verification for locally built package."
# Download package
else
    echo "Downloading Ascend CANN Toolkit..."
    if ! wget -O "${AMCT_GRAPH_FILE}" "$SOURCE_URL"; then
        echo "Error: Failed to download Ascend CANN Toolkit"
        rm -f "${AMCT_GRAPH_FILE}"
        [ -n "${BUNDLE_DIR}" ] && rm -rf "$BUNDLE_DIR"
        exit 1
    fi
    # Only verify the downloaded package: locally built packages under
    # ${AMCT_GRAPH_PATH} never match the pinned official checksum.
    echo "Verifying sha256 of ${AMCT_GRAPH_FILE}..."
    if ! echo "${AMCT_GRAPH_SHA256}  ${AMCT_GRAPH_FILE}" | sha256sum -c -; then
        echo "Error: sha256 verification failed for ${AMCT_GRAPH_FILE}"
        rm -f "${AMCT_GRAPH_FILE}"
        [ -n "${BUNDLE_DIR}" ] && rm -rf "$BUNDLE_DIR"
        exit 1
    fi
fi

echo "tar -zxvf ${AMCT_GRAPH_FILE} -C $BUNDLE_DIR"
tar -zxvf "${AMCT_GRAPH_FILE}" -C "$BUNDLE_DIR"

if [ $? -eq 0 ]; then
    echo "Package verification successful"
    echo "Extracted to: ${AMCT_GRAPH_FILE}"

    # 显示解压后的文件结构
    echo "Extracted directory structure:"
    find "$BUNDLE_DIR" -type f | head -20
else
    rm -rf "$BUNDLE_DIR"
    echo "Warning: Package verification failed"
fi

# Remove the downloaded installer
echo "Removing downloaded installer..."
rm -f "${AMCT_GRAPH_FILE}"

echo ""
echo "=============================================="
echo "Package created successfully!"
echo "Location: $BUNDLE_DIR/"
echo "Package type: Tool Compatibility Package"
echo "Architecture: $ARCH"
echo "=============================================="
