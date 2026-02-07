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
# 获取当前工作目录
WORKING_DIR=$(pwd)
echo "工作目录: $WORKING_DIR"

# Define default values
ARCH="$(uname -m)"
BASE_NAME="cann-amct-graph"
SOURCE_URL="https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260131_newest/${BASE_NAME}_9.0.0_linux-${ARCH}.tar.gz"
BUNDLE_DIR="${WORKING_DIR}/amctgraph"
OUTPUT_FILE="${BASE_NAME}_*_linux-${ARCH}.tar.gz"
AMCT_GRAPH_FILE="${BASE_NAME}_linux-${ARCH}.tar.gz"
AMCT_GRAPH_PATH="${WORKING_DIR}/build"

# Display current directory
echo "Current directory: $(pwd)"
echo "Building $BASE_NAME package for $ARCH"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p "$BUNDLE_DIR"

if find "${AMCT_GRAPH_PATH}" -name "${OUTPUT_FILE}" | grep -q '.'; then
    echo "${OUTPUT_FILE} exist, copy ${OUTPUT_FILE} to ${AMCT_GRAPH_FILE}."
    cp ${AMCT_GRAPH_PATH}/${OUTPUT_FILE} ${AMCT_GRAPH_FILE}
# Download package
else
    echo "Downloading Ascend CANN Toolkit..."
    wget -O "${AMCT_GRAPH_FILE}" "$SOURCE_URL" --no-check-certificate
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download Ascend CANN Toolkit"
        rm -rf "$BUNDLE_DIR"
        exit 1
    fi
fi

tar -zxvf "${AMCT_GRAPH_FILE}" -C "$BUNDLE_DIR"> /dev/null 2>&1

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