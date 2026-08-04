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

set -e

BASEPATH=$(cd "$(dirname "$0")"; pwd)
CMAKE_ARGS=("-DAMCT_OPEN_DIR=${BASEPATH}")

append_cmake_arg() {
    local name="$1"
    local value="$2"
    if [ -n "${value}" ]; then
        CMAKE_ARGS+=("-D${name}=${value}")
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --cann_3rd_lib_path=*)
            CANN_3RD_LIB_PATH="${1#*=}"
            shift
            ;;
        --cann_3rd_lib_path)
            CANN_3RD_LIB_PATH="$2"
            shift 2
            ;;
        *)
            echo "Warning: unknown option ignored: $1"
            shift
            ;;
    esac
done

append_cmake_arg "CANN_3RD_LIB_PATH" "${CANN_3RD_LIB_PATH}"
append_cmake_arg "AMCT_GRAPH_ARCH" "${ARCH}"
append_cmake_arg "AMCT_GRAPH_URL" "${CANN_AMCT_GRAPH_URL}"
append_cmake_arg "AMCT_GRAPH_BASE_URL" "${CANN_AMCT_GRAPH_BASE_URL}"
append_cmake_arg "AMCT_GRAPH_VERSION" "${CANN_AMCT_GRAPH_VERSION}"
append_cmake_arg "AMCT_GRAPH_VERSIONS" "${CANN_AMCT_GRAPH_VERSIONS}"
append_cmake_arg "AMCT_GRAPH_TODAY" "${CANN_AMCT_GRAPH_TODAY}"
append_cmake_arg "AMCT_GRAPH_LOOKBACK_DAYS" "${CANN_AMCT_GRAPH_LOOKBACK_DAYS}"
append_cmake_arg "AMCT_GRAPH_DOWNLOAD_TIMEOUT" "${CANN_AMCT_GRAPH_WGET_TIMEOUT}"
append_cmake_arg "AMCT_GRAPH_DOWNLOAD_TIMEOUT" "${CANN_AMCT_GRAPH_DOWNLOAD_TIMEOUT}"
append_cmake_arg "ASCEND_HOME_PATH" "${ASCEND_HOME_PATH}"

cmake "${CMAKE_ARGS[@]}" -P "${BASEPATH}/cmake/install_graph.cmake"
