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
# One-click offline third-party packages download script.
# Usage: bash download_3rd_party.sh [--3rd_party <directory>]
# Notes:
#   - Default target is <repo root>/third_party: archives are placed directly
#     in that directory and patches in its patch/ subdirectory, matching the
#     CANN_3RD_LIB_PATH layout expected by the build;
#   - Existing files are verified by SHA256 first: matching ones are reused
#     (no re-download), mismatching ones are deleted and downloaded again,
#     and every download is verified again after completion;
#   - After downloading, build offline with the same directory:
#       bash build.sh --torch --cann_3rd_lib_path=third_party

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/third_party"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --3rd_party)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "Error: --3rd_party requires a directory." >&2
                exit 1
            fi
            WORK_DIR="$2"
            shift 2
            ;;
        --3rd_party=*)
            WORK_DIR="${1#*=}"
            if [[ -z "${WORK_DIR}" ]]; then
                echo "Error: --3rd_party requires a directory." >&2
                exit 1
            fi
            shift
            ;;
        -h|--help)
            echo "Usage: bash download_3rd_party.sh [--3rd_party <directory>]"
            echo "Default: <repo root>/third_party; all patches are saved in <directory>/patch."
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

for TOOL in wget sha256sum; do
    if ! command -v "${TOOL}" >/dev/null 2>&1; then
        echo "Error: ${TOOL} is required." >&2
        exit 1
    fi
done

# Normalize WORK_DIR to an absolute path: build.sh resolves --cann_3rd_lib_path
# from the repo root (it does "cd ${BASEPATH}" first), so a relative path here
# would otherwise land next to the caller's cwd and silently fall back to a
# git clone during the offline build. Also makes the final hint always a
# copy-paste-ready absolute path.
mkdir -p -- "${WORK_DIR}"
WORK_DIR="$(cd -- "${WORK_DIR}" && pwd)"

OBS_URL="https://cann-3rd.obs.cn-north-4.myhuaweicloud.com"
# Download list kept in sync with cann/cmake master-059 third_party definitions:
#   https://gitcode.com/cann/cmake/tree/master-059/third_party (commit 3c4a9da4)
# amct builds only depend on protobuf and its transitive dependency abseil-cpp,
# plus the cann-cmake build framework.
# Format: "URL SHA256 Path relative to third_party"
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

check_sha256() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --check --status
}

for ITEM in "${DOWNLOAD_LIST[@]}"; do
    read -r URL SHA256 FILE_NAME <<< "${ITEM}"
    FILE_PATH="${WORK_DIR}/${FILE_NAME}"

    if [[ -f "${FILE_PATH}" ]]; then
        if check_sha256 "${SHA256}" "${FILE_PATH}"; then
            echo "Verified cache, skipping: ${FILE_NAME}"
            continue
        fi
        echo "Cached SHA256 mismatch, downloading again: ${FILE_NAME}"
        rm -f -- "${FILE_PATH}"
    fi

    mkdir -p -- "$(dirname "${FILE_PATH}")"
    TEMP_FILE=$(mktemp -- "${FILE_PATH}.part.XXXXXX")
    echo "Download from ${URL} to ${FILE_PATH}"
    if ! wget -q --show-progress --timeout=60 --tries=3 -O "${TEMP_FILE}" "${URL}"; then
        echo "Error: download failed: ${FILE_NAME}" >&2
        exit 1
    fi
    if ! check_sha256 "${SHA256}" "${TEMP_FILE}"; then
        echo "Error: SHA256 mismatch: ${FILE_NAME}" >&2
        exit 1
    fi
    mv -f -- "${TEMP_FILE}" "${FILE_PATH}"
    TEMP_FILE=""
    echo "Downloaded and verified: ${FILE_NAME}"
done

echo "All third-party packages verified. Files are saved in '${WORK_DIR}'."
echo "Offline build example: bash build.sh --torch --cann_3rd_lib_path=${WORK_DIR}"
