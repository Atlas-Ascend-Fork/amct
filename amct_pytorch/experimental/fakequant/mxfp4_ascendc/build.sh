#!/usr/bin/env bash
# ==============================================================
#  MXFP4 Ascend-C Kernel — One-click Build
# ==============================================================
#  Usage:
#    bash build.sh            # build
#    bash build.sh --clean    # clean build directory
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC_DIR="${SCRIPT_DIR}/csrc"
BUILD_DIR="${SCRIPT_DIR}/build"

PYTHON="${PYTHON:-$(which python3 2>/dev/null || which python)}"
CANN_PATH="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    source "${CANN_PATH}/../../set_env.sh" 2>/dev/null || true
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
fi

# Auto-detect SoC version from npu-smi when SOC_VERSION is not provided.
# Avoids silent failure when the host hardware (e.g. A3 Ascend910_9392)
# does not match the previously hardcoded Ascend910B3 (A2) target.
detect_soc() {
    command -v npu-smi >/dev/null 2>&1 || return 1
    local chip_name
    chip_name=$(npu-smi info -t board -i 0 -c 0 2>/dev/null \
        | awk -F': ' '/NPU Name/{gsub(/[ \t\r]/,"",$2); print $2; exit}')
    [[ -z "${chip_name}" ]] && return 1
    local arch
    arch=$(uname -m)
    local search_roots=(
        "${CANN_PATH}/compiler/data/platform_config"
        "${CANN_PATH}/${arch}-linux/data/platform_config"
        "${CANN_PATH}/aarch64-linux/data/platform_config"
        "${CANN_PATH}/x86_64-linux/data/platform_config"
    )
    local candidates=()
    if [[ "${chip_name}" =~ ^9[0-9]{3}$ ]]; then
        candidates+=("Ascend910_${chip_name}")
    fi
    candidates+=("Ascend${chip_name}" "Ascend910${chip_name}")
    local root cand
    for root in "${search_roots[@]}"; do
        [[ -d "${root}" ]] || continue
        for cand in "${candidates[@]}"; do
            if [[ -f "${root}/${cand}.ini" ]]; then
                echo "${cand}"
                return 0
            fi
        done
    done
    # No matching platform_config .ini found. Do not invent an unverified
    # Ascend910_${chip_name} — that makes CMake fail with an opaque missing
    # config error instead of the caller's clear WARNING + Ascend910B3 fallback.
    return 1
}

DETECTED_SOC=""
if [[ -z "${SOC_VERSION:-}" ]]; then
    if DETECTED_SOC=$(detect_soc); then
        SOC_VERSION="${DETECTED_SOC}"
    else
        SOC_VERSION="Ascend910B3"
        echo "[build] WARNING: failed to auto-detect SoC, falling back to ${SOC_VERSION}." >&2
    fi
fi

if [[ "${1:-}" == "--clean" ]]; then
    echo "[build] Cleaning ${BUILD_DIR} ..."
    rm -rf "${BUILD_DIR}"
    exit 0
fi

echo "======================================================"
echo "  MXFP4 Ascend-C Kernel Build"
echo "  CANN:        ${CANN_PATH}"
echo "  SOC:         ${SOC_VERSION}"
if [[ -n "${DETECTED_SOC}" ]]; then
    echo "  detected_by: npu-smi"
else
    echo "  source:      env/default (set SOC_VERSION=... to override)"
fi
echo "  Python:      ${PYTHON}"
echo "======================================================"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# CANN lays out arch-specific headers/libs under <arch>-linux; pass the host
# arch so CMake picks the right directory (aarch64-linux vs x86_64-linux).
ARCH="$(uname -m)"

cmake "${CSRC_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DRUN_MODE=npu \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_PATH}" \
    -DARCH="${ARCH}" \
    -DPython3_EXECUTABLE="${PYTHON}" \
    2>&1

cmake --build . -j"$(nproc)" 2>&1

echo ""
echo "======================================================"
echo "  Build artifacts"
echo "======================================================"
for f in mxfp4_ascendc_ext*.so lib/libascendc_kernels_npu.so; do
    [[ -e "${BUILD_DIR}/${f}" ]] && ls -lh "${BUILD_DIR}/${f}"
done
echo "======================================================"
echo "  Build complete!"
echo "======================================================"
