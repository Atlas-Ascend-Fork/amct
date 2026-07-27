#!/usr/bin/env bash
# ==============================================================
#  MXFP4 Ascend-C Kernel — One-click Build
# ==============================================================
#  Usage:
#    bash build.sh            # build + copy .so into python/mxfp4/
#    bash build.sh --clean    # clean build directory
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PYTHON_PKG_DIR="${SCRIPT_DIR}/python/mxfp4"

PYTHON="${PYTHON:-$(which python3 2>/dev/null || which python)}"
CANN_PATH="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    source "${CANN_PATH}/../../set_env.sh" 2>/dev/null || true
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
fi
CANN_PATH="${ASCEND_TOOLKIT_HOME:-$CANN_PATH}"

# Auto-detect SoC version from npu-smi when SOC_VERSION is not provided.
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

ARCH="$(uname -m)"

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DRUN_MODE=npu \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_PATH}" \
    -DARCH="${ARCH}" \
    -DPython3_EXECUTABLE="${PYTHON}" \
    2>&1

cmake --build . -j"$(nproc)" 2>&1

# Stage shared libraries next to the Python package (hifloat8_cast style).
mkdir -p "${PYTHON_PKG_DIR}"
OPS_SO="$(find "${BUILD_DIR}" -maxdepth 1 -name 'libmxfp4_ops.so' | head -n1)"
KERNEL_SO="$(find "${BUILD_DIR}" -path '*/libascendc_kernels_npu.so' | head -n1)"
if [[ -z "${OPS_SO}" || -z "${KERNEL_SO}" ]]; then
    echo "[build] ERROR: expected artifacts not found under ${BUILD_DIR}" >&2
    echo "  libmxfp4_ops.so:           ${OPS_SO:-MISSING}" >&2
    echo "  libascendc_kernels_npu.so: ${KERNEL_SO:-MISSING}" >&2
    exit 1
fi
cp -f "${OPS_SO}" "${PYTHON_PKG_DIR}/"
cp -f "${KERNEL_SO}" "${PYTHON_PKG_DIR}/"

echo ""
echo "======================================================"
echo "  Build artifacts"
echo "======================================================"
ls -lh "${OPS_SO}" "${KERNEL_SO}"
echo "  staged -> ${PYTHON_PKG_DIR}/"
ls -lh "${PYTHON_PKG_DIR}/libmxfp4_ops.so" "${PYTHON_PKG_DIR}/libascendc_kernels_npu.so"
echo "======================================================"
echo "  Build complete!"
echo "======================================================"
