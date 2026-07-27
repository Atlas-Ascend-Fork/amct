/**
 * Pybind11 wrapper: launches the MXFP4 quant-dequant Ascend-C kernel
 * from a PyTorch-NPU tensor and returns the result as a tensor.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "aclrtlaunch_mxfp4_quant_dequant.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace py = pybind11;

namespace {

// Query AIV core count at runtime (Vector kernel). Fallback 20 matches the
// historical 910B3 default used before auto-detection.
uint32_t GetAivCoreNum() {
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return 20u;
    }
    uint32_t n = platform->GetCoreNumAiv();
    return n > 0u ? n : 20u;
}

}  // namespace

/* ---- exposed function ---- */
at::Tensor mxfp4_quant_dequant_npu(const at::Tensor &x, double inv_scale_factor_scale) {
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "input must be float32 (got ", x.scalar_type(), ")");
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
        "input must be on NPU device, got ", x.device());

    // inv_scale_factor_scale multiplies the kernel's built-in INV_SCALE_FACTOR
    // (1/6.0). Non-positive values make raw_scale=0 (→ NaN/inf) or invert the
    // scale semantics; reject them the same way mxfp4.py rejects scale_factor<=0.
    TORCH_CHECK(inv_scale_factor_scale > 0.0,
        "inv_scale_factor_scale must be positive, got ", inv_scale_factor_scale);
    // numel() is int64_t. The tiling layout and kernel index arithmetic are
    // int32/uint32, so a tensor with > INT32_MAX elements would be silently
    // truncated into a wrong totalLen. Reject it explicitly instead.
    int64_t numel = x.numel();
    TORCH_CHECK(numel > 0 && numel % 32 == 0 && numel <= INT32_MAX,
        "numel() must be a positive multiple of 32 and <= INT32_MAX "
        "(this kernel uses int32 tiling), got ",
        numel);
    int32_t totalLen = static_cast<int32_t>(numel);

    // Launch / tiling core count must match. Cap by MXFP4 block count so tiny
    // tensors do not over-subscribe idle AIVs.
    const uint32_t nBlocks = static_cast<uint32_t>(totalLen / 32);
    const uint32_t numCores = std::max(1u, std::min(GetAivCoreNum(), nBlocks));

    auto y = at::empty_like(x);

    // Tiling is only 4x int32. Allocate via the PyTorch NPU caching allocator
    // instead of a process-global aclrtMalloc cache: each call gets a fresh
    // buffer, and free is deferred until the launch stream completes—no
    // unbounded device-memory growth across (totalLen, cores, scale) keys.
    float invScaleMul = static_cast<float>(inv_scale_factor_scale);
    int32_t scaleBits = 0;
    std::memcpy(&scaleBits, &invScaleMul, sizeof(float));
    int32_t host_tiling[4] = {totalLen, static_cast<int32_t>(numCores), scaleBits, 0};
    auto tiling = at::empty({4}, at::TensorOptions().dtype(at::kInt).device(x.device()));
    auto tiling_cpu = at::from_blob(host_tiling, {4}, at::TensorOptions().dtype(at::kInt)).clone();
    tiling.copy_(tiling_cpu, /*non_blocking=*/true);

    // IMPORTANT: pass need_empty=true so torch_npu drains its internal task
    // queue before returning the underlying aclrtStream. This MUST be done
    // AFTER all the aten prep ops above (empty_like, tiling alloc + copy_),
    // so those ops are actually submitted to the NPU stream before we bypass
    // the queue with ACLRT_LAUNCH_KERNEL below. Without this, prior PyTorch
    // ops (the tiling H2D copy, fill_, contiguous, prior matmuls feeding `x`)
    // may still sit in the task queue and have NOT been submitted to the NPU
    // stream yet. The kernel we launch directly with ACLRT_LAUNCH_KERNEL would
    // then race ahead of those producers, reading uninitialised memory for the
    // tiling buffer (garbage totalLen/numCores/scale) or `x`, producing
    // all-zero or numerically wrong outputs (silently, since the kernel return
    // code is still 0). This was reproducible in vLLM's multi-process worker
    // setup and breaks every MoE/Linear activation.
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    // The auto-generated host stub still returns 0 ("success") when the
    // kernel handle was never registered (e.g. SoC mismatch), so this
    // check is defense-in-depth only. The Python wrapper performs an
    // additional output-value self-test on first load that reliably
    // detects silent registration failures.
    uint32_t ret = ACLRT_LAUNCH_KERNEL(mxfp4_quant_dequant)(numCores, stream, const_cast<void *>(x.data_ptr()),
        y.data_ptr(), nullptr, /* workspace */
        tiling.data_ptr());

    TORCH_CHECK(ret == 0, "mxfp4_quant_dequant kernel launch failed (ret=", ret,
        "). This usually means the compiled kernel cannot run on the "
        "current NPU. Verify the SoC reported by "
        "`npu-smi info -t board -i 0 -c 0 | grep 'NPU Name'` and rebuild "
        "the extension with the matching SOC_VERSION (e.g. "
        "SOC_VERSION=Ascend910_9392 bash build.sh).");

    return y;
}

PYBIND11_MODULE(mxfp4_ascendc_ext, m) {
    m.def("mxfp4_quant_dequant_npu", &mxfp4_quant_dequant_npu,
        "MXFP4 quant-dequant on Ascend NPU (float32 in, float32 out). "
        "AIV core count is queried via PlatformAscendC::GetCoreNumAiv().",
        py::arg("x"), py::arg("inv_scale_factor_scale") = 1.0);
}
