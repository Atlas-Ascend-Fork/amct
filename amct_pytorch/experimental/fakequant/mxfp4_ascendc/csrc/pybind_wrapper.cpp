/**
 * Pybind11 wrapper: launches the MXFP4 quant-dequant Ascend-C kernel
 * from a PyTorch-NPU tensor and returns the result as a tensor.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cstdint>
#include <cstring>

#include "aclrtlaunch_mxfp4_quant_dequant.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace py = pybind11;

/* ---- exposed function ---- */
at::Tensor mxfp4_quant_dequant_npu(const at::Tensor &x, int block_dim, double inv_scale_factor_scale) {
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "input must be float32 (got ", x.scalar_type(), ")");
    // numel() is int64_t. The tiling layout and kernel index arithmetic are
    // int32/uint32, so a tensor with > INT32_MAX elements would be silently
    // truncated into a wrong totalLen. Reject it explicitly instead.
    int64_t numel = x.numel();
    TORCH_CHECK(numel > 0 && numel % 32 == 0 && numel <= INT32_MAX,
        "numel() must be a positive multiple of 32 and <= INT32_MAX "
        "(this kernel uses int32 tiling), got ", numel);
    int32_t totalLen = static_cast<int32_t>(numel);

    auto y = at::empty_like(x);
    // IMPORTANT: pass need_empty=true so torch_npu drains its internal task
    // queue before returning the underlying aclrtStream. Without this, prior
    // PyTorch ops (e.g. fill_, contiguous, prior matmuls feeding `x`) may
    // still sit in the task queue and have NOT been submitted to the NPU
    // stream yet. The kernel we launch directly with ACLRT_LAUNCH_KERNEL
    // would then race ahead of those producers, reading uninitialised
    // memory for `x` and producing all-zero outputs (silently, since the
    // kernel return code is still 0). This was reproducible in vLLM's
    // multi-process worker setup and breaks every MoE/Linear activation.
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    // Tiling is only 4x int32. Allocate via the PyTorch NPU caching allocator
    // instead of a process-global aclrtMalloc cache: each call gets a fresh
    // buffer, and free is deferred until the launch stream completes—no
    // unbounded device-memory growth across (totalLen, cores, scale) keys.
    float invScaleMul = static_cast<float>(inv_scale_factor_scale);
    int32_t scaleBits = 0;
    std::memcpy(&scaleBits, &invScaleMul, sizeof(float));
    int32_t host_tiling[4] = {totalLen, static_cast<int32_t>(block_dim), scaleBits, 0};
    auto tiling = at::empty({4}, at::TensorOptions().dtype(at::kInt).device(x.device()));
    auto tiling_cpu = at::from_blob(host_tiling, {4}, at::TensorOptions().dtype(at::kInt)).clone();
    tiling.copy_(tiling_cpu, /*non_blocking=*/true);

    // The auto-generated host stub still returns 0 ("success") when the
    // kernel handle was never registered (e.g. SoC mismatch), so this
    // check is defense-in-depth only. The Python wrapper performs an
    // additional output-value self-test on first load that reliably
    // detects silent registration failures.
    uint32_t ret = ACLRT_LAUNCH_KERNEL(mxfp4_quant_dequant)(block_dim, stream, const_cast<void *>(x.data_ptr()),
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
        "MXFP4 quant-dequant on Ascend NPU (float32 in, float32 out)", py::arg("x"), py::arg("block_dim") = 20,
        py::arg("inv_scale_factor_scale") = 1.0);
}
