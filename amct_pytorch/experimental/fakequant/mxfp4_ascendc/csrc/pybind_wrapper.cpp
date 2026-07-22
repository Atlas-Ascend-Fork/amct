/**
 * Pybind11 wrapper: launches the MXFP4 quant-dequant Ascend-C kernel
 * from a PyTorch-NPU tensor and returns the result as a tensor.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <map>
#include <tuple>
#include <mutex>
#include <cstring>

#include "acl/acl.h"
#include "aclrtlaunch_mxfp4_quant_dequant.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace py = pybind11;

/* ---- tiling cache (avoids repeated H2D copies) ---- */
struct TilingCache {
    // Key combines (totalLen, cores, scaleBits) so different runtime scale
    // multipliers do not collide on a single cached tiling buffer.
    std::map<std::tuple<int32_t, int32_t, int32_t>, void *> cache;
    std::mutex mtx;

    void *get_or_create(int32_t totalLen, int32_t cores, float invScaleMul) {
        int32_t scaleBits;
        std::memcpy(&scaleBits, &invScaleMul, sizeof(float));
        auto k = std::make_tuple(totalLen, cores, scaleBits);

        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(k);
        if (it != cache.end())
            return it->second;

        int32_t tiling[4] = {totalLen, cores, scaleBits, 0};
        void *dev = nullptr;
        aclrtMalloc(&dev, sizeof(tiling), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dev, sizeof(tiling), tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
        cache[k] = dev;
        return dev;
    }
};
static TilingCache g_cache;

/* ---- exposed function ---- */
at::Tensor mxfp4_quant_dequant_npu(const at::Tensor &x, int block_dim, double inv_scale_factor_scale) {
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "input must be float32 (got ", x.scalar_type(), ")");
    int32_t totalLen = static_cast<int32_t>(x.numel());
    TORCH_CHECK(totalLen > 0 && totalLen % 32 == 0, "numel() must be a positive multiple of 32, got ", totalLen);

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
    void *tiling = g_cache.get_or_create(totalLen, block_dim, static_cast<float>(inv_scale_factor_scale));

    // The auto-generated host stub still returns 0 ("success") when the
    // kernel handle was never registered (e.g. SoC mismatch), so this
    // check is defense-in-depth only. The Python wrapper performs an
    // additional output-value self-test on first load that reliably
    // detects silent registration failures.
    uint32_t ret = ACLRT_LAUNCH_KERNEL(mxfp4_quant_dequant)(block_dim, stream, const_cast<void *>(x.data_ptr()),
        y.data_ptr(), nullptr, /* workspace */
        tiling);

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
