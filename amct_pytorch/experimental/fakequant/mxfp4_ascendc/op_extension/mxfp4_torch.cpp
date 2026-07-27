/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cstdint>

#include "aclrtlaunch_mxfp4_quant_dequant.h"
#include "mxfp4_tiling.h"
#include "ops.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

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

} // namespace

namespace AscendKernel {

at::Tensor Mxfp4QuantDequantTorch(const at::Tensor &x, double invScaleFactorScale) {
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "input must be float32 (got ", x.scalar_type(), ")");
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "input must be on NPU device, got ", x.device());

    // invScaleFactorScale multiplies the kernel's built-in INV_SCALE_FACTOR
    // (1/6.0). Non-positive values make raw_scale=0 (→ NaN/inf) or invert the
    // scale semantics; reject them the same way the reference rejects scale_factor<=0.
    TORCH_CHECK(invScaleFactorScale > 0.0, "inv_scale_factor_scale must be positive, got ", invScaleFactorScale);
    // numel() is int64_t. The tiling layout and kernel index arithmetic are
    // int32/uint32, so a tensor with > INT32_MAX elements would be silently
    // truncated into a wrong totalLen. Reject it explicitly instead.
    int64_t numel = x.numel();
    TORCH_CHECK(numel > 0 && numel % MXFP4_BLOCK_SIZE == 0 && numel <= INT32_MAX,
        "numel() must be a positive multiple of ", MXFP4_BLOCK_SIZE,
        " and <= INT32_MAX "
        "(this kernel uses int32 tiling), got ",
        numel);
    int32_t totalLen = static_cast<int32_t>(numel);

    // Launch / tiling core count must match. Cap by MXFP4 block count so tiny
    // tensors do not over-subscribe idle AIVs.
    const uint32_t nBlocks = static_cast<uint32_t>(totalLen / MXFP4_BLOCK_SIZE);
    const uint32_t numCores = std::max(1u, std::min(GetAivCoreNum(), nBlocks));

    auto y = at::empty_like(x);

    // Tiling is only Mxfp4TilingData (4x int32). Allocate via the PyTorch NPU
    // caching allocator instead of a process-global aclrtMalloc cache: each
    // call gets a fresh buffer, and free is deferred until the launch stream
    // completes—no unbounded device-memory growth across (totalLen, cores,
    // scale) keys.
    float invScaleMul = static_cast<float>(invScaleFactorScale);
    // Bit-cast float bits into tiling int32 slot (same pattern as kernel side).
    union {
        float f;
        int32_t i;
    } scaleU;
    scaleU.f = invScaleMul;
    int32_t scaleBits = scaleU.i;
    Mxfp4TilingData hostTiling{totalLen, static_cast<int32_t>(numCores), scaleBits, 0};
    auto tiling = at::empty({MXFP4_TILING_INTS}, at::TensorOptions().dtype(at::kInt).device(x.device()));
    auto tilingCpu = at::from_blob(&hostTiling, {MXFP4_TILING_INTS}, at::TensorOptions().dtype(at::kInt)).clone();
    tiling.copy_(tilingCpu, /*non_blocking=*/true);

    // IMPORTANT: pass need_empty=true so torch_npu drains its internal task
    // queue before returning the underlying aclrtStream. This MUST be done
    // AFTER all the aten prep ops above (empty_like, tiling alloc + copy_),
    // so those ops are actually submitted to the NPU stream before we bypass
    // the queue with ACLRT_LAUNCH_KERNEL below. Without this, prior PyTorch
    // ops may still sit in the task queue and have NOT been submitted to the
    // NPU stream yet. The kernel launched with ACLRT_LAUNCH_KERNEL would then
    // race ahead of those producers.
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

} // namespace AscendKernel
