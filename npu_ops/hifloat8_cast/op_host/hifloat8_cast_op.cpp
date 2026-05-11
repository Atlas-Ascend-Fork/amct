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

#include "hifloat8_cast_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <cstring>

// GM_ADDR is a device-side Ascend C alias for uint64_t; define it for host compilation
#ifndef GM_ADDR
using GM_ADDR = uint64_t;
#endif

// Forward declarations of kernel entry points (defined in hifloat8_cast_kernel.asc)
extern "C" __global__ __vector__ void hifloat8_cast_kernel_lut(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling, GM_ADDR lut_gm);

// ── Host-side LUT helpers ─────────────────────────────────────────────────────
// Run on CPU to precompute the 32768-entry encode LUT once per mode.

static void HostGetHif8BitsNum(int32_t expNoBias,
                                uint32_t& dotValue, uint32_t& expBitsOut, uint32_t& fracBitsOut)
{
    int32_t absExp = expNoBias < 0 ? -expNoBias : expNoBias;
    if (expNoBias == 0)                  { dotValue=0x01; expBitsOut=0; fracBitsOut=3; }
    else if (absExp == 1)                { dotValue=0x02; expBitsOut=1; fracBitsOut=3; }
    else if (absExp >= 2 && absExp <= 3) { dotValue=0x04; expBitsOut=2; fracBitsOut=3; }
    else if (absExp >= 4 && absExp <= 7) { dotValue=0x08; expBitsOut=3; fracBitsOut=2; }
    else                                 { dotValue=0x0C; expBitsOut=4; fracBitsOut=1; }
}

static uint8_t HostFp32MagnitudeToHif8(uint32_t inData)
{
    uint32_t expFp  = (inData >> 23) & 0xFF;
    uint32_t fracFp = inData & 0x7FFFFF;
    if (expFp == 255) return fracFp > 0 ? 0x80 : 0x6F;
    int32_t expNoBias = (int32_t)expFp - 127;
    if (expNoBias < -22) return 0x00;
    if (expNoBias >= 15) return 0x6E;
    if (expNoBias >= -22 && expNoBias < -15) {
        uint32_t fracHiF8 = (fracFp >> 22) >= 1 ? (uint32_t)(23 + expNoBias + 1)
                                                 : (uint32_t)(23 + expNoBias);
        return (uint8_t)fracHiF8;
    }
    uint32_t dotValue, expHiF8Bits, fracHiF8Bits;
    HostGetHif8BitsNum(expNoBias, dotValue, expHiF8Bits, fracHiF8Bits);
    uint32_t fracShift = 23 - fracHiF8Bits - 1;
    uint32_t fractmp   = fracFp >> fracShift;
    uint32_t fracMax   = (1u << (fracHiF8Bits + 1)) - 1;
    uint32_t fracHiF8  = 0;
    if (fractmp == fracMax) {
        expNoBias += 1;
        HostGetHif8BitsNum(expNoBias, dotValue, expHiF8Bits, fracHiF8Bits);
    } else if ((fractmp & 0x01) == 0x01) {
        fracHiF8 = (fractmp + 1) >> 1;
    } else {
        fracHiF8 = fractmp >> 1;
    }
    if (dotValue == 1) return (uint8_t)((dotValue << fracHiF8Bits) | fracHiF8);
    int32_t absExpFinal = expNoBias < 0 ? -expNoBias : expNoBias;
    uint32_t signExp = expNoBias < 0 ? 1u : 0u;
    uint32_t expHiF8 = (uint32_t)absExpFinal - (1u << (expHiF8Bits - 1));
    return (uint8_t)((dotValue << 3) | (signExp << (expHiF8Bits + fracHiF8Bits - 1))
                     | (expHiF8 << fracHiF8Bits) | fracHiF8);
}

// Build 32768-entry encode LUT on CPU for FP16 (castMode=0) or BF16 (castMode=1).
// Positive half only (sign bit stripped): key = absBits = val & 0x7FFF, value = magnitude.
// Sign is re-applied in ComputeEncode16.
static at::Tensor BuildLut16Cpu(int64_t castMode)
{
    uint32_t expBias   = (castMode == 0) ? 15u : 127u;
    uint32_t expBitsN  = (castMode == 0) ? 5u  : 8u;
    uint32_t fracBitsN = (castMode == 0) ? 10u : 7u;

    at::Tensor lut = at::empty({32768}, at::device(at::kCPU).dtype(at::kByte));
    uint8_t*   data = lut.data_ptr<uint8_t>();
    for (uint32_t i = 0; i < 32768; i++) {
        uint16_t val  = (uint16_t)i;  // sign bit always 0
        uint32_t exp  = (val >> fracBitsN) & ((1u << expBitsN) - 1);
        uint32_t frac = val & ((1u << fracBitsN) - 1);
        uint32_t expFp32  = exp + (127u - expBias);
        uint32_t fp32Bits = (expFp32 << 23) | (frac << (23u - fracBitsN));
        data[i] = HostFp32MagnitudeToHif8(fp32Bits);  // magnitude only, no sign
    }
    return lut;
}

// ── Torch binding ──────────────────────────────────────────────────────────────

static uint32_t GetAivCoreNum()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) return 32u;
    uint32_t n = platform->GetCoreNumAiv();
    return n > 0u ? n : 32u;
}

namespace ascend_kernel {

at::Tensor hifloat8_cast_torch(const at::Tensor& input, int64_t castMode) {
    TORCH_CHECK(castMode >= 0 && castMode <= 3, "castMode must be in [0, 3]");
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1, "input must be on NPU device");

    auto inputContiguous = input.contiguous();
    int64_t totalElements = inputContiguous.numel();
    TORCH_CHECK(totalElements > 0, "input tensor must not be empty");

    if (castMode == 0) TORCH_CHECK(input.dtype() == at::kHalf,     "castMode=0 requires half input");
    if (castMode == 1) TORCH_CHECK(input.dtype() == at::kBFloat16, "castMode=1 requires bfloat16 input");
    if (castMode >= 2) TORCH_CHECK(input.dtype() == at::kByte,     "castMode>=2 requires uint8 input");

    at::Tensor output;
    if (castMode <= 1)      output = at::empty(inputContiguous.sizes(), inputContiguous.options().dtype(at::kByte));
    else if (castMode == 2) output = at::empty(inputContiguous.sizes(), inputContiguous.options().dtype(at::kHalf));
    else                    output = at::empty(inputContiguous.sizes(), inputContiguous.options().dtype(at::kBFloat16));

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    constexpr uint32_t MIN_ELEMENTS_PER_CORE = 8192;
    uint32_t maxBlocks = GetAivCoreNum();

    uint32_t idealBlocks = static_cast<uint32_t>((totalElements + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE);
    uint32_t numBlocks = std::min(idealBlocks, maxBlocks);
    numBlocks = std::max(numBlocks, 1u);

    uint64_t numPerCore = static_cast<uint64_t>(totalElements / numBlocks);
    uint64_t tailNum    = static_cast<uint64_t>(totalElements - numPerCore * (numBlocks - 1));

    HiFloat8CastTilingData tilingHost = {numBlocks, static_cast<uint64_t>(totalElements),
                                         numPerCore, tailNum, static_cast<uint32_t>(castMode)};

    at::Tensor tilingTensor = at::empty({sizeof(HiFloat8CastTilingData)},
                                        at::device(at::kCPU).dtype(at::kByte));
    std::memcpy(tilingTensor.data_ptr(), &tilingHost, sizeof(HiFloat8CastTilingData));
    at::Tensor tilingDevice = tilingTensor.to(input.device());

    // Encode LUT 缓存：首次调用时 CPU 构建后传到 NPU，后续复用
    static at::Tensor s_lut_fp16;  // FP16 encode LUT on NPU
    static at::Tensor s_lut_bf16;  // BF16 encode LUT on NPU

    GM_ADDR lut_gm = 0;
    if (castMode <= 1) {
        at::Tensor& cached = (castMode == 0) ? s_lut_fp16 : s_lut_bf16;
        if (!cached.defined()) {
            cached = BuildLut16Cpu(castMode).to(input.device());
        }
        lut_gm = (GM_ADDR)cached.data_ptr();
    }

    hifloat8_cast_kernel_lut<<<numBlocks, nullptr, aclStream>>>(
        (GM_ADDR)inputContiguous.data_ptr(),
        (GM_ADDR)output.data_ptr(),
        (GM_ADDR)tilingDevice.data_ptr(),
        lut_gm);

    return output;
}

}
