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
#include <algorithm>
#include <cstring>
#include <mutex>

// GM_ADDR is a device-side Ascend C alias for uint64_t; define it for host compilation
#ifndef GM_ADDR
using GM_ADDR = uint64_t;
#endif

// Forward declarations of kernel entry points (defined in hifloat8_cast_kernel.asc)
extern "C" __global__ __vector__ void hifloat8_cast_kernel_lut(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling, GM_ADDR lut_gm);

constexpr uint32_t LUT16_SIZE = 32768;  // encode: 32768-entry half-space LUT (32 KB)
constexpr uint32_t LUT8_SIZE  = 256;    // decode: 256-entry LUT (512 B as uint16)

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
    if (expNoBias < -15) {
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
        uint32_t fp32Bits;
        if (exp == 0) {
            // Subnormal: no implicit leading 1; normalize by finding the leading 1 in frac.
            if (frac == 0) {
                fp32Bits = 0u;
            } else {
                // p: position of leading 1 bit; __builtin_clz is UB for 0, guarded above.
                uint32_t p = 31u - (uint32_t)__builtin_clz(frac);
                // fp32Exp = (1 - expBias) + (p - fracBitsN) + 127
                int32_t fp32Exp = (int32_t)p - (int32_t)fracBitsN - (int32_t)(expBias - 1u) + 127;
                uint32_t fp32Frac = (frac & ((1u << p) - 1u)) << (23u - p);
                fp32Bits = (fp32Exp > 0) ? (((uint32_t)fp32Exp << 23) | fp32Frac) : 0u;
            }
        } else {
            uint32_t expFp32 = exp + (127u - expBias);
            fp32Bits = (expFp32 << 23) | (frac << (23u - fracBitsN));
        }
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

static uint32_t GetUbSizeBytes()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) return 256u * 1024u;
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    return ub > 0u ? static_cast<uint32_t>(ub) : 256u * 1024u;
}

// Compute the largest TILE_LENGTH that fits in the UB for the given mode.
//
// UB layout per core:
//   encode (inputBytes=2, outputBytes=1):
//     LUT16(32KB) + DOUBLE_BUFFER*T*2B(inQ) + DOUBLE_BUFFER*T*1B(outQ) = 32768 + 6T
//   decode (inputBytes=1, outputBytes=2):
//     LUT8(512B)  + DOUBLE_BUFFER*T*1B(inQ) + DOUBLE_BUFFER*T*2B(outQ) =   512 + 6T
//
// Cap at 65536 (DataCopyParams split logic in CopyIn covers up to this size).
// Align to 32768 so that full-tile byteCount is always an exact multiple of 32768,
// enabling the {N, 32768} split without remainder handling.
static uint32_t ComputeTileLength(uint32_t ubBytes, bool isEncode)
{
    uint32_t lutBytes = isEncode ? LUT16_SIZE : (LUT8_SIZE * 2u);
    if (ubBytes <= lutBytes + 192u) return 32u;
    uint32_t maxTile = (ubBytes - lutBytes) / 6u;
    maxTile = std::min(maxTile, 65536u);
    if (maxTile >= 32768u) {
        // Align to 32768: keeps full-tile byteCount as an exact multiple of 32768,
        // enabling the efficient {N, 32768} DataCopy split in CopyIn.
        return (maxTile / 32768u) * 32768u;
    }
    // Align to 32 elements (≥ 32-byte DMA alignment for both 1B and 2B/element)
    return std::max((maxTile / 32u) * 32u, 32u);
}

namespace ascend_kernel {

at::Tensor hifloat8_cast_torch(const at::Tensor& input, int64_t castMode) {
    TORCH_CHECK(castMode >= 0 && castMode <= 3, "castMode must be in [0, 3]");
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1, "input must be on NPU device");
    if (castMode == 0) TORCH_CHECK(input.dtype() == at::kHalf,     "castMode=0 requires half input");
    if (castMode == 1) TORCH_CHECK(input.dtype() == at::kBFloat16, "castMode=1 requires bfloat16 input");
    if (castMode >= 2) TORCH_CHECK(input.dtype() == at::kByte,     "castMode>=2 requires uint8 input");

    auto inputContiguous = input.contiguous();
    int64_t totalElements = inputContiguous.numel();
    TORCH_CHECK(totalElements > 0, "input tensor must not be empty");

    at::Tensor output;
    if (castMode <= 1)      output = at::empty(inputContiguous.sizes(), inputContiguous.options().dtype(at::kByte));
    else if (castMode == 2) output = at::empty(inputContiguous.sizes(), inputContiguous.options().dtype(at::kHalf));
    else                    output = at::empty(inputContiguous.sizes(), inputContiguous.options().dtype(at::kBFloat16));

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    bool     isEncode   = (castMode <= 1);
    uint32_t ubBytes    = GetUbSizeBytes();
    uint32_t tileLength = ComputeTileLength(ubBytes, isEncode);
    uint32_t maxBlocks  = GetAivCoreNum();

    uint32_t idealBlocks = static_cast<uint32_t>((totalElements + tileLength - 1) / tileLength);
    uint32_t numBlocks = std::min(idealBlocks, maxBlocks);
    numBlocks = std::max(numBlocks, 1u);

    uint64_t numPerCore = static_cast<uint64_t>(totalElements / numBlocks);
    uint64_t tailNum    = static_cast<uint64_t>(totalElements - numPerCore * (numBlocks - 1));

    HiFloat8CastTilingData tilingHost = {numBlocks, static_cast<uint64_t>(totalElements),
                                         numPerCore, tailNum, static_cast<uint32_t>(castMode),
                                         tileLength};

    at::Tensor tilingTensor = at::empty({sizeof(HiFloat8CastTilingData)},
                                        at::device(at::kCPU).dtype(at::kByte));
    std::memcpy(tilingTensor.data_ptr(), &tilingHost, sizeof(HiFloat8CastTilingData));
    at::Tensor tilingDevice = tilingTensor.to(input.device());

    // Encode LUT 缓存：首次调用时 CPU 构建后传到 NPU，后续复用
    // std::call_once 保证多线程并发调用时只初始化一次
    static at::Tensor    s_lut_fp16, s_lut_bf16;
    static std::once_flag s_once_fp16, s_once_bf16;

    GM_ADDR lut_gm = 0;
    if (castMode <= 1) {
        if (castMode == 0) {
            std::call_once(s_once_fp16, [&]() {
                s_lut_fp16 = BuildLut16Cpu(0).to(input.device());
            });
            lut_gm = (GM_ADDR)s_lut_fp16.data_ptr();
        } else {
            std::call_once(s_once_bf16, [&]() {
                s_lut_bf16 = BuildLut16Cpu(1).to(input.device());
            });
            lut_gm = (GM_ADDR)s_lut_bf16.data_ptr();
        }
    }

    hifloat8_cast_kernel_lut<<<numBlocks, nullptr, aclStream>>>(
        (GM_ADDR)inputContiguous.data_ptr(),
        (GM_ADDR)output.data_ptr(),
        (GM_ADDR)tilingDevice.data_ptr(),
        lut_gm);

    return output;
}

}
