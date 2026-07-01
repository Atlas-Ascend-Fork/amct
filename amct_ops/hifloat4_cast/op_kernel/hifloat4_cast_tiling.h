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

#pragma once

#include <cstdint>

// HiFloat4 is a *block-scaling* format: every 64 elements share a 3-level scale
// (L1 E6M2 + L2/L3 1-bit micro-exponents). The op is a one-shot FP -> HiF4 -> FP
// simulation (fake-quant): FP16/BF16 in, same dtype out, no real 4-bit packing.
//
// Tiling invariant (see P1.5 of the design doc): a 64-element block must never be
// split across a tile or a core, otherwise the per-block max reduction is wrong.
// Therefore numPerCore, tailNumLastCore and tileLength are all multiples of 64.
// tileLength is derived on the host from the platform UB size
// (see op_extension/hifloat4_cast_torch.cpp:ComputeMaxTileLength).

constexpr uint32_t HIF4_BLOCK_SIZE = 64;   // elements per shared-scale block
constexpr uint32_t HIF4_LV3_GROUP = 4;     // elements per L3 (scale_lv3) sub-group -> 16 per block
constexpr uint32_t HIF4_LV2_GROUP = 8;     // elements per L2 (scale_lv2) sub-group -> 8  per block
constexpr uint32_t HIF4_LV3_PER_BLOCK = HIF4_BLOCK_SIZE / HIF4_LV3_GROUP;  // 16
constexpr uint32_t HIF4_LV2_PER_BLOCK = HIF4_BLOCK_SIZE / HIF4_LV2_GROUP;  // 8

enum HiFloat4DtypeFlag : uint32_t {
    HIF4_FP16 = 0,
    HIF4_BF16 = 1,
};

struct HiFloat4CastTilingData {
    uint32_t blockNum;         // number of AI cores used
    uint64_t totalLength;      // total element count (multiple of 64)
    uint64_t numPerCore;       // elements per core (multiple of 64)
    uint64_t tailNumLastCore;  // elements handled by the last core (multiple of 64)
    uint32_t dtypeFlag;        // HiFloat4DtypeFlag: 0 = fp16, 1 = bf16
    uint32_t tileLength;       // elements per tile (multiple of 64, runtime from UB size)
};
