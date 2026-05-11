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

// TILE_LENGTH is no longer a compile-time constant; it is computed at runtime on the host
// based on the platform UB size and passed in HiFloat8CastTilingData.tileLength.
// See hifloat8_cast_op.cpp:ComputeTileLength() for the derivation.
constexpr uint32_t DOUBLE_BUFFER = 2;

// Data type conversion mode
enum HiFloat8CastMode : uint32_t {
    FP32_TO_HIF8 = 0,
    FP16_TO_HIF8 = 1,
    BF16_TO_HIF8 = 2,
    HIF8_TO_FP32 = 3,
    HIF8_TO_FP16 = 4,
    HIF8_TO_BF16 = 5,
};

// Tiling data structure
struct HiFloat8CastTilingData {
    uint32_t blockNum;        // Number of blocks
    uint64_t totalLength;     // Total element count
    uint64_t numPerCore;      // Elements per core
    uint64_t tailNumLastCore; // Tail elements for last core
    uint32_t castMode;        // Conversion mode
    uint32_t tileLength;      // Elements per tile (runtime, based on platform UB size)
};