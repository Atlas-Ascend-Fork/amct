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

// Host / device shared constants and tiling layout for MXFP4 QDQ.
constexpr int32_t MXFP4_BLOCK_SIZE = 32;
constexpr float MXFP4_SCALE_FACTOR = 6.0f;
constexpr float MXFP4_INV_SCALE_FACTOR = 1.0f / 6.0f;
constexpr float MXFP4_MIN_SCALE_RAW = 9.313225746e-10f; // 2^-30
constexpr int32_t MXFP4_BLOCKS_PER_TILE = 208;
constexpr int32_t MXFP4_TILE_ELEMS = MXFP4_BLOCK_SIZE * MXFP4_BLOCKS_PER_TILE;

// Tiling is packed as 4x int32 for the Ascend-C kernel GM buffer.
// invScaleMulBits stores the IEEE-754 bit pattern of the runtime scale multiplier.
struct Mxfp4TilingData {
    int32_t totalLen;
    int32_t numCores;
    int32_t invScaleMulBits;
    int32_t reserved;
};

constexpr int32_t MXFP4_TILING_INTS = static_cast<int32_t>(sizeof(Mxfp4TilingData) / sizeof(int32_t));
