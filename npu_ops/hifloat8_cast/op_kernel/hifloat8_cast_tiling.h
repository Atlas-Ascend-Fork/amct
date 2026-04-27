// ----------------------------------------------------------------------------
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>

// UB 容量 ~1MB，双缓冲需要合理分配
// FP32→HiFloat8: input(float) 4 bytes + output(uint8) 1 byte
// TILE_LENGTH = 8192: input buffer = 32KB, output buffer = 8KB
// Double buffer total: (32KB + 8KB) * 2 = 80KB，远小于 1MB UB 容量
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;

// 数据类型标识
enum HiFloat8CastMode : uint32_t {
    FP32_TO_HIF8 = 0,
    FP16_TO_HIF8 = 1,
    BF16_TO_HIF8 = 2,
    HIF8_TO_FP32 = 3,
    HIF8_TO_FP16 = 4,
    HIF8_TO_BF16 = 5,
};

// Tiling 数据结构
struct HiFloat8CastTilingData {
    uint32_t blockNum;        // 核数
    uint64_t totalLength;     // 总元素数
    uint64_t numPerCore;      // 每核处理的元素数
    uint64_t tailNumLastCore; // 最后一个核处理的尾数
    uint32_t castMode;        // 转换模式
};