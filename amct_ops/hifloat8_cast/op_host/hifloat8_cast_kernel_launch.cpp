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

// ── OpCommand 入口层 ────────────────────────────────────────────────────────────
// 在 host 侧（__NPU_HOST__）提供 Hifloat8CastLaunchOpCommand，
// 把 <<<>>> launch 封成 lambda 通过 OpCommand::RunOpApiV2 入队，
// 消除 stream(true) 的同步开销。
//
// device 侧（__NPU_ARCH__）直接 #include kernel 实现，不改动原有逻辑。

#include "../op_kernel/hifloat8_cast_kernel.cpp"

#if defined(__NPU_HOST__)

#include <cstdint>
#include "hifloat8_cast_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void Hifloat8CastLaunchOpCommand(
    uint32_t blockNum, uint8_t *input, uint8_t *output,
    uint8_t *tiling, uint8_t *lut)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() -> int {
        hifloat8_cast_kernel_lut<<<blockNum, nullptr, acl_stream>>>(
            input, output, tiling, lut);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("hifloat8_cast", acl_call);
}

#endif  // __NPU_HOST__
