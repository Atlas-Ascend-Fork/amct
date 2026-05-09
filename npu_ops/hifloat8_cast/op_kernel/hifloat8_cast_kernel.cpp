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

#include "hifloat8_cast_scalar.h"
#include "hifloat8_cast_lut.h"

// scalar kernel: FP32 encode（FP32 取值空间 ~43 亿，无法用 LUT 覆盖，唯一实现）
extern "C" __global__ __vector__ void hifloat8_cast_kernel_scalar(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KernelHiFloat8Cast op(&pipe);
    op.Init(input, output, (__gm__ HiFloat8CastTilingData*)tiling);
    op.Process();
}

// LUT kernel: BF16/FP16 encode GM-LUT + decode UB-LUT 优化
// lut_gm: FP16/BF16 encode 时传预计算 32KB GM 指针；decode 时传 0
extern "C" __global__ __vector__ void hifloat8_cast_kernel_lut(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling, GM_ADDR lut_gm)
{
    AscendC::TPipe pipe;
    KernelHiFloat8CastLut op(&pipe);
    op.Init(input, output, (__gm__ HiFloat8CastTilingData*)tiling, lut_gm);
    op.Process();
}
