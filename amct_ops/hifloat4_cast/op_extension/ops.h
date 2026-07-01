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
#include <torch/extension.h>

namespace AscendKernel {

// FP -> HiFloat4 -> FP fake-quant. `input` must be contiguous FP16/BF16 with numel a
// multiple of 64; every consecutive 64 elements are one shared-scale block. Returns a
// tensor of the same shape/dtype. Block layout (permute/pad to make the quant dim last
// and 64-aligned) is handled by the python wrapper.
at::Tensor Hifloat4CastTorch(const at::Tensor &input);

}  // namespace AscendKernel
