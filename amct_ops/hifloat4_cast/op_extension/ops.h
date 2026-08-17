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

// FP -> HiFloat4 -> FP fake-quant. `input` is FP16/BF16 of any shape; 64-element shared-scale
// blocks are formed along `qdim`. The host moves qdim to the last axis, pads it to a multiple
// of 64, runs the flat block kernel, then slices/permutes back. Returns a tensor of the same
// shape/dtype as `input`.
at::Tensor Hifloat4CastTorch(const at::Tensor &input, int64_t qdim);

} // namespace AscendKernel
