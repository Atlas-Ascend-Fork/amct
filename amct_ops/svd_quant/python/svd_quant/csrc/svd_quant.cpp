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

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"
namespace {
using namespace at_npu::native;

static constexpr int64_t MIN_SHAPE_SIZE = 2;
at::Tensor svd_quant_npu(const at::Tensor &activation, const at::Tensor &weights, const at::Tensor &scales,
    const at::Tensor &down, const at::Tensor &up) {
    auto aShape = activation.sizes();
    int64_t nd = aShape.size();
    TORCH_CHECK(nd >= MIN_SHAPE_SIZE, "svd_quant: expected activation with shape (*, m, n), got ndim=", nd);

    // construct the output tensor
    at::SmallVector<int64_t, SIZE> oShape(aShape.begin(), aShape.end());
    oShape[aShape.size() - 1] = up.sizes()[up.sizes().size() - 1];
    at::Tensor output = at::empty(oShape, torch::TensorOptions().dtype(activation.dtype()).device(activation.device()));
    EXEC_NPU_CMD_V1(aclnnSvdQuant, activation, weights, scales, down, up, output);
    return output;
}

at::Tensor svd_quant_meta(const at::Tensor &activation, const at::Tensor &weights, const at::Tensor &scales,
    const at::Tensor &down, const at::Tensor &up) {
    at::SmallVector<int64_t, SIZE> oShape(activation.sizes().begin(), activation.sizes().end());
    oShape[activation.sizes().size() - 1] = up.sizes()[up.sizes().size() - 1];
    return at::empty(oShape, torch::TensorOptions().dtype(activation.dtype()).device(activation.device()));
}
} // namespace

TORCH_LIBRARY_IMPL(amct, PrivateUse1, m) {
    m.impl("svd_quant", &::svd_quant_npu);
}

TORCH_LIBRARY_IMPL(amct, Meta, m) {
    m.impl("svd_quant", &::svd_quant_meta);
}
