/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"
namespace {
using namespace at_npu::native;

at::Tensor svd_quant_npu(const at::Tensor& A, const at::Tensor& W, const at::Tensor& SC,
                         const at::Tensor& DP, const at::Tensor& UP)
{
    auto aShape = A.sizes();
    int64_t nd = aShape.size();
    TORCH_CHECK(nd >= 2, "svd_quant: expected A with shape (*, m, n), got ndim=", nd);

    // construct the output tensor
    at::SmallVector<int64_t, SIZE> oShape(aShape.begin(), aShape.end());
    oShape[aShape.size() - 1] = UP.sizes()[UP.sizes().size() - 1];
    at::Tensor O = at::empty(oShape, torch::TensorOptions().dtype(A.dtype()).device(A.device()));
    EXEC_NPU_CMD_V1(aclnnSvdQuant, A, W, SC, DP, UP, O);
    return O;
}

at::Tensor svd_quant_meta(const at::Tensor& A, const at::Tensor& W, const at::Tensor& SC,
                         const at::Tensor& DP, const at::Tensor& UP)
{
    at::SmallVector<int64_t, SIZE> oShape(A.sizes().begin(), A.sizes().end());
    oShape[A.sizes().size() - 1] = UP.sizes()[UP.sizes().size() - 1];
    return at::empty(oShape, torch::TensorOptions().dtype(A.dtype()).device(A.device()));
}
} // namespace

TORCH_LIBRARY_IMPL(amct_ops, PrivateUse1, m) {
    m.impl("svd_quant", &::svd_quant_npu);
}

TORCH_LIBRARY_IMPL(amct_ops, Meta, m) {
    m.impl("svd_quant", &::svd_quant_meta);
}
