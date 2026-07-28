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

#include "ops.h"

namespace {

// ── Schema ──────────────────────────────────────────────────────────────────

TORCH_LIBRARY_FRAGMENT(amct, m) {
    m.def("quant_dequant_mxfp4(Tensor x, float inv_scale_factor_scale=1.0) -> Tensor");
}

// ── PrivateUse1 (NPU) implementations ───────────────────────────────────────

static at::Tensor QuantDequantMxfp4Impl(const at::Tensor &x, double invScaleFactorScale) {
    return AscendKernel::Mxfp4QuantDequantTorch(x, invScaleFactorScale);
}

TORCH_LIBRARY_IMPL(amct, PrivateUse1, m) {
    m.impl("quant_dequant_mxfp4", TORCH_FN(QuantDequantMxfp4Impl));
}

// ── Meta (shape-only) implementations ───────────────────────────────────────

static at::Tensor QuantDequantMxfp4Meta(const at::Tensor &x, double /*invScaleFactorScale*/) {
    return at::empty(x.sizes(), x.options());
}

TORCH_LIBRARY_IMPL(amct, Meta, m) {
    m.impl("quant_dequant_mxfp4", &QuantDequantMxfp4Meta);
}

} // namespace
