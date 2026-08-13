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
// FP -> HiF4 -> FP fake-quant. FP16/BF16/FP32 in, same shape/dtype out (FP16 runs on the FP32
// kernel via an up-convert/down-convert in the host impl). 64-element shared-scale
// blocks are formed along `qdim` (default: last). The op is self-describing (qdim is a schema
// arg); the layout work lives in the host impl, not in python.

TORCH_LIBRARY_FRAGMENT(amct, m) {
    m.def("hifloat4_fake_quant(Tensor input, int qdim=-1) -> Tensor");
}

// ── PrivateUse1 (NPU) implementation ─────────────────────────────────────────
// dtype/device validation lives here (the dispatcher entry), with the actual values printed.

static at::Tensor Hifloat4FakeQuantImpl(const at::Tensor &input, int64_t qdim) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
        "hifloat4_fake_quant: input must be on NPU device, got ", input.device());
    auto dtype = input.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat,
        "hifloat4_fake_quant: expected float16, bfloat16 or float32, got ", input.dtype());
    return AscendKernel::Hifloat4CastTorch(input, qdim);
}

TORCH_LIBRARY_IMPL(amct, PrivateUse1, m) {
    m.impl("hifloat4_fake_quant", TORCH_FN(Hifloat4FakeQuantImpl));
}

// ── Meta (shape-only) implementation ─────────────────────────────────────────
// Output has the same shape/dtype as the input, so shape inference is the identity.

static at::Tensor Hifloat4FakeQuantMeta(const at::Tensor &input, int64_t qdim) {
    (void)qdim;
    return at::empty(input.sizes(), input.options());
}

TORCH_LIBRARY_IMPL(amct, Meta, m) {
    m.impl("hifloat4_fake_quant", &Hifloat4FakeQuantMeta);
}

} // namespace
