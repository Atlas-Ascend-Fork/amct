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
// Flat block fake-quant: FP16/BF16 in, same dtype out, numel a multiple of 64.
// The quant-dim permute/pad is done in python (see python/hifloat4_cast/ops.py).

TORCH_LIBRARY_FRAGMENT(amct, m) {
    m.def("hifloat4_cast(Tensor input) -> Tensor");
}

// ── PrivateUse1 (NPU) implementation ─────────────────────────────────────────

static at::Tensor Hifloat4CastImpl(const at::Tensor &input) {
    return AscendKernel::Hifloat4CastTorch(input);
}

TORCH_LIBRARY_IMPL(amct, PrivateUse1, m) {
    m.impl("hifloat4_cast", TORCH_FN(Hifloat4CastImpl));
}

// ── Meta (shape-only) implementation ─────────────────────────────────────────

static at::Tensor Hifloat4CastMeta(const at::Tensor &input) {
    return at::empty(input.sizes(), input.options());
}

TORCH_LIBRARY_IMPL(amct, Meta, m) {
    m.impl("hifloat4_cast", &Hifloat4CastMeta);
}

}  // namespace
