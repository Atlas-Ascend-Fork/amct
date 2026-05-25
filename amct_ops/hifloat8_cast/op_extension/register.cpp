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
#include "hifloat8_cast_tiling.h"

namespace {

// ── Schema ──────────────────────────────────────────────────────────────────

TORCH_LIBRARY_FRAGMENT(amct, m) {
    m.def("encode_to_hifloat8(Tensor input) -> Tensor");
    m.def("decode_from_hifloat8(Tensor input, ScalarType? dtype=None) -> Tensor");
}

// ── PrivateUse1 (NPU) implementations ───────────────────────────────────────

static at::Tensor EncodeImpl(const at::Tensor &input) {
    auto dtype = input.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "encode_to_hifloat8: expected float16 or bfloat16, got ",
        input.dtype());
    return AscendKernel::Hifloat8CastTorch(input, dtype == at::kHalf ? FP16_TO_HIF8 : BF16_TO_HIF8);
}

static at::Tensor DecodeImpl(const at::Tensor &input, c10::optional<at::ScalarType> dtype) {
    TORCH_CHECK(input.scalar_type() == at::kByte, "decode_from_hifloat8: input must be uint8, got ", input.dtype());
    at::ScalarType outDtype = dtype.value_or(at::kBFloat16);
    TORCH_CHECK(outDtype == at::kHalf || outDtype == at::kBFloat16,
        "decode_from_hifloat8: expected float16 or bfloat16, got ", outDtype);
    return AscendKernel::Hifloat8CastTorch(input, outDtype == at::kHalf ? HIF8_TO_FP16 : HIF8_TO_BF16);
}

TORCH_LIBRARY_IMPL(amct, PrivateUse1, m) {
    m.impl("encode_to_hifloat8", TORCH_FN(EncodeImpl));
    m.impl("decode_from_hifloat8", TORCH_FN(DecodeImpl));
}

// ── Meta (shape-only) implementations ───────────────────────────────────────

static at::Tensor EncodeMeta(const at::Tensor &input) {
    return at::empty(input.sizes(), input.options().dtype(at::kByte));
}

static at::Tensor DecodeMeta(const at::Tensor &input, c10::optional<at::ScalarType> dtype) {
    return at::empty(input.sizes(), input.options().dtype(dtype.value_or(at::kBFloat16)));
}

TORCH_LIBRARY_IMPL(amct, Meta, m) {
    m.impl("encode_to_hifloat8", &EncodeMeta);
    m.impl("decode_from_hifloat8", &DecodeMeta);
}

} // namespace
