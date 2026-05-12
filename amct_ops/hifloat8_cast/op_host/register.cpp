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

#include <torch/extension.h>

namespace ascend_kernel {
at::Tensor hifloat8_cast_torch(const at::Tensor& input, int64_t castMode);
}

namespace {

// ── Schema ──────────────────────────────────────────────────────────────────

TORCH_LIBRARY_FRAGMENT(amct, m)
{
    m.def("encode_to_hifloat8(Tensor input) -> Tensor");
    m.def("decode_from_hifloat8(Tensor input, ScalarType? dtype=None) -> Tensor");
}

// ── PrivateUse1 (NPU) implementations ───────────────────────────────────────

static at::Tensor encode_impl(const at::Tensor& input)
{
    auto dtype = input.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
                "encode_to_hifloat8: expected float16 or bfloat16, got ", input.dtype());
    return ascend_kernel::hifloat8_cast_torch(input, dtype == at::kHalf ? 0 : 1);
}

static at::Tensor decode_impl(const at::Tensor& input, c10::optional<at::ScalarType> dtype)
{
    TORCH_CHECK(input.scalar_type() == at::kByte,
                "decode_from_hifloat8: input must be uint8, got ", input.dtype());
    at::ScalarType outDtype = dtype.value_or(at::kBFloat16);
    TORCH_CHECK(outDtype == at::kHalf || outDtype == at::kBFloat16,
                "decode_from_hifloat8: expected float16 or bfloat16, got ", outDtype);
    return ascend_kernel::hifloat8_cast_torch(input, outDtype == at::kHalf ? 2 : 3);
}

TORCH_LIBRARY_IMPL(amct, PrivateUse1, m)
{
    m.impl("encode_to_hifloat8", TORCH_FN(encode_impl));
    m.impl("decode_from_hifloat8", TORCH_FN(decode_impl));
}

// ── Meta (shape-only) implementations ───────────────────────────────────────

static at::Tensor encode_meta(const at::Tensor& input)
{
    return at::empty(input.sizes(), input.options().dtype(at::kByte));
}

static at::Tensor decode_meta(const at::Tensor& input, c10::optional<at::ScalarType> dtype)
{
    return at::empty(input.sizes(), input.options().dtype(dtype.value_or(at::kBFloat16)));
}

TORCH_LIBRARY_IMPL(amct, Meta, m)
{
    m.impl("encode_to_hifloat8", &encode_meta);
    m.impl("decode_from_hifloat8", &decode_meta);
}

}