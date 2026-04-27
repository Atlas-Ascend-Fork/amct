// ----------------------------------------------------------------------------
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------------------------------------------------------

#include <torch/extension.h>
#include <torch/library.h>

namespace ascend_kernel {
at::Tensor hifloat8_cast_torch(const at::Tensor& input, int64_t castMode);
}

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("hifloat8_cast(Tensor input, int cast_mode) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("hifloat8_cast", TORCH_FN(ascend_kernel::hifloat8_cast_torch));
}

at::Tensor hifloat8_cast_meta(const at::Tensor& input, int64_t castMode)
{
    if (castMode <= 2) {
        return at::empty(input.sizes(), input.options().dtype(at::kByte));
    } else {
        return at::empty(input.sizes(), input.options().dtype(at::kFloat));
    }
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("hifloat8_cast", &hifloat8_cast_meta);
}

}