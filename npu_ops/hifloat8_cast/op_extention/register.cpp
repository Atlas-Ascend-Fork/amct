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