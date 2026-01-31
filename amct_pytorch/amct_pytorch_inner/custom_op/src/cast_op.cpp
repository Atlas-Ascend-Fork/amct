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
 *
 * @brief torch C++ backend api of hif8 and fp8 cast function.
 *
 * @file cast_op.cpp
 *
 * @version 1.0
 */
#include "cast_op.h"
#include "util_op.h"
#include "cast_util.h"
#include "util.h"

using namespace util;

#ifdef TORCH_VERSION_MAJOR
    #if (TORCH_VERSION_MAJOR > 2 || \
        (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1) || \
        (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 1 && TORCH_VERSION_PATCH >= 0))
        #define OUTPUT_DTYPE torch::kFloat8_e4m3fn
    #else
        #define OUTPUT_DTYPE torch::kUInt8
    #endif
#else
    #define OUTPUT_DTYPE torch::kUInt8
#endif

template <typename T>
void FloatCastToHiF8Fp8(torch::Tensor& input, torch::Tensor& output, int roundMode, int dstType)
{
    int dataType = FP32;
    auto inputPtr = reinterpret_cast<T*>(input.data_ptr());
    if (input.dtype() == torch::kFloat16) {
        dataType = FP16;
    } else if (input.dtype() == torch::kBFloat16) {
        dataType = BF16;
    }
    DataCastToHiF8Fp8Functor<util::CPUDevice, T>()(inputPtr,
        reinterpret_cast<uint8_t*>(output.data_ptr()), input.numel(), dataType, dstType, roundMode);
}

template <typename T>
void HiF8Fp8CastToFloat(torch::Tensor& input, torch::Tensor& output, int dataType, int srcType)
{
    auto inputPtr = reinterpret_cast<uint8_t*>(input.data_ptr());
    if (dataType == FP16) {
        output = output.to(torch::kFloat16);
    } else if (dataType == BF16) {
        output = output.to(torch::kBFloat16);
    } else {
        output = output.to(torch::kFloat32);
    }
    HiF8Fp8CastToFpNFunctor<util::CPUDevice, T>()(inputPtr,
        reinterpret_cast<T*>(output.data_ptr()), input.numel(), dataType, srcType);
}

template <typename T>
void FloatCastToFP4E2M1(torch::Tensor& input, torch::Tensor& output)
{
    int dataType;
    auto srcType = input.dtype();
    auto inputPtr = reinterpret_cast<T*>(input.data_ptr());
    if (srcType == torch::kFloat16) {
        dataType = FP16;
    } else if (srcType == torch::kBFloat16) {
        dataType = BF16;
    } else if (srcType == torch::kFloat32) {
        dataType = FP32;
    }
    FloatCastToFP4E2M1Functor<util::CPUDevice, T>()(inputPtr,
        output.data_ptr<uint8_t>(), input.numel(), dataType);
}

template <typename T>
void FloatCastToFP4E1M2(torch::Tensor& input, torch::Tensor& output)
{
    int dataType;
    auto srcType = input.dtype();
    auto inputPtr = reinterpret_cast<T*>(input.data_ptr());
    if (srcType == torch::kFloat16) {
        dataType = FP16;
    } else if (srcType == torch::kBFloat16) {
        dataType = BF16;
    } else if (srcType == torch::kFloat32) {
        dataType = FP32;
    }
    FloatCastToFP4E1M2Functor<util::CPUDevice, T>()(inputPtr,
        output.data_ptr<uint8_t>(), input.numel(), dataType);
}

template <typename T>
void Float4E2M1CastToFloat(torch::Tensor& input, torch::Tensor& output, int dstType)
{
    auto inputPtr = reinterpret_cast<uint8_t*>(input.data_ptr());
    FP4E2M1CastToFloatFunctor<util::CPUDevice, T>()(inputPtr,
        reinterpret_cast<T*>(output.data_ptr()), input.numel(), dstType);
}

template <typename T>
void Float4E1M2CastToFloat(torch::Tensor& input, torch::Tensor& output, int dstType)
{
    auto inputPtr = reinterpret_cast<uint8_t*>(input.data_ptr());
    FP4E1M2CastToFloatFunctor<util::CPUDevice, T>()(inputPtr,
        reinterpret_cast<T*>(output.data_ptr()), input.numel(), dstType);
}

torch::Tensor CastToFP4E2M1(torch::Tensor input)
{
    input = input.contiguous();
    torch::Tensor output = torch::zeros_like(input).to(torch::kUInt8);
    if (input.dtype() == torch::kFloat32) {
        FloatCastToFP4E2M1<uint32_t>(input, output);
    } else {
        FloatCastToFP4E2M1<uint16_t>(input, output);
    }
    return output;
}

torch::Tensor CastToFP4E1M2(torch::Tensor input)
{
    input = input.contiguous();
    torch::Tensor output = torch::zeros_like(input).to(torch::kUInt8);
    if (input.dtype() == torch::kFloat32) {
        FloatCastToFP4E1M2<uint32_t>(input, output);
    } else {
        FloatCastToFP4E1M2<uint16_t>(input, output);
    }
    return output;
}

torch::Tensor FP4E2M1CastToFloat(torch::Tensor input, int dstType)
{
    input = input.contiguous();
    torch::Tensor output = torch::zeros_like(input);
    if (dstType == FP16) {
        output = output.to(torch::kFloat16);
        Float4E2M1CastToFloat<uint16_t>(input, output, dstType);
    } else if (dstType == BF16) {
        output = output.to(torch::kBFloat16);
        Float4E2M1CastToFloat<uint16_t>(input, output, dstType);
    } else if (dstType == FP32) {
        output = output.to(torch::kFloat32);
        Float4E2M1CastToFloat<uint32_t>(input, output, dstType);
    }
    return output;
}

torch::Tensor FP4E1M2CastToFloat(torch::Tensor input, int dstType)
{
    input = input.contiguous();
    torch::Tensor output = torch::zeros_like(input);
    if (dstType == FP16) {
        output = output.to(torch::kFloat16);
        Float4E1M2CastToFloat<uint16_t>(input, output, dstType);
    } else if (dstType == BF16) {
        output = output.to(torch::kBFloat16);
        Float4E1M2CastToFloat<uint16_t>(input, output, dstType);
    } else if (dstType == FP32) {
        output = output.to(torch::kFloat32);
        Float4E1M2CastToFloat<uint32_t>(input, output, dstType);
    }
    return output;
}

torch::Tensor CastToHiF8Fp8(torch::Tensor& input, int roundMode, int dstType)
{
    input = input.contiguous();
    torch::Tensor output;
    if (dstType == FP8)
        output = torch::zeros_like(input).to(OUTPUT_DTYPE);
    else
        output = torch::zeros_like(input).to(torch::kUInt8);
    if (input.dtype() == torch::kFloat32) {
        FloatCastToHiF8Fp8<uint32_t>(input, output, roundMode, dstType);
    } else {
        FloatCastToHiF8Fp8<uint16_t>(input, output, roundMode, dstType);
    }
    return output;
}

torch::Tensor CastToFloat(torch::Tensor& input, int dataType, int srcType)
{
    input = input.contiguous();
    torch::Tensor output = torch::zeros_like(input.to(torch::kFloat32));
    if (dataType == FP32) {
        HiF8Fp8CastToFloat<uint32_t>(input, output, dataType, srcType);
    } else {
        HiF8Fp8CastToFloat<uint16_t>(input, output, dataType, srcType);
    }
    return output;
}

torch::Tensor CastToHiFP8(torch::Tensor input, int roundMode)
{
    return CastToHiF8Fp8(input, roundMode, HIF8);
}

torch::Tensor CastToFP8E4M3FN(torch::Tensor input)
{
    int roundMode = 0;
    return CastToHiF8Fp8(input, roundMode, FP8);
}

torch::Tensor HiFP8CastToFloat(torch::Tensor input, int dataType)
{
    return CastToFloat(input, dataType, HIF8);
}

torch::Tensor FP8E4M3FNCastToFloat(torch::Tensor input, int dataType)
{
    return CastToFloat(input, dataType, FP8);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float_to_hifp8", &CastToHiFP8, "Cast FpN to HiFloat8");
    m.def("hifp8_to_float", &HiFP8CastToFloat, "Cast HiFloat8 to FpN");
    m.def("float_to_fp8e4m3fn", &CastToFP8E4M3FN, "Cast FpN to Fp8");
    m.def("fp8e4m3fn_to_float", &FP8E4M3FNCastToFloat, "Cast Fp8 to FpN");
    m.def("float_to_fp4e2m1", &CastToFP4E2M1, "Cast Float to FP4E2M1");
    m.def("float_to_fp4e1m2", &CastToFP4E1M2, "Cast Float to FP4E1M2");
    m.def("fp4e2m1_to_float", &FP4E2M1CastToFloat, "Cast FP4E2M1 to Float");
    m.def("fp4e1m2_to_float", &FP4E1M2CastToFloat, "Cast FP4E1M2 to Float");
}
