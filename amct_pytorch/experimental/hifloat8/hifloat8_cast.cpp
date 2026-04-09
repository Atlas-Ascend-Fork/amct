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
 * @file hifloat8_cast.cpp
 *
 * @version 1.0
 */

#include <cmath>
#include <utility>
#include <functional>
#include <map>
#include <tuple>
#include <cstdint>
#include <torch/extension.h>


constexpr uint32_t FP32_EXP_BITS = 8;
constexpr uint32_t FP32_FRAC_BITS = 23;
constexpr uint32_t FP16_EXP_BITS = 5;
constexpr uint32_t FP16_FRAC_BITS = 10;
constexpr uint32_t BF16_EXP_BITS = 8;
constexpr uint32_t BF16_FRAC_BITS = 7;
constexpr uint32_t HIF8_SIGN_SHIFT = 7;
constexpr uint32_t HIF8_DENORMAL_EXP_BIAS = 23;
constexpr int FP32 = 0;
constexpr int FP16 = 1;
constexpr int BF16 = 2;


const std::map<uint8_t, std::tuple<uint8_t, uint8_t, uint8_t, uint8_t, uint8_t>> HIF8_DOT_MAP = {
    // mask, /*dot value,*/ exp mask, frac bits, exp sign shift, exp bits, frac mask
    // dml: dotbits | expbits | fracbits
    // dml:    4    |    0    |    3
    {0b0000, /*0x0,*/ std::make_tuple(0x00, 0x3, 0x0, 0x0, 0x7)},
    // d0: dotbits | expbits | fracbits
    // d0:    4    |    0    |    3
    {0b0001, /*0x0,*/ std::make_tuple(0x00, 0x3, 0x1, 0x0, 0x7)},
    //d1: dotbits | expbits | fracbits
    //d1:    3    |    1    |    3
    {0b0010, /*0x1,*/ std::make_tuple(0x08, 0x3, 0x3, 0x1, 0x7)},
    {0b0011, std::make_tuple(0x08, 0x3, 0x3, 0x1, 0x7)},
    //d2: dotbits | expbits | fracbits
    //d2:    2    |    2    |    3
    {0b0100, /*0x2,*/ std::make_tuple(0x18, 0x3, 0x4, 0x2, 0x7)},
    {0b0101, std::make_tuple(0x18, 0x3, 0x4, 0x2, 0x7)},
    {0b0110, std::make_tuple(0x18, 0x3, 0x4, 0x2, 0x7)},
    {0b0111, std::make_tuple(0x18, 0x3, 0x4, 0x2, 0x7)},
    // d3: dotbits | expbits | fracbits
    // d3:    2    |    3    |    2
    {0b1000, /*0x3,*/ std::make_tuple(0x1c, 0x2, 0x4, 0x3, 0x3)},
    {0b1001, std::make_tuple(0x1c, 0x2, 0x4, 0x3, 0x3)},
    {0b1010, std::make_tuple(0x1c, 0x2, 0x4, 0x3, 0x3)},
    {0b1011, std::make_tuple(0x1c, 0x2, 0x4, 0x3, 0x3)},
    // d4: dotbits | expbits | fracbits
    // d4:    2    |    4    |    1
    {0b1100, /*0x4,*/ std::make_tuple(0x1e, 0x1, 0x4, 0x4, 0x1)},
    {0b1101, std::make_tuple(0x1e, 0x1, 0x4, 0x4, 0x1)},
    {0b1110, std::make_tuple(0x1e, 0x1, 0x4, 0x4, 0x1)},
    {0b1111, std::make_tuple(0x1e, 0x1, 0x4, 0x4, 0x1)},
};

struct HiF8CastParam {
    int32_t expNoBias;
    uint32_t dotValue;
    uint32_t expHiF8Bits;
    uint32_t fracHiF8Bits;
    uint32_t fracBits;
    uint32_t fracFpN;
    uint32_t fracHiF8;
};

// hifloat8 dot/exp/frac bits num
void GetHiF8BitsNum(int32_t expNoBias, uint32_t& dotValue, uint32_t& expBits, uint32_t& fracBits)
{
    uint32_t absExpNoBias = abs(expNoBias);
    if (expNoBias == 0) { // d0 e:[0]
        dotValue = 0x01; // d0 0b0001
        // d0: dotbits | expbits | fracbits
        // d0:    4    |    0    |    3
        expBits = 0;
        fracBits = 3;
    } else if (absExpNoBias == 1) { // d3 e:[1]
        dotValue = 0x02; // d1 0b0010
        //d1: dotbits | expbits | fracbits
        //d1:    3    |    1    |    3
        expBits = 1;
        fracBits = 3;
    } else if (absExpNoBias >= 2 && absExpNoBias <= 3) { // d3 e:[2, 3]
        dotValue = 0x04; // d2 0b0100
        //d2: dotbits | expbits | fracbits
        //d2:    2    |    2    |    3
        expBits = 2;
        fracBits = 3;
    } else if (absExpNoBias >= 4 && absExpNoBias <= 7) { // d3 e:[4, 7]
        dotValue = 0x08; // d3 0b1000
        //d3: dotbits | expbits | fracbits
        //d3:    2    |    3    |    2
        expBits = 3;
        fracBits = 2;
    } else if (absExpNoBias >= 8 && absExpNoBias <= 15) { // d4 e:[8, 15]
        dotValue = 0x0C; // d4 0b1100
        //d4: dotbits | expbits | fracbits
        //d4:    2    |    4    |    1
        expBits = 4;
        fracBits = 1;
    }
}

template <typename T>
T HiF8CastByDot(uint8_t inData, uint32_t expBits, uint32_t fracBits)
{
    uint8_t dot = (inData & 0x78) >> 3;
    auto bitsInfo = HIF8_DOT_MAP.at(dot);
    auto expMask = std::get<0>(bitsInfo);
    auto fracHiF8Bits = std::get<1>(bitsInfo);
    auto expSignShift = std::get<2>(bitsInfo);
    auto expHiF8Bits = std::get<3>(bitsInfo);
    auto fracMask = std::get<4>(bitsInfo);

    uint32_t expHiF8 = (inData & expMask) >> fracHiF8Bits;
    uint32_t signExp = (inData >> expSignShift) & 0x01;
    int32_t expNoBias = 0;
    uint32_t expFpN = 0;
    uint32_t fracFpN = inData & fracMask;
    if (expSignShift == 0) {
        expNoBias = -(HIF8_DENORMAL_EXP_BIAS - (inData & fracMask));
        expFpN = expNoBias + ((1 << (expBits - 1)) - 1);
        fracFpN = 0;
    } else if (expSignShift == 1) {
        expFpN = ((1 << (expBits - 1)) - 1);
    } else {
        expNoBias = (expHiF8 | (1 << (expHiF8Bits - 1))) * (signExp == 1 ? -1 : 1);
        expFpN = expNoBias + ((1 << (expBits - 1)) - 1);
    }
    uint32_t sign = inData >> HIF8_SIGN_SHIFT;
    T outData = sign << (expBits + fracBits) | (expFpN << fracBits) | (fracFpN << (fracBits - fracHiF8Bits));
    return outData;
}


// hifloat8 value to float32/float16/bfloat16
// for float32/float16/bfloat16, the differences are exp bits num & frac bits num
template <typename T>
T HiF8ToFpN(uint8_t inData, uint32_t expBits, uint32_t fracBits)
{
    T outData = 0;
    if (inData == 0) {
        outData = 0;
    } else if (inData == 0x80) { // nan 10000000
        outData = (1 << (expBits + fracBits)) - 1;
    } else if (inData == 0x6F) { // +inf exp all 1
        outData = ((1 << expBits) - 1) << fracBits;
    } else if (inData == 0xEF) { //-inf 
        outData = ((1 << (expBits + 1)) - 1) << fracBits; // s 1 exp expBits*1 frac 0
    } else {
        outData = HiF8CastByDot<T>(inData, expBits, fracBits);
    }
    return outData;
}

// hifloat8 rounding process
void RoundHiF8(HiF8CastParam& hif8Param)
{
    uint32_t fractmp = hif8Param.fracFpN >> (hif8Param.fracBits - hif8Param.fracHiF8Bits - 1); // fracHiF8 + 1bit
    uint32_t fracMax = (1 << (hif8Param.fracHiF8Bits + 1)) - 1;
    if (fractmp == fracMax) {
        hif8Param.fracHiF8 = 0;
        hif8Param.expNoBias += 1;
        GetHiF8BitsNum(hif8Param.expNoBias, hif8Param.dotValue, hif8Param.expHiF8Bits, hif8Param.fracHiF8Bits);
    } else if ((fractmp & 0x01) == 0x01) {
        hif8Param.fracHiF8 = (fractmp + 1) >> 1;
    } else {
        hif8Param.fracHiF8 = fractmp >> 1;
    }
}

template <typename T>
uint8_t FpNToHiF8(T inData, uint32_t expBits, uint32_t fracBits)
{
    uint8_t outData = 0;
    uint32_t sign = inData >> (expBits + fracBits);
    uint32_t expFpN = (inData & ((1 << (expBits + fracBits)) - 1)) >> fracBits;
    uint32_t fracFpN = inData & ((1 << fracBits) - 1);
    uint32_t fracHiF8 = 0;
    int32_t expNoBias = 0;
    if (expFpN == ((1U << expBits) - 1) && fracFpN > 0) {
        outData = 0x80; //nan 0b10000000
    } else if (expFpN == ((1U << expBits) - 1) && fracFpN == 0) {
        outData = (sign << HIF8_SIGN_SHIFT) | 0x6F; // inf S1101111
    } else {
        expNoBias = expFpN - ((1 << (expBits - 1)) - 1);
        if (expNoBias < -22) {
            outData = 0x00; // FpN value < 2^-22, hifloat8 value = 0
        } else if (expNoBias >= 15) {
            outData = (sign << HIF8_SIGN_SHIFT) | 0x6E; // MAX hifloat8 0b01101110
        }  else if (expNoBias >= -22 && expNoBias < -15) { // hifloat8 denormal
            if ((fracFpN >> (fracBits - 1)) >= 1) {
                fracHiF8 = HIF8_DENORMAL_EXP_BIAS + expNoBias + 1;
            } else {
                fracHiF8 = HIF8_DENORMAL_EXP_BIAS + expNoBias;
            }
            outData = (sign << HIF8_SIGN_SHIFT) | fracHiF8;
        } else {
            uint32_t dotValue = 0;
            uint32_t expHiF8Bits = 0;
            uint32_t fracHiF8Bits = 0;
            GetHiF8BitsNum(expNoBias, dotValue, expHiF8Bits, fracHiF8Bits);
            HiF8CastParam hif8Param = {expNoBias, dotValue, expHiF8Bits, fracHiF8Bits, fracBits, fracFpN, fracHiF8};
            RoundHiF8(hif8Param);
            uint32_t signExp = hif8Param.expNoBias < 0 ? 1 : 0;
            if (hif8Param.dotValue == 1) {
                outData = (sign << HIF8_SIGN_SHIFT) | 
                    (hif8Param.dotValue << hif8Param.fracHiF8Bits) | hif8Param.fracHiF8;
            } else {
                // MSB of the magnitude fixed to 1
                uint32_t expHiF8 = abs(hif8Param.expNoBias) - (1 << (hif8Param.expHiF8Bits - 1));
                outData = (sign << HIF8_SIGN_SHIFT) | (hif8Param.dotValue << 3 )
                    | (signExp << (hif8Param.expHiF8Bits + hif8Param.fracHiF8Bits - 1)) 
                    | (expHiF8 << hif8Param.fracHiF8Bits) | hif8Param.fracHiF8;
            }
        }
    }
    return outData;
}

std::pair<uint32_t, uint32_t> GetExpFracBits(int dataType)
{
    uint32_t expBits = FP32_EXP_BITS;
    uint32_t fracBits = FP32_FRAC_BITS;
    if (dataType == FP16) {
        expBits = FP16_EXP_BITS;
        fracBits = FP16_FRAC_BITS;
    } else if (dataType == BF16) {
        expBits = BF16_EXP_BITS;
        fracBits = BF16_FRAC_BITS;
    }
    return std::make_pair(expBits, fracBits);
}

template <typename T>
struct HiF8CastToFpNFunctor {
    void operator()(const uint8_t* in, T* out, int length, int dstType) const
    {
        auto expManBits = GetExpFracBits(dstType);
#pragma omp parallel for
        for (auto i = 0; i < length; i++) {
            out[i] = HiF8ToFpN<T>(in[i], expManBits.first, expManBits.second);
        }
    }
};

template <typename T>
struct DataCastToHiF8Functor {
    void operator()(const T* in, uint8_t* out, int length, int srcType) const
    {
        auto expManBits = GetExpFracBits(srcType);
#pragma omp parallel for
        for (auto i = 0; i < length; i++) {
            out[i] = FpNToHiF8<T>(in[i], expManBits.first, expManBits.second);
        }
    }
};

template struct DataCastToHiF8Functor<uint32_t>;
template struct DataCastToHiF8Functor<uint16_t>;
template struct HiF8CastToFpNFunctor<uint32_t>;
template struct HiF8CastToFpNFunctor<uint16_t>;


torch::Tensor FloatToHifloat8(torch::Tensor& input)
{
    input = input.contiguous();
    torch::Tensor output = torch::zeros_like(input).to(torch::kUInt8);
    auto outputPtr = reinterpret_cast<uint8_t*>(output.data_ptr());
    if (input.dtype() == torch::kFloat32) {
        auto inputPtr = reinterpret_cast<uint32_t*>(input.data_ptr());
        DataCastToHiF8Functor<uint32_t>()(inputPtr, outputPtr, input.numel(), FP32);
    } else {
        auto inputPtr = reinterpret_cast<uint16_t*>(input.data_ptr());
        int dtype = (input.dtype() == torch::kFloat16) ? FP16 : BF16;
        DataCastToHiF8Functor<uint16_t>()(inputPtr, outputPtr, input.numel(), dtype);
    }
    return output;
}

torch::Tensor Hifloat8ToFloat32(torch::Tensor& input)
{
    input = input.contiguous();
    auto inputPtr = reinterpret_cast<uint8_t*>(input.data_ptr());
    torch::Tensor output = torch::zeros_like(input.to(torch::kFloat32));
    auto outputPtr = reinterpret_cast<uint32_t*>(output.data_ptr());

    HiF8CastToFpNFunctor<uint32_t>()(inputPtr, outputPtr, input.numel(), FP32);
    return output;
}

torch::Tensor Hifloat8ToFloat16(torch::Tensor& input)
{
    input = input.contiguous();
    auto inputPtr = reinterpret_cast<uint8_t*>(input.data_ptr());
    torch::Tensor output = torch::zeros_like(input.to(torch::kFloat16));
    auto outputPtr = reinterpret_cast<uint16_t*>(output.data_ptr());

    HiF8CastToFpNFunctor<uint16_t>()(inputPtr, outputPtr, input.numel(), FP16);
    return output;
}

torch::Tensor Hifloat8ToBFloat16(torch::Tensor& input)
{
    input = input.contiguous();
    auto inputPtr = reinterpret_cast<uint8_t*>(input.data_ptr());
    torch::Tensor output = torch::zeros_like(input.to(torch::kBFloat16));
    auto outputPtr = reinterpret_cast<uint16_t*>(output.data_ptr());

    HiF8CastToFpNFunctor<uint16_t>()(inputPtr, outputPtr, input.numel(), BF16);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // hifloat8_tensor = float_to_hifloat8(float32_tensor)
    // hifloat8_tensor = float_to_hifloat8(float16_tensor)
    // hifloat8_tensor = float_to_hifloat8(bfloat16_tensor)

    // float32_tensor  = hifloat8_to_float32(hifloat8_tensor)
    // float16_tensor  = hifloat8_to_float16(hifloat8_tensor)
    // bfloat16_tensor = hifloat8_to_bfloat16(hifloat8_tensor)
    m.doc() = "hifloat8 to float32/float16/bfloat16 and reverse";
    m.def("float_to_hifloat8",    &FloatToHifloat8,    "Cast float32/float16/bfloat16 to hifloat8");
    m.def("hifloat8_to_float32",  &Hifloat8ToFloat32,  "convert hifloat8 to float32");
    m.def("hifloat8_to_float16",  &Hifloat8ToFloat16,  "convert hifloat8 to float16");
    m.def("hifloat8_to_bfloat16", &Hifloat8ToBFloat16, "convert hifloat8 to bfloat16");
}

