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
 * @brief cast_util head file
 *
 * @file cast_util.h in common_cpp
 *
 * @version 1.0
 */

#ifndef CAST_UTIL_H
#define CAST_UTIL_H

#include <cstdint>

namespace util {

constexpr uint32_t FP16_SIGN_SHIFT = 15;
constexpr uint32_t BF16_SIGN_SHIFT = 15;
constexpr uint32_t FP16_EXP_SHIFT = 10;
constexpr uint32_t FP32_SIGN_SHIFT = 31;
constexpr uint32_t FP32_EXP_SHIFT = 23;
constexpr uint32_t FP32_EXP_BITS = 8;
constexpr uint32_t FP32_FRAC_BITS = 23;
constexpr uint32_t FP32_FRAC_SHIFT = 13;
constexpr uint32_t FP16_EXP_BITS = 5;
constexpr uint32_t FP16_FRAC_BITS = 10;
constexpr uint32_t BF16_EXP_SHIFT = 7;
constexpr uint32_t BF16_EXP_BITS = 8;
constexpr uint32_t BF16_FRAC_BITS = 7;
constexpr uint32_t FP32_DENORMAL_EXP = 127 - 14;
constexpr uint32_t FP32_NORMAL_EXP = 127 - 15;
constexpr uint32_t FP8_EXP_BIAS = 7;
constexpr uint32_t FP8_FRAC_BITS = 3;
constexpr uint32_t HIF8_SIGN_SHIFT = 7;
constexpr uint32_t FP8_SIGN_SHIFT = 7;
constexpr uint32_t HIF8_DENORMAL_EXP_BIAS = 23;
constexpr uint32_t FP32_THRESHOLD_BITS = 14;
constexpr uint32_t FP16_THRESHOLD_BITS = 2;
constexpr uint32_t FP4_SIGN_SHIFT = 3;
constexpr uint32_t FP4E2M1_EXP_BIAS = 2;
constexpr uint32_t FP4E2M1_FRAC_BIAS = 1;


union CastTransData {
    float x;
    uint32_t y;
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

float Fp16ToFp32(uint16_t inputData);
uint16_t Fp32ToFp16(float inputData);
float CastToFP16PrecisionCPU(float inputData);
float CastToS19CPU(float data);
float FakeFp16PrecisionDataCPU(float data);

template <typename Device, typename T>
struct DataCastToFloat32Functor {
    void operator()(const T* in, float* out, int length) const;
};

template <typename Device, typename T>
struct DataCastToFloat16Functor {
    void operator()(const T* in, uint16_t* out, int length) const;
};

template <typename Device, typename T>
struct DataCastToFp16Precision {
    void operator()(const T* in, float* out, int length) const;
};

template <typename Device, typename T>
struct DataCastToHiF8Fp8Functor {
    void operator()(const T* in, uint8_t* out, int length, int srcType, int dataType, int roundMode) const;
};

template <typename Device, typename T>
struct HiF8Fp8CastToFpNFunctor {
    void operator()(const uint8_t* in, T* out, int length, int dataType, int srcType) const;
};

template <typename Device, typename T>
struct FloatCastToFP4E2M1Functor {
    void operator()(const T* in, uint8_t* out, int length, int srcType) const;
};

template <typename Device, typename T>
struct FP4E2M1CastToFloatFunctor {
    void operator()(const uint8_t* in, T* out, int length, int dstType) const;
};

template <typename Device, typename T>
struct FloatCastToFP4E1M2Functor {
    void operator()(const T* in, uint8_t* out, int length, int srcType) const;
};

template <typename Device, typename T>
struct FP4E1M2CastToFloatFunctor {
    void operator()(const uint8_t* in, T* out, int length, int dstType) const;
};

}
#endif // CAST_UTIL_H
