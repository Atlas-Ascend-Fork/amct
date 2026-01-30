/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief cast_cu head file
 *
 * @file cast.cuh in common_cpp
 *
 * @version 1.0
 */
#ifndef CAST_CU_H
#define CAST_CU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "util.h"
#include "cast_util.h"

using namespace util;

__device__ inline float CastToFP16Precision(float inputData)
{
    if (inputData > MAX_FP16) {
        return MAX_FP16;
    } else if (inputData < -MAX_FP16) {
        return -MAX_FP16;
    } else if (inputData > -DENORMAL_FP16 && inputData <DENORMAL_FP16) {
        return 0.0f;
    }
    CastTransData fp32Value;
    fp32Value.x = inputData;
    uint32_t indata = fp32Value.y;
    uint32_t fp16Precison = 0;
    // 13 bits is 1,Guard bit
    if ((indata & 0x1000) == 0x1000) {
        uint32_t sign = indata >> FP32_SIGN_SHIFT;
        uint32_t exponent = (indata >> FP32_EXP_SHIFT) & 0xFF;
        uint32_t fraction = (indata & 0x7FE000) >> FP32_FRAC_SHIFT;
        uint32_t lowestPreservedBit = (indata & 0x2000) >> FP32_FRAC_SHIFT;
        uint32_t stickyRoundBit = (indata & 0xFFF) > 0;
        // Round the fraction using "Toward nearest, ties toward even"
        // 14 bits is 1 or 1-12 bits are not 0, carry
        if ((stickyRoundBit == 1) || (lowestPreservedBit == 1)) {
            fraction += 1;
        }
        // max fraction is 0x3ff, over 0x3ff, carry 1 to exponent
        if (fraction > 0x3FF) {
            exponent += 1;
        }
        fraction = fraction & 0x3FF;
        // Combine the sign, exponent, and fraction into a FP16 Precision and FP32 value
        fp16Precison = (sign << FP32_SIGN_SHIFT) | (exponent << FP32_EXP_SHIFT) | (fraction << FP32_FRAC_SHIFT);
    } else {
        fp16Precison = indata & 0xFFFFE000;
    }
    CastTransData outCast;
    outCast.y = fp16Precison;
    return outCast.x;
}

__device__ inline float CastToS19(float data)
{
    CastTransData fp32Data;
    fp32Data.x = data;
    fp32Data.y &= 0xFFFFE000;
    return fp32Data.x;
}

__device__ inline float FakeFp16PrecisionData(float data)
{
    float outputData = 0.0f;
    if (data < 0) {
        LOG_ERROR("scale_d data cannot be negative\n");
        outputData = NAN;
    }
    if (((data - MAX_FP16) > 0) || ((data - MIN_FP16) < 0)) {
        outputData = CastToFP16Precision(sqrt(data)) * CastToFP16Precision(sqrt(data));
    } else {
        outputData = CastToFP16Precision(data);
    }
    return outputData;
}

#endif // CAST_CU_H
