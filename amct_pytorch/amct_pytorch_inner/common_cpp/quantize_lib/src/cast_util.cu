/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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
 * @brief common CUDA C++ implementation for GPU.
 *
 * @file cast_util.cu
 *
 * @version 1.0
 */
#if GOOGLE_CUDA

#include <math.h>

#include "cast_util.h"
#include "util.h"

namespace util {
__global__ void CastToFloat32(const uint16_t* in, float* out, int length)
{
    CUDA_KERNEL_LOOP(index, length) {
        uint32_t sign = in[index] >> FP16_SIGN_SHIFT;
        uint32_t exponent = (in[index] >> FP16_EXP_SHIFT) & 0x1F;
        uint32_t fraction = (in[index] & 0x3FF);
        uint32_t outputData;
        if (exponent == 0) {
            if (fraction == 0) {
                outputData = (sign << FP32_SIGN_SHIFT);
            } else {
                exponent = FP32_DENORMAL_EXP;
                while ((fraction & (1 << FP16_EXP_SHIFT)) == 0) {
                    exponent--;
                    fraction <<= 1;
                }
                fraction &= 0x3FF;
                outputData = (sign << FP32_SIGN_SHIFT) | (exponent << FP32_EXP_SHIFT) | (fraction << FP32_FRAC_SHIFT);
            }
        } else if (exponent == 0x1F) {
            outputData = (sign << FP32_SIGN_SHIFT) | (0xFF << FP32_EXP_SHIFT) | (fraction << FP32_FRAC_SHIFT);
        } else {
            outputData = (sign << FP32_SIGN_SHIFT) | ((exponent + FP32_NORMAL_EXP) << FP32_EXP_SHIFT) |
                         (fraction << FP32_FRAC_SHIFT);
        }
        out[index] = *reinterpret_cast<float*> (&outputData);
    }
}

__global__ void CastToFloat(const float* in, uint16_t* out, int length)
{
    CUDA_KERNEL_LOOP(index, length) {
        uint32_t indata = *reinterpret_cast<const uint32_t*> (&in[index]);
        uint32_t sign = indata & 0x80000000u;
        uint32_t exponent = indata & 0x7F800000u;
        uint32_t valueBin = indata & 0x7FFFFFFFu;

        sign >>= (FP16_SIGN_SHIFT + 1);
        valueBin >>= FP32_FRAC_SHIFT;
        valueBin -= 0x1C000;

        valueBin = (exponent < 0x38800000u) ? 0 : valueBin;
        valueBin = (exponent > 0x8e000000u) ? 0x7bff : valueBin;
        valueBin = (exponent == 0 ? 0 : valueBin);

        valueBin |= sign;
        uint16_t innerValue = static_cast<uint16_t> (valueBin);
        out[index] = innerValue;
    }
}

__device__ void GetHiF8BitsNumGPU(int32_t expNoBias, uint32_t& dotValue, uint32_t& expBits, uint32_t& fracBits)
{
    uint32_t absExpNoBias = abs(expNoBias);
    if (expNoBias == 0) { // d0 e:[0]
        dotValue = 0x01; // d0 0b0001
        // d0: dotbits | expbits | fracbits
        // d0:    4    |    0    |    3
        expBits = 0;
        fracBits = 3;
    } else if (absExpNoBias == 1) {   // d3 e:[1]
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

__device__ void HybridHiF8GPU(HiF8CastParam& hif8Param)
{
    uint32_t absExpNoBias = abs(hif8Param.expNoBias);
    uint32_t fracMax = (1 << (hif8Param.fracHiF8Bits + 1)) - 1;
    if (absExpNoBias < 0x04) {
        uint32_t fractmp = hif8Param.fracFpN >> (hif8Param.fracBits - hif8Param.fracHiF8Bits - 1); // fracHiF8 + 1bit
        if (fractmp == fracMax) {
            hif8Param.fracHiF8 = 0;
            hif8Param.expNoBias += 1;
            GetHiF8BitsNumGPU(hif8Param.expNoBias, hif8Param.dotValue, hif8Param.expHiF8Bits, hif8Param.fracHiF8Bits);
        } else if ((fractmp & 0x01) == 0x01) {
            hif8Param.fracHiF8 = (fractmp + 1) >> 1;
        } else {
            hif8Param.fracHiF8 = fractmp >> 1;
        }
    } else {
        hif8Param.fracHiF8 = hif8Param.fracFpN >> (hif8Param.fracBits - hif8Param.fracHiF8Bits);
        uint32_t fNtN = hif8Param.fracFpN - (hif8Param.fracHiF8 << (hif8Param.fracBits - hif8Param.fracHiF8Bits));
        uint32_t Nbias = (hif8Param.fracBits == FP32_FRAC_BITS) ? FP32_THRESHOLD_BITS : FP16_THRESHOLD_BITS;
        uint32_t fNValue = fNtN >> (hif8Param.fracBits - hif8Param.fracHiF8Bits - Nbias);
        uint32_t tMask = (1 << Nbias) - 1;
        uint32_t tNValue = fNtN & tMask;
        if (fNValue >= tNValue) {
            if (hif8Param.fracHiF8 == ((1 << hif8Param.fracHiF8Bits)- 1)) {
                hif8Param.fracHiF8 = 0;
                hif8Param.expNoBias += 1;
                GetHiF8BitsNumGPU(hif8Param.expNoBias, 
                    hif8Param.dotValue, hif8Param.expHiF8Bits, hif8Param.fracHiF8Bits);
            }
        }
    }
}

__device__ void RoundHiF8GPU(HiF8CastParam& hif8Param)
{
    uint32_t fractmp = hif8Param.fracFpN >> (hif8Param.fracBits - hif8Param.fracHiF8Bits - 1); // fracHiF8 + 1bit
    uint32_t fracMax = (1 << (hif8Param.fracHiF8Bits + 1)) - 1;
    if (fractmp == fracMax) {
        hif8Param.fracHiF8 = 0;
        hif8Param.expNoBias += 1;
        GetHiF8BitsNumGPU(hif8Param.expNoBias, hif8Param.dotValue, hif8Param.expHiF8Bits, hif8Param.fracHiF8Bits);
    } else if ((fractmp & 0x01) == 0x01) {
        hif8Param.fracHiF8 = (fractmp + 1) >> 1;
    } else {
        hif8Param.fracHiF8 = fractmp >> 1;
    }
}

template <typename T>
__device__ void HiF8CastByDotGPU(uint8_t inData, T& outData, uint32_t expBits, uint32_t fracBits) 
{
    int32_t expFpN = 0;
    uint32_t fracFpN = 0;
    uint32_t fracHiF8Bits = 0;
    uint32_t dot = (inData & 0x78) >> 3; // 0x78: 111 1000,get dot value
    if (dot >= 0x0C) {
        //d4: dotbits | expbits | fracbits
        //d4:    2    |    4    |    1
        uint32_t expHiF8 = (inData & 0x1E) >> 1; // 0x1E: 11110
        uint32_t signExp = (inData >> 0x04) & 0x01;
        int32_t expNoBias = (expHiF8 | 0x08) * (signExp == 1 ? -1 : 1); // expNoBias=(expHiF8|2**3)*sign
        expFpN = expNoBias + ((1 << (expBits - 1)) - 1); // bias FP32: 127, FP16: 15 BF16: 127
        fracFpN = inData & 0x01; // 0x01: 1bit
        fracHiF8Bits = 1;
    } else if (dot >= 0x08) {
        //d3: dotbits | expbits | fracbits
        //d3:    2    |    3    |    2
        uint32_t expHiF8 = (inData & 0x1C) >> 2; // 0x1C: 11100
        uint32_t signExp = (inData >> 0x04) & 0x01;;
        int32_t expNoBias = (expHiF8 | 0x04) * (signExp == 1 ? -1 : 1); // expNoBias=(expHiF8|2**2)*sign
        expFpN = expNoBias + ((1 << (expBits - 1)) - 1);
        fracFpN = inData & 0x03; // 0x03: 11
        fracHiF8Bits = 0x02;
    } else if (dot >= 0x04) {
        //d2: dotbits | expbits | fracbits
        //d2:    2    |    2    |    3
        uint32_t expHiF8 = (inData & 0x18) >> 3; // 0x18: 11000
        uint32_t signExp = (inData >> 0x04) & 0x01;;
        int32_t expNoBias = (expHiF8 | 0x02) * (signExp == 1 ? -1 : 1); // expNoBias=(expHiF8|2**1)*sign
        expFpN = expNoBias + ((1 << (expBits - 1)) - 1);
        fracFpN = inData & 0x07; // 0x07: 111
        fracHiF8Bits = 0x03;
    } else if (dot >= 0x02) {
        //d1: dotbits | expbits | fracbits
        //d1:    3    |    1    |    3
        uint32_t expHiF8 = (inData & 0x08) >> 3; // 0x08: 1000
        uint32_t signExp = (inData >> 0x03) & 0x01;
        int32_t expNoBias = (expHiF8 | 0x01) * (signExp == 1 ? -1 : 1);
        expFpN = expNoBias + ((1 << (expBits - 1)) - 1);
        fracFpN = inData & 0x07; // 0x07: 111
        fracHiF8Bits = 0x03;
    } else if (dot == 0x01) {
        // d0: dotbits | expbits | fracbits
        // d0:    4    |    0    |    3
        expFpN = ((1 << (expBits - 1)) - 1);
        fracFpN = inData & 0x07; // 0x07: 111
        fracHiF8Bits = 3;
    } else if (dot == 0x00) {
        // dml: dotbits | expbits | fracbits
        // dml:    4    |    0    |    3
        // dml value = (-1) ^ sign * 2 ^ (M - 23)
        int32_t expNoBias = -(HIF8_DENORMAL_EXP_BIAS - (inData & 0x07)); // HIF8_DENORMAL_EXP_BIAS = 23
        expFpN = expNoBias + ((1 << (expBits - 1)) - 1);
        fracFpN = 0x00;
        fracHiF8Bits = 0x03;
    }
    uint32_t sign = inData >> HIF8_SIGN_SHIFT;
    outData = sign << (expBits + fracBits) | (expFpN << fracBits) | (fracFpN << (fracBits - fracHiF8Bits));
}

template <typename T>
__global__ void HiF8ToFpNCUDA(const uint8_t* in, T* out, int length, uint32_t expBits, uint32_t fracBits)
{
    CUDA_KERNEL_LOOP(index, length) {
        uint8_t inData = in[index];
        T outData = 0;
        if (inData == 0) {
            outData = 0;
        } else if (inData == 0x80) { //nan 10000000
            outData = (1 << (expBits + fracBits)) - 1;
        } else if (inData == 0x6F) { //+inf exp all 1
            outData = ((1 << expBits) - 1) << fracBits;
        } else if (inData == 0xEF) { //-inf 
            outData = ((1 << (expBits + 1)) - 1) << fracBits;// s 1 exp expBits*1 frac 0
        } else {
            HiF8CastByDotGPU(inData, outData, expBits, fracBits);
        }
        out[index] = outData;
    }
}

template <typename T>
__global__ void FpNToHiF8CUDA(const T* in, uint8_t* out, int length, uint32_t expBits, uint32_t fracBits, int roundMode)
{
    CUDA_KERNEL_LOOP(index, length) {
        T inData = in[index];
        uint8_t outData = 0;
        uint32_t sign = inData >> (expBits + fracBits);
        uint32_t expFpN = (inData & ((1 << (expBits + fracBits)) - 1)) >> fracBits;
        uint32_t fracFpN = inData & ((1 << fracBits) - 1);
        uint32_t fracHiF8 = 0;
        int32_t expNoBias = 0;
        if (expFpN == ((1 << expBits) - 1) && fracFpN > 0) {
            outData = 0x80; //nan 0b10000000
        } else if (expFpN == ((1 << expBits) - 1) && fracFpN == 0) {
            outData = sign << HIF8_SIGN_SHIFT | 0x6F; // inf S1101111
        } else {
            expNoBias = expFpN - ((1 << (expBits - 1)) - 1);
            if (expNoBias < -22) {
                outData = 0x00; // FpN value < 2^-22, HiF8 value = 0
            } else if (expNoBias >= 15) {
                outData = sign << HIF8_SIGN_SHIFT | 0x6E; //MAX HiF8 0b01101110
            }  else if (expNoBias >= -22 && expNoBias < -15) { //HiF8 denormal
                if ((fracFpN >> (fracBits - 1)) >= 1) {
                    fracHiF8 = HIF8_DENORMAL_EXP_BIAS + expNoBias + 1 ;
                } else {
                    fracHiF8 = HIF8_DENORMAL_EXP_BIAS + expNoBias;
                }
                outData = sign << HIF8_SIGN_SHIFT | fracHiF8;
            } else {
                uint32_t dotValue = 0;
                uint32_t expHiF8Bits = 0;
                uint32_t fracHiF8Bits = 0;
                GetHiF8BitsNumGPU(expNoBias, dotValue, expHiF8Bits, fracHiF8Bits);
                HiF8CastParam hif8Param = {expNoBias, dotValue, expHiF8Bits, fracHiF8Bits, fracBits, fracFpN, fracHiF8};
                if (roundMode == ROUND) {
                    RoundHiF8GPU(hif8Param);
                } else {
                    HybridHiF8GPU(hif8Param);
                }
                uint32_t signExp = hif8Param.expNoBias < 0 ? 1 : 0;
                if (hif8Param.dotValue == 1) {
                    outData = (sign << HIF8_SIGN_SHIFT) | 
                        (hif8Param.dotValue << hif8Param.fracHiF8Bits) | hif8Param.fracHiF8;
                } else {
                    // MSB of the magnitude fixed to 1
                    uint32_t expHiF8 = abs(hif8Param.expNoBias) - (1 << (hif8Param.expHiF8Bits - 1));
                    outData = (sign << HIF8_SIGN_SHIFT) | (hif8Param.dotValue << 3) 
                        | (signExp << (hif8Param.expHiF8Bits + hif8Param.fracHiF8Bits - 1)) 
                        | (expHiF8 << hif8Param.fracHiF8Bits) | hif8Param.fracHiF8;
                }
            }
        }
        out[index] = outData;
    }
}

template <typename T>
__global__ void Fp8ToFpNCUDA(const uint8_t* in, T* out, int length, uint32_t expBits, uint32_t fracBits)
{
    CUDA_KERNEL_LOOP(index, length) {
        uint8_t inData = in[index];
        uint16_t sign = inData >> FP8_SIGN_SHIFT;
        uint16_t expFp8 = (inData >> FP8_FRAC_BITS) & 0x0F;
        uint16_t fracFp8 = inData & 0x07;
        uint16_t expFpN = 0;
        T outData = 0;
        if (inData == 0) {
            outData = 0;
        } else if (inData == 0x7F || inData == 0xFF) { //nan 0b1111111
            outData =  (1 << (expBits + fracBits)) - 1;
        } else if (expFp8 == 0x00) {
            // denormal
            if (fracFp8 >= 4) {
                // 2**-7
                expFpN = ((1 << (expBits - 1)) - 1) - 7;
                fracFp8 = (fracFp8 & 0x03) << 1;
            } else if (fracFp8 >= 2) {
                // 2**-8
                expFpN = ((1 << (expBits - 1)) - 1) - 8;
                fracFp8 = (fracFp8 & 0x01) << 2;
            } else if (fracFp8 >= 1) {
                // 2**-9
                expFpN = ((1 << (expBits - 1)) - 1) - 9;
                fracFp8 = 0;
            }
            outData = sign << (expBits + fracBits) | expFpN << fracBits | fracFp8 << (fracBits - FP8_FRAC_BITS);
        } else {
            // expFpN = expFp8 + expFpNbias - expFp8bias
            uint16_t expFpN = expFp8 + ((1 << (expBits - 1)) - 1) - FP8_EXP_BIAS;
            outData = sign << (expBits + fracBits) | expFpN << fracBits | fracFp8 << (fracBits - FP8_FRAC_BITS);
        }
        out[index] = outData;
    }
}

template <typename T>
__global__ void FpNToFp8CUDA(const T* in, uint8_t* out, int length, uint32_t expBits, uint32_t fracBits, int roundMode)
{
    CUDA_KERNEL_LOOP(index, length) {
        T inData = in[index];
        uint8_t outData = 0;
        uint32_t sign = inData >> (expBits + fracBits);
        uint32_t expOri = (inData & ((1 << (expBits + fracBits)) - 1)) >> fracBits;
        uint32_t fracOri = inData & ((1 << fracBits) - 1);
        // round-to-nearest-even: add last bit after truncated mantissa
        uint32_t tmp = inData + (1 << (fracBits - FP8_FRAC_BITS - 1));
        uint32_t expBias = (1 << (expBits - 1)) - 1; // mantissa
        int32_t expNoBias = expOri - expBias; //actual value
        uint32_t expFp8 = 0;
        uint32_t fracFp8 = 0;
        if (expOri == ((1 << expBits) - 1) && fracOri > 0) {
            outData = sign << FP8_SIGN_SHIFT | 0x7F; //nan S1111111
        } else if (expNoBias < -6) {
            // FP32 expbias is 127, FP32 fracBits is 23
            uint32_t tmp = (static_cast<uint32_t>(expNoBias + 127) << 23) | (fracOri << (23 - fracBits));
            CastTransData tmpCast;
            tmpCast.y = tmp;
            float absOriData = tmpCast.x;
            fracFp8 = static_cast<uint32_t>(rintf(absOriData * 64 * 8)); // *64 scale to [0, 1), *8 round to 3bit
            outData = (sign << FP8_SIGN_SHIFT) | fracFp8; // 2^(-7) * (1/2 + x)  FpN nomal,Fp8 denormal
        } else if (expNoBias < -10) {
            outData = 0;
        } else {
            uint8_t lsb = (fracOri >> (fracBits - FP8_FRAC_BITS)) & 0x1;
            uint8_t gr = (fracOri >> (fracBits - FP8_FRAC_BITS - 2)) & 0x3;
            uint32_t sticky = fracOri & ((1 << (fracBits - FP8_FRAC_BITS - 2)) - 1);
            uint8_t stickyBit = (sticky == 0) ? 0 : 1;
            uint8_t grs = (gr << 1) + stickyBit;
            uint32_t tmp = inData;
            if ((grs > 4) || ((grs == 4) && (lsb == 1))) { // carry up
                tmp += (1 << (fracBits - FP8_FRAC_BITS));
            }
            uint32_t fracFpN = tmp & ((1 << fracBits) - 1);
            fracFp8 = fracFpN >> (fracBits - FP8_FRAC_BITS);
            uint32_t expFpN = (tmp & ((1 << (expBits + fracBits)) - 1)) >> fracBits;
            expNoBias = expFpN - expBias;
            expFp8 = expNoBias + FP8_EXP_BIAS;
            if (expNoBias > 8 || (expNoBias == 8 && fracFp8 >= 6)) {
                outData = sign << FP8_SIGN_SHIFT | 0x7E; // max
            } else {
                outData = sign << FP8_SIGN_SHIFT | expFp8 << FP8_FRAC_BITS | fracFp8; // normal
            }
        }
        out[index] = outData;
    }
}

template <typename T>
__global__ void FloatToFP4E2M1CUDA(const T* in, uint8_t* out, int length, uint32_t expBits, uint32_t fracBits)
{
    CUDA_KERNEL_LOOP(index, length) {
        T inData = in[index];
        uint32_t signBit = (inData >> (expBits + fracBits));
        uint32_t expBias = (1 << (expBits - 1)) - 1;
        uint32_t expOri = (inData & ((1 << (expBits + fracBits)) - 1)) >> fracBits;
        uint32_t fracOri = inData & ((1 << fracBits) - 1);

        float value = 0.0f;
        if (expOri == 0) {
            // calculate denormal FP16/BF16 actual value: 2 ** (1 - expBias) * (0.0 + frac)
            value = std::ldexp(static_cast<float>(fracOri) / (1 << fracBits), 1 - static_cast<int32_t>(expBias));
        } else {
            // calculate normal FP16/BF16 actual value: 2 ** (exp) * (1.0 + frac)
            int16_t exp = expOri - expBias;
            value = std::ldexp(1.0f + static_cast<float>(fracOri) / (1 << fracBits), exp);
        }
        uint8_t outData = 0;
        if (value <= 0.25f) {           // 0
            outData = 0x0;
        } else if (value < 0.75f) {     // 0.5
            outData = 0x1;
        } else if (value <= 1.25f) {    // 1.0
            outData = 0x2;
        } else if (value < 1.75f) {     // 1.5
            outData = 0x3;
        } else if (value <= 2.5f) {     // 2.0
            outData = 0x4;
        } else if (value < 3.5f) {      // 3.0
            outData = 0x5;
        } else if (value <= 5.0f) {     // 4.0
            outData = 0x6;
        } else {                         // over max
            outData = 0x7;
        }
        out[index] = (signBit << FP4_SIGN_SHIFT) | outData;
    }
}

template <typename T>
__global__ void FP4E2M1ToFloatCUDA(const uint8_t* in, T* out, int length, uint32_t fracBits)
{
    CUDA_KERNEL_LOOP(index, length) {
        uint8_t inData = in[index];
        uint8_t sign = (inData >> FP4_SIGN_SHIFT) & 0x1; // fp4e2m1 sign bits
        uint8_t noSign = inData & 0x7; // fp4e2m1 no sign bits
        T outData = 0;
        if (fracBits == FP16_FRAC_BITS) {
            if (noSign == 0x0) {
                outData = 0;
            } else if (noSign == 0x1) { // 0.5
                outData = 0x3800u; // 0x3800: FP16
            } else if (noSign == 0x2) { // 1.0
                outData = 0x3c00u; // 0x3c00: FP16
            } else if (noSign == 0x3) { // 1.5
                outData = 0x3e00u; // 0x3e00: FP16
            } else if (noSign == 0x4) { // 2.0
                outData = 0x4000u; // 0x4000: FP16
            } else if (noSign == 0x5) { // 3.0
                outData = 0x4200u; // 0x4200: FP16
            } else if (noSign == 0x6) { // 4.0
                outData = 0x4400u; // 0x4400: FP16
            } else if (noSign == 0x7) { // 6.0
                outData = 0x4600u; // 0x4600: FP16
            }
            outData = (sign << FP16_SIGN_SHIFT) | outData;
        } else if (fracBits == BF16_FRAC_BITS) {
            if (noSign == 0x0) {
                outData = 0;
            } else if (noSign == 0x1) { // 0.5
                outData = 0x3f00u; // 0x3f00: BF16
            } else if (noSign == 0x2) { // 1.0
                outData = 0x3f80u; // 0x3f80: BF16
            } else if (noSign == 0x3) { // 1.5
                outData = 0x3fc0u; // 0x3fc0: BF16
            } else if (noSign == 0x4) { // 2.0
                outData = 0x4000u; // 0x4000: BF16
            } else if (noSign == 0x5) { // 3.0
                outData = 0x4040u; // 0x4040: BF16
            } else if (noSign == 0x6) { // 4.0
                outData = 0x4080u; // 0x4080: BF16
            } else if (noSign == 0x7) { // 6.0
                outData = 0x40c0u; // 0x40c0: BF16
            }
            outData = (sign << BF16_SIGN_SHIFT) | outData;
        }
        else if (fracBits == FP32_FRAC_BITS) {
            if (noSign == 0x0) {
                outData = 0;
            } else if (noSign == 0x1) { // 0.5
                outData = 0x3f000000u; // 0x3f000000: FP32
            } else if (noSign == 0x2) { // 1.0
                outData = 0x3f800000u; // 0x3f800000: FP32
            } else if (noSign == 0x3) { // 1.5
                outData = 0x3fc00000u; // 0x3fc00000: FP32
            } else if (noSign == 0x4) { // 2.0
                outData = 0x40000000u; // 0x40000000: FP32
            } else if (noSign == 0x5) { // 3.0
                outData = 0x40400000u; // 0x40400000: FP32
            } else if (noSign == 0x6) { // 4.0
                outData = 0x40800000u; // 0x40800000: FP32
            } else if (noSign == 0x7) { // 6.0
                outData = 0x40c00000u; // 00x40c00000: FP32
            }
            outData = (sign << FP32_SIGN_SHIFT) | outData;
        }
        out[index] = outData;
    }
}

template <typename T>
__global__ void FloatToFP4E1M2CUDA(const T* in, uint8_t* out, int length, uint32_t expBits, uint32_t fracBits)
{
    CUDA_KERNEL_LOOP(index, length) {
        T inData = in[index];
        uint32_t signBit = (inData >> (expBits + fracBits));
        uint32_t expBias = (1 << (expBits - 1)) - 1;
        uint32_t expOri = (inData & ((1 << (expBits + fracBits)) - 1)) >> fracBits;
        uint32_t fracOri = inData & ((1 << fracBits) - 1);

        float value = 0.0f;
        if (expOri == 0) {
            // calculate denormal FP16/BF16 actual value: 2 ** (1 - expBias) * (0.0 + frac)
            value = std::ldexp(static_cast<float>(fracOri) / (1 << fracBits), 1 - static_cast<int32_t>(expBias));
        } else {
            // calculate normal FP16/BF16 actual value: 2 ** (exp) * (1.0 + frac)
            int16_t exp = expOri - expBias;
            value = std::ldexp(1.0f + static_cast<float>(fracOri) / (1 << fracBits), exp);
        }
        uint8_t outData = 0;
        if (value <= 0.125f) {           // 0
            outData = 0x0;
        } else if (value < 0.375f) {     // 0.25
            outData = 0x1;
        } else if (value <= 0.625f) {    // 0.5
            outData = 0x2;
        } else if (value < 0.875f) {     // 0.75
            outData = 0x3;
        } else if (value <= 1.125f) {     // 1.0
            outData = 0x4;
        } else if (value < 1.375f) {      // 1.25
            outData = 0x5;
        } else if (value <= 1.625f) {     // 1.5
            outData = 0x6;
        } else {                         // over max
            outData = 0x7;
        }
        out[index] = (signBit << FP4_SIGN_SHIFT) | outData;
    }
}

template <typename T>
__global__ void FP4E1M2ToFloatCUDA(const uint8_t* in, T* out, int length, uint32_t fracBits)
{
    CUDA_KERNEL_LOOP(index, length) {
        uint8_t inData = in[index];
        uint8_t sign = (inData >> FP4_SIGN_SHIFT) & 0x1; // fp4e2m1 sign bits
        uint8_t noSign = inData & 0x7; // fp4e2m1 no sign bits
        T outData = 0;
        if (fracBits == FP16_FRAC_BITS) {
            if (noSign == 0x0) {
                outData = 0;
            } else if (noSign == 0x1) { // 0.25
                outData = 0x3400u; // 0x3400u: FP16
            } else if (noSign == 0x2) { // 0.5
                outData = 0x3800u; // 0x3800u: FP16
            } else if (noSign == 0x3) { // 0.75
                outData = 0x3a00u; // 0x3a00u: FP16
            } else if (noSign == 0x4) { // 1.0
                outData = 0x3c00u; // 0x3c00u: FP16
            } else if (noSign == 0x5) { // 1.25
                outData = 0x3d00u; // 0x3d00u: FP16
            } else if (noSign == 0x6) { // 1.5
                outData = 0x3e00u; // 0x3e00u: FP16
            } else if (noSign == 0x7) { // 1.75
                outData = 0x3f00u; // 0x3f00u: FP16
            }
            outData = (sign << FP16_SIGN_SHIFT) | outData;
        } else if (fracBits == BF16_FRAC_BITS) {
            if (noSign == 0x0) {
                outData = 0;
            } else if (noSign == 0x1) { // 0.25
                outData = 0x3e80u; // 0x3e80u: BF16
            } else if (noSign == 0x2) { // 0.5
                outData = 0x3f00u; // 0x3f00u: BF16
            } else if (noSign == 0x3) { // 0.75
                outData = 0x3f40u; // 0x3f40u: BF16
            } else if (noSign == 0x4) { // 1.0
                outData = 0x3f80u; // 0x3f80u: BF16
            } else if (noSign == 0x5) { // 1.25
                outData = 0x3fa0u; // 0x3fa0u: BF16
            } else if (noSign == 0x6) { // 1.5
                outData = 0x3fc0u; // 0x3fc0u: BF16
            } else if (noSign == 0x7) { // 1.75
                outData = 0x3fe0u; // 0x3fe0u: BF16
            }
            outData = (sign << BF16_SIGN_SHIFT) | outData;
        } else if (fracBits == FP32_FRAC_BITS) {
            if (noSign == 0x0) {
                outData = 0;
            } else if (noSign == 0x1) { // 0.25
                outData = 0x3e800000u; // 0x3e800000u: FP32
            } else if (noSign == 0x2) { // 0.5
                outData = 0x3f000000u; // 0x3f000000u: FP32
            } else if (noSign == 0x3) { // 0.75
                outData = 0x3f400000u; // 0x3f400000u: FP32
            } else if (noSign == 0x4) { // 1.0
                outData = 0x3f800000u; // 0x3f800000u: FP32
            } else if (noSign == 0x5) { // 1.25
                outData = 0x3fa00000u; // 0x3fa00000u: FP32
            } else if (noSign == 0x6) { // 1.5
                outData = 0x3fc00000u; // 0x3fc00000u: FP32
            } else if (noSign == 0x7) { // 1.75
                outData = 0x3fe00000u; // 0x3fe00000u: FP32
            }
            outData = (sign << FP32_SIGN_SHIFT) | outData;
        }
        out[index] = outData;
    }
}

template <typename T>
struct DataCastToFloat32Functor<GPUDevice, T> {
    void operator()(const T* in, float* out, int length) const
    {
        CastToFloat32<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length);
    }
};

template <typename T>
struct DataCastToFloat16Functor<GPUDevice, T> {
    void operator()(const T* in, uint16_t* out, int length) const
    {
        CastToFloat<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length);
    }
};
 


void GetExpFracBitsGPU(int dataType, uint32_t& expBits, uint32_t& fracBits)
{
    if (dataType == FP16) {
        expBits = FP16_EXP_BITS;
        fracBits = FP16_FRAC_BITS;
    } else if (dataType == BF16) {
        expBits = BF16_EXP_BITS;
        fracBits = BF16_FRAC_BITS;
    }
}

template <typename T>
struct DataCastToHiF8Fp8Functor<util::GPUDevice, T> {
    void operator()(const T* in, uint8_t* out, int length, int srcType, int dstType, int roundMode) const
    {
        uint32_t expBits = FP32_EXP_BITS;
        uint32_t fracBits = FP32_FRAC_BITS;
        GetExpFracBitsGPU(srcType, expBits, fracBits);
        if (dstType == FP8) {
            FpNToFp8CUDA<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length, 
                expBits, fracBits, roundMode);
        } else if (dstType == HIF8) {
            FpNToHiF8CUDA<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length, 
                expBits, fracBits, roundMode);
        }
    }
};

template <typename T>
struct HiF8Fp8CastToFpNFunctor<util::GPUDevice, T> {
    void operator()(const uint8_t* in, T* out, int length, int dstType, int srcType) const
    {
        uint32_t expBits = FP32_EXP_BITS;
        uint32_t fracBits = FP32_FRAC_BITS;
        GetExpFracBitsGPU(dstType, expBits, fracBits);
        if (srcType == FP8) {
            Fp8ToFpNCUDA<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length, expBits, fracBits);
        } else if (srcType == HIF8) {
            HiF8ToFpNCUDA<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length, expBits, fracBits);
        }
    }
};

template <typename T>
struct FloatCastToFP4E2M1Functor<util::GPUDevice, T> {
    void operator()(const T* in, uint8_t* out, int length, int srcType) const
    {
        uint32_t expBits = FP32_EXP_BITS;
        uint32_t fracBits = FP32_FRAC_BITS;
        GetExpFracBitsGPU(srcType, expBits, fracBits);
        FloatToFP4E2M1CUDA<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length, expBits, fracBits);
    }
};

template <typename T>
struct FP4E2M1CastToFloatFunctor<util::GPUDevice, T> {
    void operator()(const uint8_t* in, T* out, int length, int dstType) const
    {
        uint32_t expBits = FP32_EXP_BITS;
        uint32_t fracBits = FP32_FRAC_BITS;
        GetExpFracBitsGPU(dstType, expBits, fracBits);
        FP4E2M1ToFloatCUDA<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length, fracBits);
    }
};

template <typename T>
struct FloatCastToFP4E1M2Functor<util::GPUDevice, T> {
    void operator()(const T* in, uint8_t* out, int length, int srcType) const
    {
        uint32_t expBits = FP32_EXP_BITS;
        uint32_t fracBits = FP32_FRAC_BITS;
        GetExpFracBitsGPU(srcType, expBits, fracBits);
        FloatToFP4E1M2CUDA<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length, expBits, fracBits);
    }
};

template <typename T>
struct FP4E1M2CastToFloatFunctor<util::GPUDevice, T> {
    void operator()(const uint8_t* in, T* out, int length, int dstType) const
    {
        uint32_t expBits = FP32_EXP_BITS;
        uint32_t fracBits = FP32_FRAC_BITS;
        GetExpFracBitsGPU(dstType, expBits, fracBits);
        FP4E1M2ToFloatCUDA<<<AMCT_GET_BLOCKS(length), AMCT_CUDA_NUM_THREADS>>>(in, out, length, fracBits);
    }
};

template struct DataCastToFloat32Functor<GPUDevice, uint16_t>;
template struct DataCastToFloat16Functor<GPUDevice, float>;
template struct DataCastToHiF8Fp8Functor<util::GPUDevice, uint32_t>;
template struct DataCastToHiF8Fp8Functor<util::GPUDevice, uint16_t>;
template struct HiF8Fp8CastToFpNFunctor<util::GPUDevice, uint32_t>;
template struct HiF8Fp8CastToFpNFunctor<util::GPUDevice, uint16_t>;
template struct FloatCastToFP4E2M1Functor<util::GPUDevice, uint16_t>;
template struct FP4E2M1CastToFloatFunctor<util::GPUDevice, uint16_t>;
template struct FloatCastToFP4E2M1Functor<util::GPUDevice, uint32_t>;
template struct FP4E2M1CastToFloatFunctor<util::GPUDevice, uint32_t>;
template struct FloatCastToFP4E1M2Functor<util::GPUDevice, uint16_t>;
template struct FP4E1M2CastToFloatFunctor<util::GPUDevice, uint16_t>;
template struct FloatCastToFP4E1M2Functor<util::GPUDevice, uint32_t>;
template struct FP4E1M2CastToFloatFunctor<util::GPUDevice, uint32_t>;

} // util

#endif  // GOOGLE_CUDA
