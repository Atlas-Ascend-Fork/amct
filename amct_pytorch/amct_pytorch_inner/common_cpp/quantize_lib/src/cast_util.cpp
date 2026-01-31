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
 * @brief cast data type C++ implement
 *
 * @file cast_util.cpp
 *
 * @version 1.0
 */

#include <cmath>
#include <utility>
#include <functional>
#include <map>
#include <tuple>

#include "util.h"
#include "cast_util.h"
namespace {
    using namespace util;
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

    const std::map<uint8_t, std::tuple<uint16_t, uint16_t, uint32_t>> FP4E2M1_TO_FLOAT_MAP = {
        // FP4E2M1 tuple(float16, bfloat16, float32)
        {0x0, std::make_tuple(0x0,     0x0,     0x0)},
        {0x1, std::make_tuple(0x3800u, 0x3f00u, 0x3f000000u)},
        {0x2, std::make_tuple(0x3c00u, 0x3f80u, 0x3f800000u)},
        {0x3, std::make_tuple(0x3e00u, 0x3fc0u, 0x3fc00000u)},
        {0x4, std::make_tuple(0x4000u, 0x4000u, 0x40000000u)},
        {0x5, std::make_tuple(0x4200u, 0x4040u, 0x40400000u)},
        {0x6, std::make_tuple(0x4400u, 0x4080u, 0x40800000u)},
        {0x7, std::make_tuple(0x4600u, 0x40c0u, 0x40c00000u)},
    };

    const std::map<uint8_t, std::tuple<uint16_t, uint16_t, uint32_t>> FP4E1M2_TO_FLOAT_MAP = {
        // FP4E1M2 tuple(float16, bfloat16, float32)
        {0x0, std::make_tuple(0x0,     0x0,     0x0)},
        {0x1, std::make_tuple(0x3400u, 0x3e80u, 0x3e800000u)},
        {0x2, std::make_tuple(0x3800u, 0x3f00u, 0x3f000000u)},
        {0x3, std::make_tuple(0x3a00u, 0x3f40u, 0x3f400000u)},
        {0x4, std::make_tuple(0x3c00u, 0x3f80u, 0x3f800000u)},
        {0x5, std::make_tuple(0x3d00u, 0x3fa0u, 0x3fa00000u)},
        {0x6, std::make_tuple(0x3e00u, 0x3fc0u, 0x3fc00000u)},
        {0x7, std::make_tuple(0x3f00u, 0x3fe0u, 0x3fe00000u)},
    };

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

    void HybridHiF8(HiF8CastParam& hif8Param)
    {
        uint32_t absExpNoBias = abs(hif8Param.expNoBias);
        uint32_t fracMax = (1 << (hif8Param.fracHiF8Bits + 1)) - 1;
        if (absExpNoBias < 0x04) {
            uint32_t fractmp = hif8Param.fracFpN >> 
                (hif8Param.fracBits - hif8Param.fracHiF8Bits - 1); // fracHiF8 + 1bit
            if (fractmp == fracMax) {
                hif8Param.fracHiF8 = 0;
                hif8Param.expNoBias += 1;
                GetHiF8BitsNum(hif8Param.expNoBias, hif8Param.dotValue, hif8Param.expHiF8Bits, hif8Param.fracHiF8Bits);
            } else if ((fractmp & 0x01) == 0x01) {
                hif8Param.fracHiF8 = (fractmp + 1) >> 1;
            } else {
                hif8Param.fracHiF8 = fractmp >> 1;
            }
        } else {
            hif8Param.fracHiF8 = hif8Param.fracFpN >> (hif8Param.fracBits - hif8Param.fracHiF8Bits);
            uint32_t fNtN = hif8Param.fracFpN - (hif8Param.fracHiF8 << (hif8Param.fracBits - hif8Param.fracHiF8Bits));
            uint32_t nBias = (hif8Param.fracBits == FP32_FRAC_BITS) ? FP32_THRESHOLD_BITS : FP16_THRESHOLD_BITS;
            uint32_t fNValue = fNtN >> (hif8Param.fracBits - hif8Param.fracHiF8Bits - nBias);
            uint32_t tMask = (1 << nBias) - 1;
            uint32_t tNValue = fNtN & tMask;
            if (fNValue >= tNValue) {
                if (hif8Param.fracHiF8 == ((1U << hif8Param.fracHiF8Bits)- 1)) {
                    hif8Param.fracHiF8 = 0;
                    hif8Param.expNoBias += 1;
                    GetHiF8BitsNum(hif8Param.expNoBias, hif8Param.dotValue,
                        hif8Param.expHiF8Bits, hif8Param.fracHiF8Bits);
                }
            }
        }
    }
}

namespace util {
    float Fp16ToFp32(uint16_t inputData)
    {
        uint32_t sign = inputData >> FP16_SIGN_SHIFT;
        uint32_t exponent = (inputData >> FP16_EXP_SHIFT) & 0x1F;
        uint32_t fraction = (inputData & 0x3FF);
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
                outputData =
                    (sign << FP32_SIGN_SHIFT) | (exponent << FP32_EXP_SHIFT) | (fraction << FP32_FRAC_SHIFT);
            }
        } else if (exponent == 0x1F) {
            outputData = (sign << FP32_SIGN_SHIFT) | (0xFF << FP32_EXP_SHIFT) | (fraction << FP32_FRAC_SHIFT);
        } else {
            outputData = (sign << FP32_SIGN_SHIFT) | ((exponent + (FP32_NORMAL_EXP)) << FP32_EXP_SHIFT) |
                         (fraction << FP32_FRAC_SHIFT);
        }
        CastTransData outCast;
        outCast.y = outputData;
        return outCast.x;
    }

    uint16_t Fp32ToFp16(float inputData)
    {
        CastTransData inCast;
        inCast.x = inputData;
        uint32_t indata = inCast.y;
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
        return innerValue;
    }

    float CastToFP16PrecisionCPU(float inputData)
    {
        if (inputData > MAX_FP16) {
            return MAX_FP16;
        } else if (inputData < -MAX_FP16) {
            return -MAX_FP16;
        } else if (inputData > -DENORMAL_FP16 && inputData < DENORMAL_FP16) {
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
    
    float CastToS19CPU(float data)
    {
        CastTransData fp32Data;
        fp32Data.x = data;
        fp32Data.y &= 0xFFFFE000;
        return fp32Data.x;
    }
    
    float FakeFp16PrecisionDataCPU(float data)
    {
        float outputData = 0.0f;
        if (data < 0) {
            LOG_ERROR("scale_d data cannot be negative\n");
            outputData = NAN;
        }
        if (((data - MAX_FP16) > 0) || ((data - MIN_FP16) < 0)) {
            outputData = CastToFP16PrecisionCPU(sqrt(data)) * CastToFP16PrecisionCPU(sqrt(data));
        } else {
            outputData = CastToFP16PrecisionCPU(data);
        }
        return outputData;
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

    template <typename T>
    uint8_t FpNToHiF8(T inData, uint32_t expBits, uint32_t fracBits, int roundMode)
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
                outData = 0x00; // FpN value < 2^-22, HiF8 value = 0
            } else if (expNoBias >= 15) {
                outData = (sign << HIF8_SIGN_SHIFT) | 0x6E; // MAX HiF8 0b01101110
            }  else if (expNoBias >= -22 && expNoBias < -15) { // HiF8 denormal
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
                if (roundMode == ROUND) {
                    RoundHiF8(hif8Param);
                } else {
                    HybridHiF8(hif8Param);
                }
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

    template <typename T>
    T Fp8ToFpN(uint8_t inData, uint32_t expBits, uint32_t fracBits)
    {
        T outData = 0;
        uint16_t sign = inData >> FP8_SIGN_SHIFT;
        uint16_t expFp8 = (inData >> FP8_FRAC_BITS) & 0x0F;
        uint16_t fracFp8 = inData & 0x07;
        uint16_t expFpN = 0;
        if (inData == 0) {
            outData = 0;
        } else if (inData == 0x7F || inData == 0xFF) { //nan 0b1111111
            outData = (1 << (expBits + fracBits)) - 1;
        } else if (expFp8 == 0x00) {
            // denormal
            if (fracFp8 >= 4) { // frac: 1xx
                expFpN = ((1 << (expBits - 1)) - 1) - 7; // 2**-7
                fracFp8 = (fracFp8 & 0x03) << 1; // fracFp8 covert to fracFpN
            } else if (fracFp8 >= 2) { // frac: 01x
                expFpN = ((1 << (expBits - 1)) - 1) - 8; // 2**-8
                fracFp8 = (fracFp8 & 0x01) << 2; // 
            } else if (fracFp8 >= 1) { // frac 001
                expFpN = ((1 << (expBits - 1)) - 1) - 9; // 2**-9
                fracFp8 = 0;
            }
            outData = sign << (expBits + fracBits) | expFpN << fracBits | fracFp8 << (fracBits - FP8_FRAC_BITS);
        } else {
            expFpN = expFp8 + ((1 << (expBits - 1)) - 1) - FP8_EXP_BIAS; // expFpN = expFp8 + expFpNbias - expFp8bias
            outData = sign << (expBits + fracBits) | expFpN << fracBits | fracFp8 << (fracBits - FP8_FRAC_BITS);
        }
        return outData;
    }

    template <typename T>
    uint8_t FpNToFp8(T inData, uint32_t expBits, uint32_t fracBits, int roundMode)
    {
        (void)roundMode;
        uint8_t outData = 0;
        uint32_t sign = inData >> (expBits + fracBits);
        uint32_t expOri = (inData & ((1 << (expBits + fracBits)) - 1)) >> fracBits;
        uint32_t fracOri = inData & ((1 << fracBits) - 1);
        uint32_t expBias = (1 << (expBits - 1)) - 1; // mantissa
        int32_t expNoBias = expOri - expBias; //actual value
        uint32_t expFp8 = 0;
        uint32_t fracFp8 = 0;
        if (expOri == ((1U << expBits) - 1U) && fracOri > 0) {
            outData = (sign << FP8_SIGN_SHIFT) | 0x7F; //nan S1111111
        } else if (expNoBias < -6) { //FP8 denormal
            // FP32 expbias is 127, FP32 fracBits is 23
            uint32_t tmp = (static_cast<uint32_t>(expNoBias + 127) << 23) | (fracOri << (23 - fracBits));
            CastTransData tmpCast;
            tmpCast.y = tmp;
            float absOriData = tmpCast.x;
            fracFp8 = static_cast<uint32_t>(rintf(absOriData * 64 * 8)); // *64 scale to [0, 1), *8 round to 3bit
            outData = (sign << FP8_SIGN_SHIFT) | fracFp8; // 2^(-7) * (1/2 + x)  FpN nomal,Fp8 denormal
        } else if (expNoBias < -10) { // min FP8 denormal
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
                outData = (sign << FP8_SIGN_SHIFT) | 0x7E; // max
            } else {
                outData = (sign << FP8_SIGN_SHIFT) | (expFp8 << FP8_FRAC_BITS )| fracFp8; // normal
            }
        }
        return outData;
    }

    template <typename T>
    uint8_t FloatToFP4E2M1(T inData, uint32_t expBits, uint32_t fracBits)
    {
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
        if (value <= 0.25f) {           // 0    0b00000000
            outData = 0x0;
        } else if (value < 0.75f) {     // 0.5  0b00000001
            outData = 0x1;
        } else if (value <= 1.25f) {    // 1.0  0b00000010
            outData = 0x2;
        } else if (value < 1.75f) {     // 1.5  0b00000011
            outData = 0x3;
        } else if (value <= 2.5f) {     // 2.0  0b00000100
            outData = 0x4;
        } else if (value < 3.5f) {      // 3.0  0b00000101
            outData = 0x5;
        } else if (value <= 5.0f) {     // 4.0  0b00000110
            outData = 0x6;
        } else {                         // over max 0b00000111
            outData = 0x7;
        }
        return (signBit << FP4_SIGN_SHIFT) | outData;
    }

    template <typename T>
    uint8_t FloatToFP4E1M2(T inData, uint32_t expBits, uint32_t fracBits)
    {
        uint8_t signBit = (inData >> (expBits + fracBits));
        int32_t expBias = (1 << (expBits - 1)) - 1;
        int32_t expOri = (inData & ((1 << (expBits + fracBits)) - 1)) >> fracBits;
        uint32_t fracOri = inData & ((1 << fracBits) - 1);

        float value = 0.0f;
        if (expOri == 0) {
            // calculate denormal FP16/BF16 actual value: 2 ** (1 - expBias) * (0.0 + frac)
            value = std::ldexp(static_cast<float>(fracOri) / (1 << fracBits), 1 - expBias);
        } else {
            // calculate normal FP16/BF16 actual value: 2 ** (exp) * (1.0 + frac)
            int16_t exp = expOri - expBias;
            value = std::ldexp(1.0f + static_cast<float>(fracOri) / (1 << fracBits), exp);
        }
        uint8_t outData = 0;
        if (value <= 0.125f) {           // 0     0b00000000
            outData = 0x0;
        } else if (value < 0.375f) {     // 0.25  0b00000001
            outData = 0x1;
        } else if (value <= 0.625f) {    // 0.5   0b00000010
            outData = 0x2;
        } else if (value < 0.875f) {     // 0.75  0b00000011
            outData = 0x3;
        } else if (value <= 1.125f) {     // 1.0  0b00000100
            outData = 0x4;
        } else if (value < 1.375f) {      // 1.25 0b00000101
            outData = 0x5;
        } else if (value <= 1.625f) {     // 1.5  0b00000110
            outData = 0x6;
        } else {                         // over max  0b00000111
            outData = 0x7;
        }
        return (signBit << FP4_SIGN_SHIFT) | outData;
    }

    template <typename T>
    T FP4E2M1ToFloat(uint8_t inData, uint32_t fracBits)
    {
        uint8_t sign = (inData >> FP4_SIGN_SHIFT) & 0x1; // fp4e2m1 sign bits
        uint8_t noSign = inData & 0x7; // fp4e2m1 no sign bits
        T outData = 0;
        auto bitsInfo = FP4E2M1_TO_FLOAT_MAP.at(noSign);
        if (fracBits == FP16_FRAC_BITS) {
            outData = std::get<0>(bitsInfo);
            outData = (sign << FP16_SIGN_SHIFT) | outData;
        } else if (fracBits == BF16_FRAC_BITS) {
            outData = std::get<1>(bitsInfo);
            outData = (sign << BF16_SIGN_SHIFT) | outData;
        } else if (fracBits == FP32_FRAC_BITS) {
            outData = std::get<2>(bitsInfo);
            outData = (sign << FP32_SIGN_SHIFT) | outData;
        }
        return outData;
    }

    template <typename T>
    T FP4E1M2ToFloat(uint8_t inData, uint32_t fracBits)
    {
        T sign = (inData >> FP4_SIGN_SHIFT) & 0x1; // fp4e1m2 sign bits
        uint8_t noSign = inData & 0x7; // fp4e1m2 no sign bits
        T outData = 0;
        auto bitsInfo = FP4E1M2_TO_FLOAT_MAP.at(noSign);
        if (fracBits == FP16_FRAC_BITS) {
            outData = std::get<0>(bitsInfo);
            outData = (sign << FP16_SIGN_SHIFT) | outData;
        } else if (fracBits == BF16_FRAC_BITS) {
            outData = std::get<1>(bitsInfo);
            outData = (sign << BF16_SIGN_SHIFT) | outData;
        } else if (fracBits == FP32_FRAC_BITS) {
            outData = std::get<2>(bitsInfo);
            outData = (sign << FP32_SIGN_SHIFT) | outData;
        }
        return outData;
    }

    template <typename T>
    struct DataCastToFloat32Functor<util::CPUDevice, T> {
        void operator()(const T* in, float* out, int length) const
        {
            for (auto i = 0; i < length; i++) {
                out[i] = Fp16ToFp32(in[i]);
            }
        }
    };

    template <typename T>
    struct DataCastToFloat16Functor<util::CPUDevice, T> {
        void operator()(const T* in, uint16_t* out, int length) const
        {
            for (auto i = 0; i < length; i++) {
                out[i] = Fp32ToFp16(in[i]);
            }
        }
    };

    template <typename T>
    struct DataCastToHiF8Fp8Functor<util::CPUDevice, T> {
        void operator()(const T* in, uint8_t* out, int length, int srcType, int dstType, int roundMode) const
        {
            auto expManBits = GetExpFracBits(srcType);
            std::function<uint8_t(T, uint32_t, uint32_t, int)> func = (dstType == FP8) ? FpNToFp8<T> : FpNToHiF8<T>;
#pragma omp parallel for
            for (auto i = 0; i < length; i++) {
                out[i] = func(in[i], expManBits.first, expManBits.second, roundMode);
            }
        }
    };

    template <typename T>
    struct HiF8Fp8CastToFpNFunctor<util::CPUDevice, T> {
        void operator()(const uint8_t* in, T* out, int length, int dstType, int srcType) const
        {
            auto expManBits = GetExpFracBits(dstType);
            std::function<T(uint8_t, uint32_t, uint32_t)> func = (srcType == FP8) ? Fp8ToFpN<T> : HiF8ToFpN<T>;
#pragma omp parallel for
            for (auto i = 0; i < length; i++) {
                out[i] = func(in[i], expManBits.first, expManBits.second);
            }
        }
    };

    template <typename T>
    struct FloatCastToFP4E2M1Functor<util::CPUDevice, T> {
        void operator()(const T* in, uint8_t* out, int length, int srcType) const
        {
            auto expManBits = GetExpFracBits(srcType);
#pragma omp parallel for
            for (int i = 0; i < length; i++) {
                out[i] = FloatToFP4E2M1(in[i], expManBits.first, expManBits.second);
            }
        }
    };

    template <typename T>
    struct FP4E2M1CastToFloatFunctor<util::CPUDevice, T> {
        void operator()(const uint8_t* in, T* out, int length, int dstType) const
        {
            auto expManBits = GetExpFracBits(dstType);
#pragma omp parallel for
            for (int i = 0; i < length; i++) {
                out[i] = FP4E2M1ToFloat<T>(in[i], expManBits.second);
            }
        }
    };

    template <typename T>
    struct FloatCastToFP4E1M2Functor<util::CPUDevice, T> {
        void operator()(const T* in, uint8_t* out, int length, int srcType) const
        {
            auto expManBits = GetExpFracBits(srcType);
#pragma omp parallel for
            for (int i = 0; i < length; i++) {
                out[i] = FloatToFP4E1M2(in[i], expManBits.first, expManBits.second);
            }
        }
    };

    template <typename T>
    struct FP4E1M2CastToFloatFunctor<util::CPUDevice, T> {
        void operator()(const uint8_t* in, T* out, int length, int dstType) const
        {
            auto expManBits = GetExpFracBits(dstType);
#pragma omp parallel for
            for (int i = 0; i < length; i++) {
                out[i] = FP4E1M2ToFloat<T>(in[i], expManBits.second);
            }
        }
    };

template struct DataCastToFloat32Functor<util::CPUDevice, uint16_t>;
template struct DataCastToFloat16Functor<util::CPUDevice, float>;
template struct DataCastToHiF8Fp8Functor<util::CPUDevice, uint32_t>;
template struct DataCastToHiF8Fp8Functor<util::CPUDevice, uint16_t>;
template struct HiF8Fp8CastToFpNFunctor<util::CPUDevice, uint32_t>;
template struct HiF8Fp8CastToFpNFunctor<util::CPUDevice, uint16_t>;
template struct FloatCastToFP4E2M1Functor<util::CPUDevice, uint16_t>;
template struct FP4E2M1CastToFloatFunctor<util::CPUDevice, uint16_t>;
template struct FloatCastToFP4E2M1Functor<util::CPUDevice, uint32_t>;
template struct FP4E2M1CastToFloatFunctor<util::CPUDevice, uint32_t>;
template struct FloatCastToFP4E1M2Functor<util::CPUDevice, uint16_t>;
template struct FP4E1M2CastToFloatFunctor<util::CPUDevice, uint16_t>;
template struct FloatCastToFP4E1M2Functor<util::CPUDevice, uint32_t>;
template struct FP4E1M2CastToFloatFunctor<util::CPUDevice, uint32_t>;


} // cast_uitl
