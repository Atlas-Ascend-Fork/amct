/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
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
 * @brief ifmr header file
 *
 * @file ifmr.h in common_cpp
 *
 * @version 1.0
 */

#ifndef IFMR_H
#define IFMR_H

#include "util.h"

namespace AmctCommon {
/**
* @ingroup quantize lib
* @brief: Params for Ifmr Quantiation.
*/
struct IfmrParam {
    unsigned int calibration;
    unsigned int numBits;
    bool withOffset;
    bool needDump;
    float startRatio;
    float endRatio;
    float step;
    float maxPercentile;
    float minPercentile;
};

// error calculation function
template <class T>
void CalcErrorGpu(T* data, T* clipMin, T* clipMax, T* error,
    bool withOffset, util::ClipInfo clipInfo, const unsigned int inputSize);

/**
* @ingroup quantize lib
* @brief: Params for Ifmr Quantiation
*/
template <class T>
struct MaxMinValue {
    T maxValue;
    T minValue;
};

/**
  * @ingroup quantize lib
  * @brief: Ifmr Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] ifmrParam: ifmr quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int IfmrQuant(float* data, unsigned int length, const AmctCommon::IfmrParam &ifmrParam,
    const util::FloatData &scale, const util::IntData &offset);

/**
  * @ingroup quantize lib
  * @brief: Ifmr Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] ifmrParam: ifmr quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int IfmrQuant(double* data, unsigned int length, const AmctCommon::IfmrParam &ifmrParam,
    const util::FloatData &scale, const util::IntData &offset);

int IfmrQuantGpu(float* deviceData, float* hostDatadata, unsigned int length, AmctCommon::IfmrParam ifmrParam,
    util::FloatData scale, util::IntData offset);

/**
  * @ingroup quantize lib
  * @brief: Ifmr Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] ifmrParam: ifmr quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int IfmrQuantGpu(double* deviceData, double* hostDatadata, unsigned int length, AmctCommon::IfmrParam ifmrParam,
    util::FloatData scale, util::IntData offset);
}

#endif /* IFMR_H */
