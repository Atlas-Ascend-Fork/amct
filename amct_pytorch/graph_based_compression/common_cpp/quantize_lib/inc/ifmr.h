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
