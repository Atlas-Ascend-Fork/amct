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
 * @brief ULQ header file
 *
 * @file ulq.h in common_cpp
 *
 * @version 1.0
 */

#ifndef ULQ_H
#define ULQ_H

#include "util.h"

namespace AmctCommon {
/**
 * @ingroup quantize lib
 * @brief: Switch the first and second axis
 * @param: [in|out] data: flatten input data
 * @param: [in] length: the length of data
 * @param: [in|out] shape: the shape of data
 */
template <class T>
void TransposeAB(T* data, int length, std::vector<int>& shape);

/**
 * @ingroup quantize lib
 * @brief: Check whether the clip value is reasonable
 * @param: [in|out] clipMax: the new clip max value
 * @param: [in|out] clipMin: the new clip min value
 * @param: [in|out] clipMaxPre: last clip max value
 * @param: [in|out] clipMinPre: last clip min value
 */
template <class T>
bool ClipCheck(T& clipMax, T& clipMin, T& clipMaxPre, T& clipMinPre);

/**
 * @ingroup quantize lib
 * @brief: Universal Linear Quantization on Activations
 * @param [in|out] data: input data
 * @param [in] length: input data length
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 */
template <class T>
void Ulq(const T* inData, T* outData, int length,
    const util::FloatData &scale, const util::IntData &offset, const int numBits = 8);

/**
 * @ingroup quantize lib
 * @brief: ULQ gradient calculation
 * @param: [in] bottomData: bottom data
 * @param: [in] topDiff: top diff
 * @param: [in] length: the length of data
 * @param: [in] clip: a vector of clip_max, clip_min, clip_max_ori, clip_min_ori
 * @return: vector<double> a vector of clip_max_diff and clip_min_diff
 */
template <class T>
std::vector<T> UlqDiff(const T* bottomData, const T* topDiff, const int length, std::vector<T> clip);

int ActQuantForwardGpu(const int count, const double* bottomData, double* topData,
    double* clipMaxGpu, double* clipMinGpu, bool fixedMin, int quantBitNum);

int ActQuantForwardGpu(const int count, const float* bottomData, float* topData,
    float* clipMaxGpu, float* clipMinGpu, bool fixedMin, int quantBitNum);

int UlqDiffGpu(const int count, const float* bottomData, float* bottomDiff, const float* topDiff,
    const float* clipMaxGpu, const float* clipMinGpu, float& diffMaxCpuRef, float& diffMinCpuRef, int quantBitNum);

int UlqDiffGpu(const int count, const double* bottomData, double* bottomDiff, const double* topDiff,
    const double* clipMaxGpu, const double* clipMinGpu, double& diffMaxCpuRef, double& diffMinCpuRef, int quantBitNum);
}
#endif /* ARQ_H */
