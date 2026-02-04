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
 * @brief search_n header file
 *
 * @file search_n.h in common_cpp
 *
 * @version 1.0
 */

#ifndef SEARCH_N_H
#define SEARCH_N_H

#include "util.h"

namespace AmctCommon {
int SearchNGpu(const std::vector<std::vector<int>>& storedData,
    std::vector<int>& bestN);

void SearchShiftBits(const std::vector<std::vector<int>>& s32Data, std::vector<int>& bestN);

template <typename T>
void EvaluateShiftNErrorCpu(const int dataLength, const T* inputData, double* error, const int shift);

template<typename T>
int EvaluateSearchNErrorCudaHost(const int searchNDataSize, T* s32Data, double* error, const int shiftBits);

template<typename T>
int SearchNQuantForwardCudaHost(T* s32Data, float* deqScale,
    const int searchNDataSize, const int searchNDataChannelSize);

template <typename T>
void Clip(T& clipValue, int clipMin, int clipMax);
}

#endif /* SEARCH_N_H */
