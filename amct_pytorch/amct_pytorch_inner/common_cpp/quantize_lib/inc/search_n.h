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
