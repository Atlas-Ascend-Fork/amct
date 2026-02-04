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
 * @brief search shift bit core algorithm cpp implementation
 *
 * @version 1.0
 */

#include "search_n.h"
#include <cmath>
#include <numeric>

using namespace util;

namespace {
std::vector<int> GetShiftFactor()
{
    std::vector<int> shiftFactor;
    shiftFactor.push_back(1);
    for (unsigned int shiftBit = MIN_SHIFT_BIT; shiftBit <= MAX_SHIFT_BIT; shiftBit++) {
        shiftFactor.push_back(static_cast<int>(pow(BINARY_BASE, shiftBit)));
    }
    return shiftFactor;
}

void SearchShiftBitsInternal(const std::vector<std::vector<int>>& s32Data, std::vector<int>& bestN)
{
    std::vector<int> shiftFactor = GetShiftFactor();

    const size_t channelNum = s32Data.size();
    std::vector<int> tmpBestN(channelNum);

#pragma omp parallel for
    for (size_t i = 0; i < channelNum; i++) {
        std::vector<double> error(MAX_SHIFT_BIT + 1, 0);
#pragma omp parallel for
        for (unsigned int shiftBit = 1; shiftBit <= MAX_SHIFT_BIT; shiftBit++) {
            std::vector<double> errorSingle(s32Data[i].size());
            AmctCommon::EvaluateShiftNErrorCpu(static_cast<int>(s32Data[i].size()), &s32Data[i][0],
                errorSingle.data(), shiftFactor[shiftBit]);
            error[shiftBit] = std::accumulate(errorSingle.begin(), errorSingle.end(), static_cast<double>(0));
        }
        size_t minErrorPosition = 1;
        for (size_t k = 1; k < error.size(); k++) {
            if (error[k] < error[minErrorPosition]) {
                minErrorPosition = k;
            }
        }
        tmpBestN[i] = static_cast<int>(minErrorPosition);
    }
    (void)bestN.insert(bestN.cend(), tmpBestN.cbegin(), tmpBestN.cend());
    return;
}
}

namespace AmctCommon {
template <typename T>
void Clip(T& clipValue, int clipMin, int clipMax)
{
    if (clipValue < clipMin) {
        clipValue = clipMin;
    } else if (clipValue > clipMax) {
        clipValue = clipMax;
    }
    return;
}

template <typename T>
void EvaluateShiftNErrorCpu(const int dataLength, const T* inputData, double* error, const int shift)
{
#pragma omp parallel for
    for (int index = 0; index < dataLength; index++) {
        T outputS16 = static_cast<T>(floor(static_cast<double>(inputData[index] / shift)));
        Clip(outputS16, INT16_MIN, INT16_MAX);

        outputS16 = outputS16 * shift;
        error[index] = pow((outputS16 - inputData[index]), BASE);
    }
}

void SearchShiftBits(const std::vector<std::vector<int>>& s32Data, std::vector<int>& bestN)
{
    return SearchShiftBitsInternal(s32Data, bestN);
}

template void EvaluateShiftNErrorCpu(const int, const float*, double*, const int);
template void EvaluateShiftNErrorCpu(const int, const int*, double*, const int);
template void Clip(int&, int, int);
template void Clip(float&, int, int);
}
