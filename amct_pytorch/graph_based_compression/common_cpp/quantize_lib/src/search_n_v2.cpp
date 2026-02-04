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
 * @brief search n v2 shift bit core algorithm cpp implementation
 *
 * @version 1.0
 */

#include "search_n_v2.h"
#include <cmath>
#include <vector>

using namespace util;

namespace AmctCommon {
template <typename T>
int EvaluateErrorCpu(const int n, const T* s32Data, T* shiftError, const int shift)
{
    const int shiftBound = static_cast<int>(pow(BASE, SHIFT_POW));
    if (shift == 0) {
        return AmctCommon::ZERO_DIVISION_ERROR;
    }
    for (int index = 0; index < n; index++) {
        T s16Data = static_cast<T>(floor(s32Data[index] / shift));
        if (s16Data < -shiftBound) {
            s16Data = -shiftBound;
        } else if (s16Data > (shiftBound - 1)) {
            s16Data = shiftBound - 1;
        }
        s16Data = s16Data * shift;
        shiftError[index] = static_cast<T>(pow((s16Data - s32Data[index]), BASE));
    }
    return AmctCommon::SUCCESS;
}


template <typename T>
int SearchNV2FindBestNCpu(std::vector<std::vector<T>>& storeError, IntData& bestN, bool isBroadcast)
{
    for (unsigned int i = 0; i < bestN.length; i++) {
        if (isBroadcast) {
            auto smallest = std::min_element(std::begin(storeError[0]), std::end(storeError[0]));
            // valid value start form 1 to 16 [1,16];
            bestN.data[i] = std::distance(std::begin(storeError[0]), smallest) + 1;
        } else {
            auto smallest = std::min_element(std::begin(storeError[i]), std::end(storeError[i]));
            // valid value start form 1 to 16 [1,16];
            bestN.data[i] = std::distance(std::begin(storeError[i]), smallest) + 1;
        }
    }
    return AmctCommon::SUCCESS;
}


template <typename T>
int SearchNV2AccumulateError(std::vector<std::vector<T>>& currentData, std::vector<std::vector<T>>& searchNError,
    const FloatData &deqScaleCpu, bool isBroadcast)
{
    // data size in one channel;
    if ((currentData.size() == 0) || (currentData[0].size() == 0)) {
        return AmctCommon::CONTAINER_EMPTY_ERROR;
    }
    size_t searchNDataChannelSize = currentData[0].size();
    size_t searchNDataChannelNum = currentData.size();
    size_t searchNDataSize = searchNDataChannelNum * searchNDataChannelSize;

    std::vector<T> s32Data(searchNDataSize);
    std::vector<T> quantError(searchNDataSize);

    // dequantize process is channel wise when isBroadcast
    for (size_t index = 0; index < searchNDataSize; index++) {
        size_t channelIndex = static_cast<size_t>(index / searchNDataChannelSize);
        size_t channelRemainder = static_cast<size_t>(index % searchNDataChannelSize);
        s32Data[index] = static_cast<T>(
            round((currentData[channelIndex][channelRemainder]) / deqScaleCpu.data[channelIndex]));
    }
    for (int shiftNCandidate = 1; shiftNCandidate <= SHIFT_BITS; shiftNCandidate++) {
        auto status = EvaluateErrorCpu<T>(
            static_cast<int>(searchNDataSize), s32Data.data(), quantError.data(),
            static_cast<int>(pow(BASE, shiftNCandidate)));
        if (status != AmctCommon::SUCCESS) {
            return status;
        }
        if (isBroadcast) {
            T error = 0;
            for (size_t j = 0; j < searchNDataSize; j++) {
                error += quantError[j];
            }
            searchNError[0][static_cast<unsigned int>(shiftNCandidate - 1)] += error;
        } else {
            for (size_t i = 0; i < searchNError.size(); i++) {
                T error = 0;
                for (size_t j = i * searchNDataChannelSize; j < (i + 1) * searchNDataChannelSize; j++) {
                    error += quantError[j];
                }
                searchNError[i][shiftNCandidate - 1] += error;
            }
        }
    }
    return AmctCommon::SUCCESS;
}


template int SearchNV2FindBestNCpu(std::vector<std::vector<float>>& storeError,
    IntData& bestN, bool isBroadcast);

template int SearchNV2AccumulateError(std::vector<std::vector<float>>& currentData,
    std::vector<std::vector<float>>& searchNError, const FloatData &deqScaleCpu, bool isBroadcast);

template int SearchNV2FindBestNCpu(std::vector<std::vector<double>>& storeError,
    IntData& bestN, bool isBroadcast);

template int SearchNV2AccumulateError(std::vector<std::vector<double>>& currentData,
    std::vector<std::vector<double>>& searchNError, const FloatData &deqScaleCpu, bool isBroadcast);
}
