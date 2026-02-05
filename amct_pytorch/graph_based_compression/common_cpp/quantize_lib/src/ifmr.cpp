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
 * @brief IFMR algorithm C++ implementation for CPU.
 *
 * @file ifmr.cpp
 *
 * @version 1.0
 */

#include "ifmr.h"
#include <cfloat>
#include <cmath>
#include <numeric>

using namespace util;

namespace AmctCommon {
template <class T>
static Status CheckIfmrQuantParams(const T* data, const FloatData &scale, const IntData &offset)
{
    if (data == nullptr) {
        LOG_ERROR("Empty pointer, all input tensor is empty\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (scale.length != 1 || offset.length != 1) {
        LOG_ERROR("scale.length = %u and offset.length = %u should be 1.\n", scale.length, offset.length);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    return AmctCommon::SUCCESS;
}

template <class T>
static Status IfmrQuantWithOffset(const T* data, unsigned int length, const IfmrParam &ifmrParam,
    const MaxMinValue<T>& maxminValueRef, int& bestMaxIndex)
{
    const T maxValue = maxminValueRef.maxValue;
    const T minValue = maxminValueRef.minValue;
    const float startRatio = ifmrParam.startRatio;
    const float endRatio = ifmrParam.endRatio;
    const float step = ifmrParam.step;

    const unsigned int baseNum = 2;
    const unsigned int numBits = ifmrParam.numBits;
    auto maxLimit = static_cast<unsigned int>(pow(baseNum, numBits) - 1);
    if (maxLimit == 0) {
        LOG_ERROR("[%s]maxLimit should not be zero.\n", __func__);
        return AmctCommon::GENERIC_ERROR;
    }

    auto maxSize = static_cast<unsigned int>((endRatio - startRatio) / step + 1);
    std::vector<float> noises(maxSize);
#pragma omp parallel for
    for (unsigned int i = 0; i < maxSize; i++) {
        T maxCandidates = (startRatio + i * step) * maxValue;
        T currentScale = (maxCandidates - minValue) / maxLimit;
        if (currentScale < 0) {
            continue;
        }
        int ret = util::ProcessScale(currentScale);
        if (ret != 0) {
            continue;
        }
        // calculate noise
        std::vector<float> currentNoise(length);
#pragma omp parallel for
        for (unsigned int j = 0; j < length; j++) {
            T valueClip = data[j];
            if (data[j] < minValue) {
                valueClip = minValue;
            } else if (data[j] > maxCandidates) {
                valueClip = maxCandidates;
            }
            valueClip = static_cast<T>(round(valueClip / currentScale) * currentScale);
            currentNoise[j] = static_cast<float>((valueClip - data[j]) * (valueClip - data[j]) / length);
        }
        noises[i] = std::accumulate(currentNoise.begin(), currentNoise.end(), static_cast<float>(0));
    }
    bestMaxIndex = std::distance(std::begin(noises), std::min_element(std::begin(noises), std::end(noises)));
    return AmctCommon::SUCCESS;
}

template <class T>
static Status CalScaleWithOffset(const MaxMinValue<T> &maxminValueRef, const IfmrParam &ifmrParam,
    const FloatData &scale, const IntData &offset, int bestMaxIndex)
{
    T maxValue = maxminValueRef.maxValue;
    T minValue = maxminValueRef.minValue;
    const float startRatio = ifmrParam.startRatio;
    const float step = ifmrParam.step;
    const unsigned int numBits = ifmrParam.numBits;
    const unsigned int baseNum = 2;

    const unsigned int maxLimit = static_cast<unsigned int>(pow(baseNum, numBits) - 1);
    if (maxLimit == 0) {
        LOG_ERROR("[%s]maxLimit should not be zero.\n", __func__);
        return AmctCommon::NOT_SUPPORT_ERROR;
    }

    float bestMaxValue = static_cast<float>((startRatio + bestMaxIndex * step) * maxValue);
    scale.data[0] = static_cast<float>((bestMaxValue - minValue) / maxLimit);
    Status ret = util::ProcessScale(scale.data[0]);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("[%s]scale failed.\n", __func__);
        return ret;
    }
    offset.data[0] = -static_cast<int>(round(minValue / scale.data[0])) - static_cast<int>(pow(baseNum, numBits - 1));

    return AmctCommon::SUCCESS;
}

template <class T>
static Status CalScaleWithoutOffset(const MaxMinValue<T> &maxminValueRef, const IfmrParam &ifmrParam,
    const FloatData &scale, const IntData &offset, int bestMaxIndex)
{
    const float startRatio = ifmrParam.startRatio;
    const float step = ifmrParam.step;
    T maxValue = maxminValueRef.maxValue;
    T minValue = maxminValueRef.minValue;

    const unsigned int baseNum = 2;
    const unsigned int numBits = ifmrParam.numBits;
    const unsigned int min = static_cast<unsigned int>(pow(baseNum, numBits - 1));
    const unsigned int max = static_cast<unsigned int>(pow(baseNum, numBits - 1) - 1);
    if (min == 0 || max == 0) {
        LOG_ERROR("[%s]min or max should not be zero.\n", __func__);
        return AmctCommon::NOT_SUPPORT_ERROR;
    }

    float absMaxValue = (fabs(maxValue) > fabs(minValue)) ? static_cast<float>(maxValue) : static_cast<float>(minValue);
    float bestMaxValue = static_cast<float>((startRatio + bestMaxIndex * step) * fabs(absMaxValue));
    scale.data[0] = (absMaxValue > 0) ? bestMaxValue / max : bestMaxValue / min;
    Status ret = util::ProcessScale(scale.data[0]);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("[%s]scale failed.\n", __func__);
        return ret;
    }
    offset.data[0] = 0;

    return AmctCommon::SUCCESS;
}

template <class T>
static Status IfmrQuantWithoutOffset(const T* data, unsigned int length, const IfmrParam &ifmrParam,
    const MaxMinValue<T>& maxminValueRef, int& bestMaxIndex)
{
    const float startRatio = ifmrParam.startRatio;
    const float endRatio = ifmrParam.endRatio;
    const float step = ifmrParam.step;
    const T minValue = maxminValueRef.minValue;
    const T maxValue = maxminValueRef.maxValue;
    auto maxSize = static_cast<unsigned int>((endRatio - startRatio) / step + 1);

    const unsigned int baseNum = 2;
    const unsigned int numBits = ifmrParam.numBits;
    float absMaxValue = (fabs(maxValue) > fabs(minValue)) ? static_cast<float>(maxValue) : static_cast<float>(minValue);
    auto maxLimit = (absMaxValue > 0) ? static_cast<unsigned int>(pow(baseNum, numBits - 1) - 1) : \
        static_cast<unsigned int>(pow(baseNum, numBits - 1));
    if (maxLimit == 0) {
        LOG_ERROR("[%s]maxLimit should not be zero.\n", __func__);
        return AmctCommon::GENERIC_ERROR;
    }

    absMaxValue = static_cast<float>(fabs(absMaxValue));
    std::vector<float> noises(maxSize);
#pragma omp parallel for
    for (unsigned int i = 0; i < maxSize; i++) {
        T maxCandidates = (startRatio + i * step) * absMaxValue;
        T currentScale = maxCandidates / maxLimit;
        int ret = util::ProcessScale(currentScale);
        if (ret != 0) {
            LOG_ERROR("[%s]scale is illegal.\n", __func__);
            continue;
        }
        // calculate noise
        std::vector<float> currentNoise(length);
#pragma omp parallel for
        for (unsigned int j = 0; j < length; j++) {
            T clipValue = data[j];
            if (data[j] < -maxCandidates) {
                clipValue = -maxCandidates;
            } else if (data[j] > maxCandidates) {
                clipValue = maxCandidates;
            }
            clipValue = static_cast<T>(round(clipValue / currentScale) * currentScale);
            currentNoise[j] = static_cast<float>((clipValue - data[j]) * (clipValue - data[j]) / length);
        }
        noises[i] = std::accumulate(currentNoise.begin(), currentNoise.end(), static_cast<float>(0));
    }
    bestMaxIndex = std::distance(std::begin(noises), std::min_element(std::begin(noises), std::end(noises)));
    return AmctCommon::SUCCESS;
}

template <typename T>
bool Compare(const T a, const T b) {
    if (std::isnan(a)) {
        return true;
    } else if (std::isnan(b)) {
        return false;
    } else {
        return a > b;
    }
}

template <class T>
Status FindMaxAndMinValueWithPercentile(const T* data, const unsigned int length, MaxMinValue<T>& maxminValueRef,
    const float maxPercentile, const float minPercentile)
{
    const unsigned int maxSize = (fabs(maxPercentile - 1) < DBL_EPSILON) ?
        1 : length - static_cast<unsigned int>(floor(maxPercentile * length));
    const unsigned int minSize = (fabs(minPercentile - 1) < DBL_EPSILON) ?
        1 : length - static_cast<unsigned int>(floor(minPercentile * length));

    std::vector<T> inputMax(data, data + length);
    std::vector<T> inputMin(data, data + length);

    std::partial_sort(inputMax.begin(), inputMax.begin() + maxSize, inputMax.end(), Compare<T>);
    std::partial_sort(inputMin.begin(), inputMin.begin() + minSize, inputMin.end());

    if ((maxSize - 1) > length || (minSize - 1) > length) {
        LOG_ERROR("maxSize > length may rise array bound error!\n");
        return AmctCommon::NOT_SUPPORT_ERROR;
    }

    maxminValueRef.maxValue = std::max(static_cast<T>(0), inputMax[maxSize - 1]);
    maxminValueRef.minValue = std::min(static_cast<T>(0), inputMin[minSize - 1]);

    if (std::isinf(inputMax.front()) || std::isinf(inputMin.front())) {
        LOG_ERROR("IFMR calibration data has inf value.\n");
        return AmctCommon::NOT_SUPPORT_ERROR;
    }
    if (std::isnan(inputMax.front())) {
        LOG_ERROR("IFMR calibration data has nan value.\n");
        return AmctCommon::NOT_SUPPORT_ERROR;
    }

    return AmctCommon::SUCCESS;
}

template <class T>
static Status IfmrQuantCalibration(const T* data, unsigned int length, const IfmrParam &ifmrParam,
    const FloatData &scale, const IntData &offset)
{
    MaxMinValue<T> maxminValue = {0, 0};
    MaxMinValue<T>& maxminValueRef = maxminValue;
    Status ret = FindMaxAndMinValueWithPercentile(data, length, maxminValueRef,
        ifmrParam.maxPercentile, ifmrParam.minPercentile);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("FindMaxAndMinValueWithPercentile failed.\n");
        return ret;
    }

    int bestMaxIndex = 0;
    if (ifmrParam.withOffset) {
        ret = IfmrQuantWithOffset(data, length, ifmrParam, maxminValueRef, bestMaxIndex);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("IfmrQuantWithOffset failed.\n");
            return ret;
        }
        ret = CalScaleWithOffset(maxminValueRef, ifmrParam, scale, offset, bestMaxIndex);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("CalScaleWithOffset failed.\n");
            return ret;
        }
    } else {
        ret = IfmrQuantWithoutOffset(data, length, ifmrParam, maxminValueRef, bestMaxIndex);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("IfmrQuantWithoutOffset failed.\n");
            return ret;
        }
        ret = CalScaleWithoutOffset(maxminValueRef, ifmrParam, scale, offset, bestMaxIndex);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("CalScaleWithoutOffset failed.\n");
            return ret;
        }
    }

    return AmctCommon::SUCCESS;
}

template <class T>
Status ClipAndQuant(T* data, const unsigned int length, const int numBits,
    const float scaleValue, const int offsetValue)
{
    const int baseNum = 2;
    const int minLimit = -static_cast<int>(pow(baseNum, numBits - 1));
    const int maxLimit = static_cast<int>(pow(baseNum, numBits - 1) - 1);

    if (scaleValue < DBL_EPSILON) {
        LOG_ERROR("scale value should not be less than zero!\n");
        return AmctCommon::NOT_SUPPORT_ERROR;
    }

    for (unsigned int i = 0; i < length; ++i) {
        int dataTemp = static_cast<int>(round(data[i] / scaleValue)) + offsetValue;
        dataTemp = dataTemp < minLimit ? minLimit : dataTemp;
        dataTemp = dataTemp > maxLimit ? maxLimit : dataTemp;
        data[i] = (dataTemp - offsetValue) * scaleValue;
    }

    return AmctCommon::SUCCESS;
}

template <class T>
static Status IfmrQuantInference(T* data, unsigned int length, const IfmrParam &ifmrParam,
    const FloatData &scale, const IntData &offset)
{
    float scaleValue = scale.data[0];
    int offsetValue = offset.data[0];

    Status ret = ClipAndQuant(data, length, static_cast<int>(ifmrParam.numBits), scaleValue, offsetValue);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("ClipAndQuant failed.\n");
        return ret;
    }

    return AmctCommon::SUCCESS;
}

template <class T>
static Status IfmrQuantInternel(T* data, unsigned int length,
    const IfmrParam &ifmrParam, const FloatData &scale, const IntData &offset)
{
    Status ret = CheckIfmrQuantParams(data, scale, offset);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("CheckIfmrQuantParams failed.");
        return ret;
    }

    if (ifmrParam.calibration > 1) {
        LOG_ERROR("ifmrParam.calibration = %u.\n", ifmrParam.calibration);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (ifmrParam.calibration == 0) {
        return IfmrQuantCalibration(data, length, ifmrParam, scale, offset);
    } else {
        return IfmrQuantInference(data, length, ifmrParam, scale, offset);
    }
}

int IfmrQuant(float* data, unsigned int length,
    const IfmrParam &ifmrParam, const FloatData &scale, const IntData &offset)
{
    return IfmrQuantInternel(data, length, ifmrParam, scale, offset);
}

int IfmrQuant(double* data, unsigned int length,
    const IfmrParam &ifmrParam, const FloatData &scale, const IntData &offset)
{
    return IfmrQuantInternel(data, length, ifmrParam, scale, offset);
}
}