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
 * @brief dmq_balance algorithm C++ implementation
 *
 * @file dmq_balance.cpp
 *
 * @version 1.0
 */

#include "dmq_balance.h"

#include <cmath>
#include <cfloat>
#include <algorithm>

using namespace util;

namespace {
constexpr float MIGRATION_STRENGTH_MIN = 0.2f;
constexpr float MIGRATION_STRENGTH_MAX = 0.8f;

void ProcessZeroChannel(std::vector<float> &actMaxValues, std::vector<float> &wtsMaxValues, float *balanceFactor)
{
    float actOutlier = 0;
    float wtsOutlier = 0;
    for (size_t channelIdx = 0; channelIdx < actMaxValues.size(); ++channelIdx) {
        if (balanceFactor[channelIdx] < FLT_EPSILON) {
            continue;
        }
        actOutlier = actOutlier > (actMaxValues[channelIdx] / balanceFactor[channelIdx]) ?
            actOutlier : (actMaxValues[channelIdx] / balanceFactor[channelIdx]);
        wtsOutlier = wtsOutlier > (wtsMaxValues[channelIdx] * balanceFactor[channelIdx]) ?
            wtsOutlier : (wtsMaxValues[channelIdx] * balanceFactor[channelIdx]);
    }

    if ((actOutlier < FLT_EPSILON) || (wtsOutlier < FLT_EPSILON)) {
        for (size_t channelIdx = 0; channelIdx < actMaxValues.size(); ++channelIdx) {
            balanceFactor[channelIdx] = 1;
        }
        return;
    }

    for (size_t channelIdx = 0; channelIdx < actMaxValues.size(); ++channelIdx) {
        if (actMaxValues[channelIdx] < FLT_EPSILON) {
            if (wtsMaxValues[channelIdx] < FLT_EPSILON) {
                balanceFactor[channelIdx] = 1;
            } else {
                balanceFactor[channelIdx] = wtsOutlier / wtsMaxValues[channelIdx];
            }
        } else {
            if (wtsMaxValues[channelIdx] < FLT_EPSILON) {
                balanceFactor[channelIdx] = actMaxValues[channelIdx] / actOutlier;
            }
        }
    }
}
}

namespace AmctCommon {
Status CheckDMQBParam(const FloatData &act, const FloatData &wts, float migrationStrength,
    uint32_t channelNum, const float *balanceFactor)
{
    if ((act.data == nullptr) || (act.length == 0)) {
        LOG_ERROR("act pointer is null or length is 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if ((wts.data == nullptr) || (wts.length == 0)) {
        LOG_ERROR("wts pointer is null or length is 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (migrationStrength < MIGRATION_STRENGTH_MIN || migrationStrength > MIGRATION_STRENGTH_MAX) {
        LOG_ERROR("migrationStrength: %f not support, should be in [0.2, 0.8]!\n", migrationStrength);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (channelNum == 0) {
        LOG_ERROR("channelNum is 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if ((act.length % channelNum != 0) || (wts.length % channelNum != 0)) {
        LOG_ERROR("act.length: %u or wts.length: %u cannot be divisible by channelNum: %u!\n",
            act.length, wts.length, channelNum);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (balanceFactor == nullptr) {
        LOG_ERROR("balanceFactor is null!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    return AmctCommon::SUCCESS;
}

static Status FindMaxValue(const FloatData &data, uint32_t channelNum, uint32_t channelIdx, float &maxValue)
{
    if (channelNum == 0U) {
        LOG_ERROR("channelNum[%u] max value is illegal!\n", channelNum);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    uint32_t dataNum = data.length / channelNum;
    for (uint32_t dataIdx = channelIdx * dataNum; dataIdx < (channelIdx + 1) * dataNum; dataIdx++) {
        if (fabs(data.data[dataIdx]) > maxValue) {
            maxValue = fabs(data.data[dataIdx]);
        }
    }

    if (std::isnan(maxValue) || std::isinf(maxValue)) {
        LOG_ERROR("channel[%u] max value is illegal!\n", channelIdx);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    return AmctCommon::SUCCESS;
}

Status DMQBalance(const FloatData &act, const FloatData &wts, float migrationStrength,
    uint32_t channelNum, float *balanceFactor)
{
    Status ret = CheckDMQBParam(act, wts, migrationStrength, channelNum, balanceFactor);
    if (ret != AmctCommon::SUCCESS) {
        return ret;
    }

    std::vector<float> actMaxValues(channelNum, 0);
    std::vector<float> wtsMaxValues(channelNum, 0);
    for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx++) {
        ret = FindMaxValue(act, channelNum, channelIdx, actMaxValues[channelIdx]);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("find act max value fail.\n");
            return ret;
        }

        ret = FindMaxValue(wts, channelNum, channelIdx, wtsMaxValues[channelIdx]);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("find wts max value fail.\n");
            return ret;
        }
    }

    float actMax = *max_element(actMaxValues.begin(), actMaxValues.end());
    if (actMax < FLT_EPSILON) {
        for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx++) {
            balanceFactor[channelIdx] = 1;
        }
        return AmctCommon::SUCCESS;
    }

    float wtsMax = *max_element(wtsMaxValues.begin(), wtsMaxValues.end());
    if (wtsMax < FLT_EPSILON) {
        for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx++) {
            balanceFactor[channelIdx] = 1;
        }
        return AmctCommon::SUCCESS;
    }

    bool hasZeroChannel = false;
    for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx++) {
        if ((actMaxValues[channelIdx] < FLT_EPSILON) || (wtsMaxValues[channelIdx] < FLT_EPSILON)) {
            hasZeroChannel = true;
            balanceFactor[channelIdx] = 0;
            continue;
        }
        balanceFactor[channelIdx] = pow(actMaxValues[channelIdx], migrationStrength) /
            pow(wtsMaxValues[channelIdx], 1 - migrationStrength);
    }

    if (hasZeroChannel) {
        ProcessZeroChannel(actMaxValues, wtsMaxValues, balanceFactor);
    }

    return AmctCommon::SUCCESS;
}
}
