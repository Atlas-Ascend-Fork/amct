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
 * @brief cpp implementation of NUQ algorithm on CPU.
 *
 * @file nuq.cpp
 *
 * @version 1.0
 */

#include "nuq.h"
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <numeric>
#include "arq.h"

using namespace util;

namespace {
template <class T>
Status CheckNuqQuantParams(const T* data, unsigned int length, const FloatData &scale, const IntData &offset)
{
    if (data == nullptr) {
        LOG_ERROR("Empty data pointer!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (length == 0) {
        LOG_ERROR("The length cannot be 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    for (unsigned int i = 0; i < length; i++) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            LOG_ERROR("The data contains invalid values!\n");
            return AmctCommon::BAD_PARAMETERS_ERROR;
        }
    }

    if (scale.length == 0) {
        LOG_ERROR("The scale.length should not be 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (length % scale.length != 0) {
        LOG_ERROR("The length of data should be an integer multiple of scale.length!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (scale.length != offset.length) {
        LOG_ERROR("The offset.length should be same as scale.length!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    return AmctCommon::SUCCESS;
}

Status NuqReal(double* data, unsigned int length, const FloatData &scale, const IntData &offset, char* int8Data)
{
    Status ret = CheckNuqQuantParams(data, length, scale, offset);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("Check NUQ parameters failed!\n");
        return ret;
    }

    unsigned int perChannelLength = length / scale.length;
    for (unsigned int channelIndex = 0; channelIndex < scale.length; channelIndex++) {
        unsigned int baseOffset = channelIndex * perChannelLength;
        double* offsetData = data + baseOffset;
        char* offsetChar = int8Data + baseOffset;
        for (unsigned int i = 0; i < perChannelLength; i++) {
            offsetChar[i] = static_cast<char>(
                round(offsetData[i] / static_cast<double>(scale.data[channelIndex]))
                + static_cast<double>(offset.data[channelIndex]));
        }
    }

    return AmctCommon::SUCCESS;
}

// calculate the normalized scale for K-Means++ clustering
std::vector<double> Weight(unsigned int length, const FloatData &scale, unsigned int perChannelLength)
{
    double sumWeight = 0;
    std::vector<double> weight(scale.data, scale.data + scale.length);
    double minWeight = *std::min_element(weight.begin(), weight.end());

    for (unsigned int channelIndex = 0; channelIndex < scale.length; channelIndex++) {
        weight[channelIndex] = pow(weight[channelIndex] / minWeight, EVEN);
        sumWeight += weight[channelIndex];
    }

    for (unsigned int channelIndex = 0; channelIndex < scale.length; channelIndex++) {
        weight[channelIndex] = weight[channelIndex] / sumWeight;
    }

    std::vector<double> expandWeight(length);

    for (unsigned int i = 0; i < length; i++) {
        expandWeight[i] = weight[i / perChannelLength];
    }

    return expandWeight;
}
}

namespace AmctCommon {
template <class T>
void ClipAndQuant(std::vector<T>& data, const FloatData &scale, const IntData &offset, unsigned int perChannelLength)
{
    int minLimit = -static_cast<int>(pow(BINARY, NUM_BITS_QUANT - 1));
    int maxLimit = static_cast<int>(pow(BINARY, NUM_BITS_QUANT - 1) - 1);

    for (unsigned int channelIndex = 0; channelIndex < scale.length; channelIndex++) {
        unsigned int channelHeadIdx = perChannelLength * channelIndex;
        for (unsigned int i = 0; i < perChannelLength; i++) {
            T tmpValue = data[channelHeadIdx + i] / scale.data[channelIndex] + offset.data[channelIndex];
            tmpValue = tmpValue < minLimit ? minLimit : tmpValue;
            tmpValue = tmpValue > maxLimit ? maxLimit : tmpValue;
            data[channelHeadIdx + i] = tmpValue;
        }
    }
}


template <class T>
T Median(std::vector<T> data)
{
    T median;

    if (data.size() % EVEN == 0) {
        std::nth_element(data.begin(), data.begin() + data.size() / EVEN - 1, data.end());
        T medianLeft = data[data.size() / EVEN - 1];
        std::nth_element(data.begin(), data.begin() + data.size() / EVEN, data.end());
        T medianRight = data[data.size() / EVEN];
        median = (medianLeft + medianRight) / EVEN;
    } else {
        std::nth_element(data.begin(), data.begin() + (data.size() - 1) / EVEN, data.end());
        median = data[(data.size() - 1) / EVEN];
    }

    return median;
}

template <class T>
void CalcDistance(std::vector<T>& data, std::vector<T>& distance, std::vector<T>& clusterCenter)
{
    for (size_t i = 0; i < data.size(); i++) {
        distance[i] = pow(data[i] - clusterCenter[0], EVEN);
    }
}

template <class T>
void KmeansInit(std::vector<T>& data, std::vector<T>& clusterCenter)
{
    int numCandidate = BINARY + static_cast<int>(log(clusterCenter.size()));
    clusterCenter[0] = Median(data);
    std::vector<T> distance(data.size(), 0);
    std::vector<T> cumulativeDistance(data.size());
    T equidistant;
    std::vector<T> candidateCenter(numCandidate);
    std::vector<T> canidateDistance(data.size());
    std::vector<T> canidateTotalDistance(numCandidate);

    CalcDistance(data, distance, clusterCenter);

    for (size_t snCenter = 1; snCenter < clusterCenter.size(); snCenter++) {
        // Calculate the cumulative distance distribution
        cumulativeDistance[0] = distance[0];
        for (size_t i = 1; i < data.size(); i++) {
            cumulativeDistance[i] = distance[i] + cumulativeDistance[i - 1];
        }
        // Five equally spaced distribution points as candidate points
        equidistant = (cumulativeDistance.back() / (numCandidate + 1.0f)) * 1.0f;
        for (int snCandidate = 0; snCandidate < numCandidate; snCandidate++) {
            for (size_t i = 0; i < data.size(); i++) {
                if (cumulativeDistance[i] > equidistant * (snCandidate + 1)) {
                    candidateCenter[snCandidate] = data[i];
                    break;
                }
            }
            // Calculate the minimum distance between the candidate point and selected centers to all points
            for (size_t i = 0; i < data.size(); i++) {
                canidateDistance[i] = (distance[i] < pow(data[i] - candidateCenter[snCandidate], EVEN)) ?
                    distance[i] : static_cast<T>(pow(data[i] - candidateCenter[snCandidate], EVEN));
            }
            canidateTotalDistance[snCandidate] = std::accumulate(canidateDistance.begin(), canidateDistance.end(), 0);
        }
        // The candidate cluster center with the minimum sum of distances is taken as the next cluster center
        clusterCenter[snCenter] = candidateCenter[std::distance(
            canidateTotalDistance.begin(), std::min_element(
                canidateTotalDistance.begin(), canidateTotalDistance.end()))];
        // Upadte distance, unless the last iteration
        if (snCenter + 1 == clusterCenter.size()) {
            break;
        } else {
            for (size_t i = 0; i < data.size(); i++) {
                distance[i] = (distance[i] < pow(data[i] - clusterCenter[snCenter], EVEN)) ?
                    distance[i] : pow(data[i] - clusterCenter[snCenter], EVEN);
            }
        }
    }
}


template <class T>
void Classify(const std::vector<T>& data, std::vector<unsigned int>& dataCluster, const std::vector<T>& clusterCenter)
{
    for (size_t i = 0; i < data.size(); i++) {
        T minValue = FLT_MAX;
        T nowValue;
        for (size_t snCluster = 0; snCluster < clusterCenter.size(); snCluster++) {
            nowValue = static_cast<T>(fabs(data[i] - clusterCenter[snCluster]));
            if (minValue > nowValue) {
                minValue = nowValue;
                dataCluster[i] = static_cast<unsigned int>(snCluster);
            }
        }
    }
}


// Kmeans++ Algorithm
template <class T>
void Kmeans(std::vector<T>& data, unsigned int numCluster, std::vector<double>& weight, std::vector<T>& clusterToStore)
{
    std::vector<T> clusterCenter(numCluster);

    // apply app_k-means++ init
    KmeansInit(data, clusterCenter);

    std::vector<unsigned int> dataCluster(data.size(), 0);
    std::vector<T> sumCluster(numCluster);
    std::vector<T> weightCluster(numCluster);
    std::vector<T> clusterCenterLast(numCluster, 0);
    T loss;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        loss = 0;
        Classify(data, dataCluster, clusterCenter);
        for (unsigned int snCluster = 0; snCluster < numCluster; snCluster++) {
            loss += static_cast<T>(fabs(clusterCenter[snCluster] - clusterCenterLast[snCluster]));
        }
        if (loss < EPS) {
            break;
        } else {
            clusterCenterLast.assign(clusterCenter.begin(), clusterCenter.end());
            weightCluster.assign(numCluster, 0);
            sumCluster.assign(numCluster, 0);
            for (size_t i = 0; i < data.size(); i++) {
                weightCluster[dataCluster[i]] += static_cast<T>(weight[i]);
                sumCluster[dataCluster[i]] += static_cast<T>(data[i] * weight[i]);
            }
            for (unsigned int snCluster = 0; snCluster < numCluster; snCluster++) {
                T weightClusterData = (weightCluster[snCluster] < FLT_EPSILON) ? 1 : weightCluster[snCluster];
                clusterCenter[snCluster] = sumCluster[snCluster] / weightClusterData;
            }
        }
    }

    for (unsigned int snCluster = 0; snCluster < numCluster; snCluster++) {
        clusterCenter[snCluster] = round(clusterCenter[snCluster]);
    }

    Classify(data, dataCluster, clusterCenter);

    for (size_t i = 0; i < data.size(); i++) {
        data[i] = clusterCenter[dataCluster[i]];
    }

    // store the cluster center
    for (size_t index = 0; index < clusterCenter.size(); index++) {
        clusterToStore[index] = clusterCenter[index];
    }
}

template <class T>
float OptimizeScale(const T* data, unsigned int baseOffset, unsigned int length,
    std::vector<T>& clusterData, int offset)
{
    T numerator = 0.0;
    T denominator = 0.0;
    const T* offsetData = data + baseOffset;

    for (unsigned int i = 0; i < length; i++) {
        numerator += offsetData[i] * (clusterData[baseOffset + i] + offset);
        denominator += static_cast<T>(pow(clusterData[baseOffset + i] + offset, EVEN));
    }

    float scale = static_cast<float>(numerator / denominator);
    if (std::isnan(scale)) {
        return 1.0;
    }

    return scale;
}


template <class T>
int OptimizeOffset(const T* data, unsigned int baseOffset, unsigned int length,
    std::vector<T>& clusterData, float scale)
{
    T numerator = 0.0;
    const T* offsetData = data + baseOffset;

    for (unsigned int i = 0; i < length; i++) {
        numerator += offsetData[i] / (scale + EPS) - clusterData[baseOffset + i];
    }

    int offset = static_cast<int>(round(numerator / length));

    return offset;
}


template <class T>
void Restore(T* data, std::vector<T> clusterData, const FloatData &scale, const IntData &offset,
    unsigned int perChannelLength)
{
    for (unsigned int channelIndex = 0; channelIndex < scale.length; channelIndex++) {
        unsigned int baseOffset = channelIndex * perChannelLength;
        T* offsetData = data + baseOffset;
        for (unsigned int i = 0; i < perChannelLength; i++) {
            offsetData[i] = (clusterData[baseOffset + i] + offset.data[channelIndex]) * scale.data[channelIndex];
        }
    }
}


template <class T>
Status NuqCalibrationCpuKernel(const T* data, std::vector<T>& clusterData, std::vector<T>& clusterToStore,
    unsigned int length, const NuqParam &nuqParam, const FloatData &scale, const IntData &offset)
{
    Status ret = CheckNuqQuantParams(data, length, scale, offset);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("Check NUQ parameters failed!\n");
        return ret;
    }

    const unsigned int perChannelLength = length / scale.length;

    AmctCommon::ArqParam arqParam =  {NUM_BITS_QUANT, true, nuqParam.withOffset};
    ret = AmctCommon::WtsArqCalibrationCpuKernel(data, length, arqParam, scale, offset, CO_CI_KH_KW);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("NuqCalibrationCpuKernel run ARQ init failed! \n");
        return ret;
    }

    ClipAndQuant(clusterData, scale, offset, perChannelLength);

    unsigned int baseOffset;
    std::vector<double> weight = Weight(length, scale, perChannelLength);
    Kmeans(clusterData, nuqParam.numSteps, weight, clusterToStore);

    for (unsigned int iter = 0; iter < nuqParam.numIter; iter++) {
        // Optimize the scale and offset
        for (unsigned int channelIndex = 0; channelIndex < scale.length; channelIndex++) {
            baseOffset = channelIndex * perChannelLength;
            scale.data[channelIndex] =
                OptimizeScale(data, baseOffset, perChannelLength, clusterData, offset.data[channelIndex]);
            if (nuqParam.withOffset) {
                offset.data[channelIndex] =
                    OptimizeOffset(data, baseOffset, perChannelLength, clusterData, scale.data[channelIndex]);
            }
        }
        if (iter + 1 == nuqParam.numIter) {
            break;
        } else {
            // Reclustering
            clusterData.assign(data, data + length);
            ClipAndQuant(clusterData, scale, offset, perChannelLength);
            weight = Weight(length, scale, perChannelLength);
            Kmeans(clusterData, nuqParam.numSteps, weight, clusterToStore);
        }
    }
    for (unsigned int i = 0; i < scale.length; i++) {
        ret = util::ProcessScale(scale.data[i]);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("NUQ scale is illegal.\n");
            return AmctCommon::NOT_SUPPORT_ERROR;
        }
    }
    return AmctCommon::SUCCESS;
}


template <class T>
Status NuqKernel(T* data, unsigned int length, const NuqParam &nuqParam, const FloatData &scale, const IntData &offset)
{
    if (scale.length == 0) {
        LOG_ERROR("The scale.length should not be 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    const unsigned int perChannelLength = length / scale.length;
    std::vector<T> clusterData(data, data + length);
    std::vector<T> clusterToStore(nuqParam.numSteps);
    (void)NuqCalibrationCpuKernel(data, clusterData, clusterToStore, length, nuqParam, scale, offset);
    // modify the input data and fake quantize the cluster data
    Restore(data, clusterData, scale, offset, perChannelLength);

    return AmctCommon::SUCCESS;
}

Status NuqQuant(float* data, unsigned int length, const AmctCommon::NuqParam &nuqParam,
                const FloatData &scale, const IntData &offset)
{
    return NuqKernel(data, length, nuqParam, scale, offset);
}

template Status NuqKernel<float>(float*, unsigned int, const NuqParam &, const FloatData &, const IntData &);
template Status NuqCalibrationCpuKernel(const float* data, std::vector<float>& clusterData,
    std::vector<float>& clusterToStore, unsigned int length, const NuqParam &nuqParam,
    const FloatData &scale, const IntData &offset);
template Status NuqCalibrationCpuKernel(const double* data, std::vector<double>& clusterData,
    std::vector<double>& clusterToStore, unsigned int length, const NuqParam &nuqParam,
    const FloatData &scale, const IntData &offset);
}

int NuqQuantPython(double* data, unsigned int length, AmctCommon::NuqParam nuqParam, FloatData scale, IntData offset)
{
    return AmctCommon::NuqKernel(data, length, nuqParam, scale, offset);
}


int NuqQuantRealPython(double* data, unsigned int length, FloatData scale, IntData offset, char* int8Data)
{
    return NuqReal(data, length, scale, offset, int8Data);
}
