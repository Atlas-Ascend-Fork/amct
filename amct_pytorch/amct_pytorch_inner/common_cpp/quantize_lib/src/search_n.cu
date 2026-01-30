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
 * @brief SearchN algorithm CUDA C++ implementation for GPU.
 *
 * @file search_n.cu
 *
 * @version 1.0
 */

#include "search_n.h"
#include <cfloat>
#include <cmath>
#include <thrust/reduce.h>

using namespace util;

namespace AmctCommon {
inline cudaError_t CheckCudaError(cudaError_t error, const char *file, const int line)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
    }
    return error;
}

template<typename inType>
__global__ void EvaluateError(
    const int n,
    inType* bottomData,
    double* topData,
    const int shift) {
    CUDA_KERNEL_LOOP(index, n) {
        inType outputS16 = static_cast<inType>(floor(static_cast<double>(bottomData[index] / shift)));
        if (outputS16 < INT16_MIN) {
            outputS16 = INT16_MIN;
        } else if (outputS16 > INT16_MAX) {
            outputS16 = INT16_MAX;
        }
        outputS16 = outputS16 * shift;
        double singleDistance = static_cast<double>(outputS16 - bottomData[index]);
        topData[index] = singleDistance * singleDistance;
    }
}

template<typename T>
__global__ void QuantForward(
    const int n,
    const int channelSize,
    T* in,
    float* deqScale) {
    CUDA_KERNEL_LOOP(index, n) {
        in[index] = static_cast<int>(round((in[index]) / (deqScale[index / channelSize])));
    }
}

int SearchNInternel(const std::vector<std::vector<int>>& s32Data, std::vector<int>& bestN)
{
    int* searchNDataGpu = 0;
    int channelNum = s32Data.size();
    int perChannelSize = s32Data[0].size();
    int searchNDataChannelSize = perChannelSize;
    int searchNDataSize = perChannelSize * channelNum;
    if (CheckCudaError(cudaMalloc(&searchNDataGpu, sizeof(int) * searchNDataSize), __FILE__, __LINE__)) {
        return AmctCommon::CUDA_ERROR;
    }
    for (int i = 0; i < channelNum; i++) {
        // dst, src, size, HostToDevice
        if (CheckCudaError(cudaMemcpy(searchNDataGpu + i * searchNDataChannelSize,
            &s32Data[i][0], sizeof(int) * searchNDataChannelSize, cudaMemcpyHostToDevice), __FILE__, __LINE__)) {
            return AmctCommon::CUDA_ERROR;
        }
    }
    std::vector<std::vector<double>> bestError(channelNum);
    // shiftNCandidate is [1, 2, 3, ..., 16];
    for (int shiftNCandidate = 1; shiftNCandidate <= 16; shiftNCandidate++) {
        double* topData = nullptr;
        if (CheckCudaError(cudaMalloc(&topData, sizeof(double) * searchNDataSize), __FILE__, __LINE__)) {
            return AmctCommon::CUDA_ERROR;
        }
        EvaluateError<int><<<AMCT_GET_BLOCKS(searchNDataSize), AMCT_CUDA_NUM_THREADS>>>(
            searchNDataSize, searchNDataGpu, topData, pow(2, shiftNCandidate));

        for (int i = 0; i < channelNum; i++) {
            double error = thrust::reduce(thrust::device,
                topData + i * searchNDataChannelSize,
                topData + (i + 1) * searchNDataChannelSize);
            bestError[i].push_back(error);
        }
        cudaFree(topData);
        topData = nullptr;
    }
    cudaFree(searchNDataGpu);
    searchNDataGpu = nullptr;
    for (size_t i = 0; i < bestError.size(); i++) {
        auto smallest = std::min_element(std::begin(bestError[i]), std::end(bestError[i]));
        bestN.push_back(std::distance(std::begin(bestError[i]), smallest) + 1);
    }
    return 0;
}


int SearchNGpu(const std::vector<std::vector<int>>& s32Data, std::vector<int>& bestN)
{
    return SearchNInternel(s32Data, bestN);
}

template<typename T>
int EvaluateSearchNErrorCudaHost(const int searchNDataSize, T* s32Data, double* error, const int shiftBits)
{
    EvaluateError<T><<<AMCT_GET_BLOCKS(searchNDataSize), AMCT_CUDA_NUM_THREADS>>>(
        searchNDataSize, s32Data, error, shiftBits);
    return 0;
}

template<typename T>
int SearchNQuantForwardCudaHost(T* s32Data, float* deqScale,
    const int searchNDataSize, const int searchNDataChannelSize)
{
    QuantForward<T><<<AMCT_GET_BLOCKS(searchNDataSize), AMCT_CUDA_NUM_THREADS>>>(
        searchNDataSize, searchNDataChannelSize, s32Data, deqScale);
    return 0;
}

template int EvaluateSearchNErrorCudaHost(const int, float*, double*, const int);
template int SearchNQuantForwardCudaHost(float*, float*, const int, const int);
}
