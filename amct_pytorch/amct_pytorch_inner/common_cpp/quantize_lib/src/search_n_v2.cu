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
 * @brief search n v2 shift bit core algorithm cuda implementation
 *
 * @version 1.0
 */

#include "search_n_v2.h"
#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <thrust/reduce.h>

using namespace util;

namespace AmctCommon {
template <typename T>
__global__ void EvaluateErrorKernelPerTensor(
    const int n,
    const T* in,
    float* deqScale,
    T* quantErrorSum,
    const int shiftValue,
    const int shiftBound)
{
    __shared__ T l2Loss[AMCT_CUDA_NUM_THREADS];
    unsigned int tid = threadIdx.x;
    l2Loss[tid] = 0;
    __syncthreads();
    CUDA_KERNEL_LOOP(index, n) {
        T s32Data = round(in[index] / deqScale[0]);
        T s16Data = floor(s32Data / shiftValue);
        if (s16Data < -shiftBound) {
            s16Data = -shiftBound;
        } else if (s16Data > (shiftBound - 1)) {
            s16Data = shiftBound - 1;
        }
        s16Data = s16Data * shiftValue;
        l2Loss[tid] = pow((s16Data - s32Data), BASE);
        __syncthreads();
    }
    // 0 thread reduce the l2loss and write back to global memory
    if (tid == 0) {
        T l2LossSum = 0;
        for (unsigned int i = 0; i < AMCT_CUDA_NUM_THREADS; i++) {
            l2LossSum += l2Loss[i];
        }
        quantErrorSum[blockIdx.x] = l2LossSum;
    }
}


template <typename T>
__global__ void EvaluateErrorNchwKernel(
    const int n,
    const T* in,
    T* quantErrorSum,
    float* deqScale,
    const int curChannel,
    const int shiftValue,
    const int shiftBound,
    struct NchwShapeInfo shapeInfo)
{
    __shared__ T l2Loss[AMCT_CUDA_NUM_THREADS];
    unsigned int tid = threadIdx.x;
    l2Loss[tid] = 0;
    __syncthreads();
    CUDA_KERNEL_LOOP(index, n) {
        T tmpL2loss = 0;
        int channelId = (index / shapeInfo.hwFactor) % shapeInfo.cFactor;

        T s32Data = round(in[index] / deqScale[channelId]);
        T s16Data = floor(s32Data / shiftValue);
        if (s16Data < -shiftBound) {
            s16Data = -shiftBound;
        } else if (s16Data > (shiftBound - 1)) {
            s16Data = shiftBound - 1;
        }
        s16Data = s16Data * shiftValue;
        tmpL2loss = channelId == curChannel ? pow((s16Data - s32Data), BASE) : 0;
        l2Loss[tid] = tmpL2loss;
        __syncthreads();
    }
    // 0 thread reduce the l2loss and write back to global memory
    if (tid == 0) {
        T l2LossSum = 0;
        for (unsigned int i = 0; i < AMCT_CUDA_NUM_THREADS; i++) {
            l2LossSum += l2Loss[i];
        }
        quantErrorSum[blockIdx.x] = l2LossSum;
    }
}


template <typename T>
__global__ void EvaluateErrorNchwBroadcastKernel(
    const int n,
    const T* in,
    T* quantErrorSum,
    float* deqScale,
    const int shiftValue,
    const int shiftBound,
    struct NchwShapeInfo shapeInfo)
{
    __shared__ T l2Loss[AMCT_CUDA_NUM_THREADS];
    unsigned int tid = threadIdx.x;
    l2Loss[tid] = 0;
    __syncthreads();
    CUDA_KERNEL_LOOP(index, n) {
        int channelId = (index / shapeInfo.hwFactor) % shapeInfo.cFactor;
        T s32Data = round(in[index] / deqScale[channelId]);
        T s16Data = floor(s32Data / shiftValue);
        if (s16Data < -shiftBound) {
            s16Data = -shiftBound;
        } else if (s16Data > (shiftBound - 1)) {
            s16Data = shiftBound - 1;
        }
        s16Data = s16Data * shiftValue;
        l2Loss[tid] = pow((s16Data - s32Data), BASE);
        __syncthreads();
    }
    // 0 thread reduce the l2loss and write back to global memory
    if (tid == 0) {
        T l2LossSum = 0;
        for (unsigned int i = 0; i < AMCT_CUDA_NUM_THREADS; i++) {
            l2LossSum += l2Loss[i];
        }
        quantErrorSum[blockIdx.x] = l2LossSum;
    }
}


template <typename T>
__global__ void EvaluateErrorNhwcKernel(
    const int n,
    const T* in,
    T* quantErrorSum,
    float* deqScale,
    const int curChannel,
    const int shiftValue,
    const int shiftBound,
    struct NhwcShapeInfo shapeInfo)
{
    __shared__ T l2Loss[AMCT_CUDA_NUM_THREADS];
    unsigned int tid = threadIdx.x;
    l2Loss[tid] = 0;
    __syncthreads();
    CUDA_KERNEL_LOOP(index, n) {
        T tmpL2loss = 0;
        int channelId = index % shapeInfo.cFactor;
        T s32Data = round(in[index] / deqScale[channelId]);
        T s16Data = floor(s32Data / shiftValue);
        if (s16Data < -shiftBound) {
            s16Data = -shiftBound;
        } else if (s16Data > (shiftBound - 1)) {
            s16Data = shiftBound - 1;
        }
        s16Data = s16Data * shiftValue;
        tmpL2loss = (channelId == curChannel) ? pow((s16Data - s32Data), BASE) : 0;
        l2Loss[tid] = tmpL2loss;
        __syncthreads();
    }
    // 0 thread reduce the l2loss and write back to global memory
    if (tid == 0) {
        T l2LossSum = 0;
        for (unsigned int i = 0; i < AMCT_CUDA_NUM_THREADS; i++) {
            l2LossSum += l2Loss[i];
        }
        quantErrorSum[blockIdx.x] = l2LossSum;
    }
}


template <typename T>
__global__ void EvaluateErrorNhwcBroadcastKernel(
    const int n,
    const T* in,
    T* quantErrorSum,
    float* deqScale,
    const int shiftValue,
    const int shiftBound,
    struct NhwcShapeInfo shapeInfo)
{
    __shared__ T l2Loss[AMCT_CUDA_NUM_THREADS];
    unsigned int tid = threadIdx.x;
    l2Loss[tid] = 0;
    __syncthreads();
    CUDA_KERNEL_LOOP(index, n) {
        int channelId = index % shapeInfo.cFactor;
        T s32Data = round(in[index] / deqScale[channelId]);
        T s16Data = floor(s32Data / shiftValue);
        if (s16Data < -shiftBound) {
            s16Data = -shiftBound;
        } else if (s16Data > (shiftBound - 1)) {
            s16Data = shiftBound - 1;
        }
        s16Data = s16Data * shiftValue;
        l2Loss[tid] = pow((s16Data - s32Data), BASE);
        __syncthreads();
    }
    // 0 thread reduce the l2loss and write back to global memory
    if (tid == 0) {
        T l2LossSum = 0;
        for (unsigned int i = 0; i < AMCT_CUDA_NUM_THREADS; i++) {
            l2LossSum += l2Loss[i];
        }
        quantErrorSum[blockIdx.x] = l2LossSum;
    }
}


int CheckNhwcShapeInfo(struct NhwcShapeInfo& shapeInfo)
{
    if (shapeInfo.cFactor == 0) {
        LOG_ERROR("The NhwcShapeInfo.cFactor should not be 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if (shapeInfo.channelSize == 0) {
        LOG_ERROR("The NhwcShapeInfo.channelSize should not be 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    return AmctCommon::SUCCESS;
}


int CheckNchwShapeInfo(struct NchwShapeInfo& shapeInfo)
{
    if (shapeInfo.cFactor == 0) {
        LOG_ERROR("The NchwShapeInfo.cFactor should not be 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if (shapeInfo.nFactor == 0) {
        LOG_ERROR("The NchwShapeInfo.nFactor should not be 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if (shapeInfo.hwFactor == 0) {
        LOG_ERROR("The NchwShapeInfo.hwFactor should not be 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if (shapeInfo.channelSize == 0) {
        LOG_ERROR("The NchwShapeInfo.channelSize should not be 0!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    return AmctCommon::SUCCESS;
}


template <typename T>
int AccumulateErrorPerChannelNchwCuda(
    struct SearchnV2InputParam<T>& inputParam,
    struct SearchnV2AlgoParam<T>& algoParam,
    struct NchwShapeInfo& shapeInfo)
{
    int status = CheckNchwShapeInfo(shapeInfo);
    if (status != AmctCommon::SUCCESS) {
        return status;
    }
    int searchNDataSize = inputParam.size;
    for (int shiftNCandidate = 1; shiftNCandidate <= SHIFT_BITS; shiftNCandidate++) {
        if (algoParam.isBroadcast) {
            // dequant to s32 is perChannel but reduce l2error perTensor;
            EvaluateErrorNchwBroadcastKernel<T><<<AMCT_GET_BLOCKS(searchNDataSize), AMCT_CUDA_NUM_THREADS>>>(
                inputParam.size, inputParam.in, algoParam.quantErrorSum, algoParam.deqScale,
                static_cast<int>(pow(BASE, shiftNCandidate)), static_cast<int>(pow(BASE, SHIFT_POW)), shapeInfo);
            T error = thrust::reduce(
                thrust::device, algoParam.quantErrorSum,
                algoParam.quantErrorSum + AMCT_GET_BLOCKS(searchNDataSize));
            for (size_t i = 0; i < inputParam.storeError.size(); i++) {
                inputParam.storeError[i][shiftNCandidate - 1] += error;
            }
        } else {
            for (size_t channelId = 0; channelId < inputParam.storeError.size(); channelId++) {
                EvaluateErrorNchwKernel<T><<<AMCT_GET_BLOCKS(searchNDataSize), AMCT_CUDA_NUM_THREADS>>>(
                    inputParam.size, inputParam.in, algoParam.quantErrorSum, algoParam.deqScale, channelId,
                    pow(BASE, shiftNCandidate), pow(BASE, SHIFT_POW), shapeInfo);
                T error = thrust::reduce(
                    thrust::device, algoParam.quantErrorSum,
                    algoParam.quantErrorSum + AMCT_GET_BLOCKS(searchNDataSize));
                inputParam.storeError[channelId][shiftNCandidate - 1] += error;
            }
        }
    }
    return AmctCommon::SUCCESS;
}


template <typename T>
int AccumulateErrorPerChannelNhwcCuda(
    struct SearchnV2InputParam<T>& inputParam,
    struct SearchnV2AlgoParam<T>& algoParam,
    struct NhwcShapeInfo& shapeInfo)
{
    int status = CheckNhwcShapeInfo(shapeInfo);
    if (status != AmctCommon::SUCCESS) {
        return status;
    }
    int searchNDataSize = inputParam.size;
    for (int shiftNCandidate = 1; shiftNCandidate <= SHIFT_BITS; shiftNCandidate++) {
        if (algoParam.isBroadcast) {
            // dequant to s32 is perChannel but reduce l2error perTensor;
            EvaluateErrorNhwcBroadcastKernel<T><<<AMCT_GET_BLOCKS(searchNDataSize), AMCT_CUDA_NUM_THREADS>>>(
                inputParam.size, inputParam.in, algoParam.quantErrorSum, algoParam.deqScale,
                static_cast<int>(pow(BASE, shiftNCandidate)), static_cast<int>(pow(BASE, SHIFT_POW)), shapeInfo);
            T error = thrust::reduce(
                thrust::device, algoParam.quantErrorSum,
                algoParam.quantErrorSum + AMCT_GET_BLOCKS(searchNDataSize));
            for (size_t i = 0; i < inputParam.storeError.size(); i++) {
                inputParam.storeError[i][shiftNCandidate - 1] += error;
            }
        } else {
            for (size_t channelId = 0; channelId < inputParam.storeError.size(); channelId++) {
                EvaluateErrorNhwcKernel<T><<<AMCT_GET_BLOCKS(searchNDataSize), AMCT_CUDA_NUM_THREADS>>>(
                    inputParam.size, inputParam.in, algoParam.quantErrorSum, algoParam.deqScale, channelId,
                    pow(BASE, shiftNCandidate), pow(BASE, SHIFT_POW), shapeInfo);
                T error = thrust::reduce(
                    thrust::device, algoParam.quantErrorSum,
                    algoParam.quantErrorSum + AMCT_GET_BLOCKS(searchNDataSize));
                inputParam.storeError[channelId][shiftNCandidate - 1] += error;
            }
        }
    }
    return AmctCommon::SUCCESS;
}


template <typename T>
int AccumulateErrorPerTensorCuda(
    struct SearchnV2InputParam<T>& inputParam,
    struct SearchnV2AlgoParam<T>& algoParam)
{
    int searchNDataSize = inputParam.size;
    for (int shiftNCandidate = 1; shiftNCandidate <= SHIFT_BITS; shiftNCandidate++) {
        EvaluateErrorKernelPerTensor<T><<<AMCT_GET_BLOCKS(searchNDataSize), AMCT_CUDA_NUM_THREADS>>>(
            inputParam.size, inputParam.in, algoParam.deqScale, algoParam.quantErrorSum,
            static_cast<int>(pow(BASE, shiftNCandidate)), static_cast<int>(pow(BASE, SHIFT_POW)));
        T error = thrust::reduce(
            thrust::device, algoParam.quantErrorSum, algoParam.quantErrorSum + AMCT_GET_BLOCKS(searchNDataSize));
        inputParam.storeError[0][shiftNCandidate - 1] += error;
    }
    return AmctCommon::SUCCESS;
}


template int AccumulateErrorPerChannelNchwCuda(
    struct SearchnV2InputParam<float>& inputParam,
    struct SearchnV2AlgoParam<float>& algoParam,
    struct NchwShapeInfo& shapeInfo);

template int AccumulateErrorPerChannelNhwcCuda(
    struct SearchnV2InputParam<float>& inputParam,
    struct SearchnV2AlgoParam<float>& algoParam,
    struct NhwcShapeInfo& shapeInfo);

template int AccumulateErrorPerTensorCuda(
    struct SearchnV2InputParam<float>& inputParam,
    struct SearchnV2AlgoParam<float>& algoParam);

template int AccumulateErrorPerChannelNchwCuda(
    struct SearchnV2InputParam<double>& inputParam,
    struct SearchnV2AlgoParam<double>& algoParam,
    struct NchwShapeInfo& shapeInfo);

template int AccumulateErrorPerChannelNhwcCuda(
    struct SearchnV2InputParam<double>& inputParam,
    struct SearchnV2AlgoParam<double>& algoParam,
    struct NhwcShapeInfo& shapeInfo);

template int AccumulateErrorPerTensorCuda(
    struct SearchnV2InputParam<double>& inputParam,
    struct SearchnV2AlgoParam<double>& algoParam);
}
