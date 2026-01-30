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
 * @brief FIMR algorithm CUDA C++ implementation for GPU.
 *
 * @file ifmr.cu
 *
 * @version 1.0
 */

#include "ifmr.h"
#include <cfloat>
#include <cmath>
#include <functional>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

using namespace util;

#define INFOLEVEL 1
#define DEBUGLEVEL 2
#define WARNINGLEVEL 3
#define ERRORLEVEL 4


#define CHECK_ERROR(error)                                 \
{                                                          \
    Status ret = CheckCudaError(error, __FILE__, __LINE__);\
    if (ret != AmctCommon::SUCCESS) {                                  \
        return ret;                                        \
    }                                                      \
}                                                          \

#define CHECK_FREE(checkPtr)                               \
{                                                          \
    if (checkPtr != NULL) {                                \
        cudaFree(checkPtr);                                \
        checkPtr = NULL;                                   \
    }                                                      \
}                                                          \

#define CHECK_FREE_4(checkPtr, checkPtr1, checkPtr2, checkPtr3)\
{                                                              \
    CHECK_FREE(checkPtr);                                      \
    CHECK_FREE(checkPtr1);                                     \
    CHECK_FREE(checkPtr2);                                     \
    CHECK_FREE(checkPtr3);                                     \
}                                                              \

#define CHECK_ERROR_FREE_4(error, checkPtr, checkPtr1, checkPtr2, checkPtr3)\
{                                                                           \
    Status ret = CheckCudaError(error, __FILE__, __LINE__);                 \
    if (ret != AmctCommon::SUCCESS) {                                                   \
        CHECK_FREE_4(checkPtr, checkPtr1, checkPtr2, checkPtr3);            \
        return ret;                                                         \
    }                                                                       \
}                                                                           \

inline Status CheckCudaError(cudaError_t error, const char *file, const int line)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
        return AmctCommon::CUDA_ERROR;
    }
    return AmctCommon::SUCCESS;
}

using namespace AmctCommon;

// clip function
template <class T>
__global__ void EvalErrorWithOffset(T* devGpuIn,
                                    T MIN,
                                    T MAX,
                                    T* devOut,
                                    struct ClipInfo clipInfo,
                                    const unsigned int inputSize)
{
    const unsigned int numBits = clipInfo.quantBits;
    T valueClip = 0.0;

    CUDA_KERNEL_LOOP(i, inputSize) {
        valueClip = devGpuIn[i];
        if (devGpuIn[i] < MIN) {
            valueClip = MIN;
        }
        if (devGpuIn[i] > MAX) {
            valueClip = MAX;
        }
        double baseNum = 2.0;
        float steps = (T)(pow(baseNum, (double)(numBits)) - 1.0);
        float scale = (T)((MAX - MIN) / steps);

        if (fabs(scale) < DBL_EPSILON) {
            scale = 1.0;
        }

        valueClip = (round(valueClip / scale)) * scale - devGpuIn[i];
        devOut[i] = valueClip * valueClip;
    }
}

// clip function without offset
template <class T>
__global__ void EvalErrorWithoutOffset(T* devGpuIn,
                                       T MAX,
                                       T* devOut,
                                       struct ClipInfo clipInfo,
                                       const unsigned int inputSize)
{
    const unsigned int numBits = clipInfo.quantBits;
    T valueClip = 0.0;
    double baseNum = 2.0;
    float maxLimit = (MAX > 0) ? pow(baseNum, (double)(numBits - 1)) - 1: pow(baseNum, (double)(numBits - 1));
    T absMaxValue = fabs(MAX);

    CUDA_KERNEL_LOOP(i, inputSize) {
        valueClip = devGpuIn[i];
        if (devGpuIn[i] < -absMaxValue) {
            valueClip = -absMaxValue;
        }
        if (devGpuIn[i] > absMaxValue) {
            valueClip = absMaxValue;
        }
        double scale = absMaxValue / maxLimit;

        if (fabs(scale) < DBL_EPSILON) {
            scale = 1.0;
        }

        valueClip = (round(valueClip / scale)) * scale - devGpuIn[i];
        devOut[i] = valueClip * valueClip;
    }
}

namespace AmctCommon {
    template <class T>
    void CalcErrorGpu(T* data,
                      T* clipMin,
                      T* clipMax,
                      T* error,
                      bool withOffset,
                      struct ClipInfo clipInfo,
                      const unsigned int inputSize)
    {
        if (withOffset) {
            EvalErrorWithOffset<<<AMCT_GET_BLOCKS(inputSize), AMCT_CUDA_NUM_THREADS>>>(data, *clipMin, *clipMax,
                error, clipInfo, inputSize);
        } else {
            EvalErrorWithoutOffset<<<AMCT_GET_BLOCKS(inputSize), AMCT_CUDA_NUM_THREADS>>>(data, *clipMax,
                error, clipInfo, inputSize);
        }
    }

    template void CalcErrorGpu(float*, float*, float*, float*, bool, struct ClipInfo, const unsigned int);
}

template <class T>
Status MultiLevel(T* devGpuIn,
                  IfmrParam ifmrParam,
                  T devMin,
                  T maxCandidates,
                  ClipInfo clipInfo,
                  const unsigned int inputSize,
                  T& currentNoise)
{
    T* devOut = 0;
    int elementNum = inputSize;

    // apply gpu memroy for quantlization error
    cudaError_t cudaStatus = cudaMalloc(&devOut, sizeof(T) * (inputSize));
    CHECK_ERROR(cudaStatus);
    // quantize input using by current max and put result in devOut
    if (ifmrParam.withOffset) {
        EvalErrorWithOffset<<<AMCT_GET_BLOCKS(inputSize), AMCT_CUDA_NUM_THREADS>>>(devGpuIn, devMin, maxCandidates,
            devOut, clipInfo, inputSize);
    } else {
        EvalErrorWithoutOffset<<<AMCT_GET_BLOCKS(inputSize), AMCT_CUDA_NUM_THREADS>>>(devGpuIn, maxCandidates,
            devOut, clipInfo, inputSize);
    }

    thrust::device_vector<T> dvectorMult(devOut, devOut+elementNum);
    currentNoise = thrust::reduce(dvectorMult.begin(), dvectorMult.end(), (T) 0, thrust::plus<T>());

    cudaStatus = cudaFree(devOut);
    CHECK_ERROR(cudaStatus);
    devOut = NULL;

    return AmctCommon::SUCCESS;
}

template <class T>
static Status CheckIfmrQuantParams(T* data,
                                   unsigned int length,
                                   IfmrParam ifmrParam,
                                   FloatData scale,
                                   IntData offset)
{
    if (data == NULL) {
        LOG_ERROR("Empty pointer, all input tensor is empty\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (scale.length != 1 || offset.length != 1) {
        LOG_ERROR("scale.length = %d and offset.length = %d should be 1.\n",
            scale.length, offset.length);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    return AmctCommon::SUCCESS;
}

template <class T>
static Status findMaxMinWithPercentile(unsigned int elementNum,
                                       T* devIn,
                                       T* hostDatadata,
                                       T& currentMax,
                                       T& currentMin,
                                       float maxPercentile,
                                       float minPercentile)
{
    if (devIn == NULL) {
        LOG_ERROR("Empty pointer.\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (elementNum == 0) {
        LOG_ERROR("data length is 0.\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    const unsigned int maxSize = (fabs(maxPercentile - 1) < DBL_EPSILON) ?
                                 1 : elementNum - static_cast<uint32_t>(floor(maxPercentile * elementNum));
    const unsigned int minSize = (fabs(minPercentile - 1) < DBL_EPSILON) ?
                                 1 : elementNum - static_cast<uint32_t>(floor(minPercentile * elementNum));

    std::vector<T> inputMax(hostDatadata, hostDatadata + elementNum);
    std::vector<T> inputMin(hostDatadata, hostDatadata + elementNum);

    std::partial_sort(inputMax.begin(), inputMax.begin() + maxSize, inputMax.end(), std::greater<T>());
    std::partial_sort(inputMin.begin(), inputMin.begin() + minSize, inputMin.end());

    currentMax = std::max(currentMax, inputMax[maxSize - 1]);
    currentMin = std::min(currentMin, inputMin[minSize - 1]);

    return AmctCommon::SUCCESS;
}

template <class T>
static Status IfmrQuantData(T* devDatadata,
                            unsigned int length,
                            IfmrParam ifmrParam,
                            T& currentMax,
                            T& currentMin,
                            T* maxValue,
                            T* minValue,
                            unsigned int& bestMaxIndex)
{
    if (devDatadata == NULL) {
        LOG_ERROR("Empty pointer\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    const float startRatio = ifmrParam.startRatio;
    const float endRatio = ifmrParam.endRatio;
    const float step = ifmrParam.step;
    const unsigned int numBits = static_cast<unsigned int>(ifmrParam.numBits);
    const unsigned int maxSize = static_cast<unsigned int>((endRatio - startRatio) / step + 1);
    if(maxSize == 0) {
        LOG_ERROR("maxSize should not be 0.\n");
    }
    double smallestNoise = DBL_MAX;

    for (unsigned int i = 0; i < maxSize; i++) {
        ClipInfo clipInfo = {
                i,
                numBits
        };
        T maxCandidates = 0;
        if (ifmrParam.withOffset) {
            maxCandidates = (startRatio + i * step) * currentMax;
            if (maxCandidates < currentMin) {
                continue;
            }
        } else {
            T absMaxValue = fabs(currentMax) > fabs(currentMin)? currentMax : currentMin;
            maxCandidates = (startRatio + i * step) * absMaxValue;
        }

        // MultiLevel calculate error square sum
        T currentNoise = 0;
        Status ret = MultiLevel(devDatadata, ifmrParam, currentMin, maxCandidates, clipInfo, length, currentNoise);
        if (ret != AmctCommon::SUCCESS) {
            return ret;
        }
        // update optimal max index
        if (smallestNoise > currentNoise) {
            smallestNoise = currentNoise;
            bestMaxIndex = i;
        }
    }
    return AmctCommon::SUCCESS;
}

template <class T>
static Status IfmrQuantCalibration(T* devDatadata,
                                   T* hostDatadata,
                                   unsigned int length,
                                   IfmrParam ifmrParam,
                                   FloatData scale,
                                   IntData offset)
{
    Status ret = CheckIfmrQuantParams(devDatadata, length, ifmrParam, scale, offset);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("CheckIfmrQuantParams failed.\n");
        return ret;
    }
    // cuda->cpu init
    IfmrParam hIfmrParam = ifmrParam;
    float startRatio = hIfmrParam.startRatio;
    float step = hIfmrParam.step;
    const unsigned int numBits = hIfmrParam.numBits;
    const unsigned int inputSize = length;
    T* maxValue = NULL;
    T* minValue = NULL;
    T* devMinOut = NULL;
    T* devMaxOut = NULL;
    cudaError_t cudaStatus = cudaMalloc(&maxValue, sizeof(T) * (1));
    CHECK_ERROR_FREE_4(cudaStatus, maxValue, minValue, devMinOut, devMaxOut);
    cudaStatus = cudaMalloc(&minValue, sizeof(T) * (1));
    CHECK_ERROR_FREE_4(cudaStatus, maxValue, minValue, devMinOut, devMaxOut);
    unsigned int elementNum = inputSize;
    cudaStatus = cudaMalloc(&devMaxOut, sizeof(T) * (1));
    CHECK_ERROR_FREE_4(cudaStatus, maxValue, minValue, devMinOut, devMaxOut);
    cudaStatus = cudaMalloc(&devMinOut, sizeof(T) * (1));
    CHECK_ERROR_FREE_4(cudaStatus, maxValue, minValue, devMinOut, devMaxOut);
    T currentMax = 0, currentMin = 0;
    findMaxMinWithPercentile(elementNum, devDatadata, hostDatadata, currentMax,
        currentMin, hIfmrParam.maxPercentile, hIfmrParam.minPercentile);

    unsigned int bestMaxIndex = 0;
    float baseNum = 2.0;

    if (ifmrParam.withOffset){
        Status ret = IfmrQuantData(devDatadata, elementNum, ifmrParam,
            currentMax, currentMin, maxValue, minValue, bestMaxIndex);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("IfmrQuantWithOffset failed.\n");
            CHECK_FREE_4(devMinOut, maxValue, minValue, devMaxOut);
            return ret;
        }

        float maxLimit = pow(baseNum, numBits) - 1;
        float bestMaxValue = (startRatio + bestMaxIndex * step) * currentMax;
        int NUM_LIMIT = static_cast<int>(pow(baseNum, numBits - 1));

        scale.data[0] = (bestMaxValue - currentMin) / maxLimit;
        Status result = util::ProcessScale(scale.data[0]);
        if (result != AmctCommon::SUCCESS) {
            LOG_ERROR("IfmrQuantCalibration with offset scale is illegal.\n");
            CHECK_FREE_4(devMinOut, maxValue, minValue, devMaxOut);
            return result;
        }
        offset.data[0] = -static_cast<int>(round(currentMin / scale.data[0])) - NUM_LIMIT;
    } else {
        Status ret = IfmrQuantData(devDatadata, elementNum, ifmrParam, currentMax,
            currentMin, maxValue, minValue, bestMaxIndex);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("IfmrQuantWithoutOffset failed.\n");
            CHECK_FREE_4(devMinOut, maxValue, minValue, devMaxOut);
            return ret;
        }

        const float min = pow(baseNum, numBits - 1);
        const float max = pow(baseNum, numBits - 1) - 1;
        float absMaxValue = (fabs(currentMax) > fabs(currentMin))? currentMax : currentMin;
        float bestMaxValue = (startRatio + bestMaxIndex * step) * fabs(absMaxValue);
        scale.data[0] = (absMaxValue > 0) ? bestMaxValue / max : bestMaxValue / min;
        Status result = util::ProcessScale(scale.data[0]);
        if (result != AmctCommon::SUCCESS) {
            LOG_ERROR("IfmrQuantCalibration without offset scale is illegal.\n");
            CHECK_FREE_4(devMinOut, maxValue, minValue, devMaxOut);
            return result;
        }
        offset.data[0] = 0;
    }

    cudaFree(devMinOut);
    devMinOut = NULL;
    cudaFree(devMaxOut);
    devMaxOut = NULL;
    cudaFree(maxValue);
    maxValue = NULL;
    cudaFree(minValue);
    minValue = NULL;

    return AmctCommon::SUCCESS;
}

template <class T>
__global__ void ImfrQuantQuantize(T* data,
                                  unsigned int dlength,
                                  int numBits,
                                  int minLimit,
                                  int maxLimit,
                                  float scaleValue,
                                  int offsetValue)
{
    unsigned int tid = static_cast<unsigned int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (tid < dlength) {
        int dataTmp = static_cast<int>(round(data[tid] / scaleValue)) + offsetValue;
        if (dataTmp < minLimit) {
            dataTmp = minLimit;
        }
        if (dataTmp > maxLimit) {
            dataTmp = maxLimit;
        }
        data[tid] = (dataTmp - offsetValue) * scaleValue;
    }
}

template <class T>
static Status IfmrQuantInference(T* data,
                                 unsigned int length,
                                 IfmrParam ifmrParam,
                                 FloatData scale,
                                 IntData offset)
{
    Status ret = CheckIfmrQuantParams(data, length, ifmrParam, scale, offset);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("CheckIfmrQuantParams failed.\n");
        return ret;
    }

    int blockNum = (length / AMCT_CUDA_NUM_THREADS) + ((length % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
    float scaleValue = 0.0;
    int offsetValue = 0;
    scaleValue = *(scale.data);
    offsetValue = *(offset.data);
    if (scaleValue < FLT_EPSILON || std::isinf(scaleValue)) {
        LOG_ERROR("IfmrQuantInference scale is zero or inf!\n");
        return AmctCommon::NOT_SUPPORT_ERROR;
    }
    const unsigned int numBits = static_cast<unsigned int>(ifmrParam.numBits);
    const int baseNum = 2;
    const int minLimit = -static_cast<int>(pow(baseNum, numBits - 1));
    const int maxLimit = static_cast<int>(pow(baseNum, numBits -1) -1);

    ImfrQuantQuantize<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(data, length, numBits, minLimit,
        maxLimit, scaleValue, offsetValue);

    return AmctCommon::SUCCESS;
}

template <class T>
static Status IfmrQuantInternel(T* deviceData,
                                T* hostDatadata,
                                unsigned int length,
                                IfmrParam ifmrParam,
                                FloatData scale,
                                IntData offset)
{
    if (deviceData == NULL) {
        LOG_ERROR("Empty pointer\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (ifmrParam.calibration >1) {
        LOG_ERROR("ifmrParam.calibration = %d.\n", ifmrParam.calibration);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (ifmrParam.calibration == 0) {
        T* data = NULL;
        cudaError_t cudaStatus = cudaMalloc(&data, sizeof(T) * length);
        CHECK_ERROR(cudaStatus);
        CHECK_ERROR(cudaMemcpy(data, hostDatadata, length * sizeof(T), cudaMemcpyHostToDevice));
        Status ret = IfmrQuantCalibration(data, hostDatadata, length, ifmrParam, scale, offset);
        cudaStatus = cudaFree(data);
        CHECK_ERROR(cudaStatus);
        data = NULL;
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("IfmrQuantInternel calibration failed.\n");
        }
        return ret;
    } else {
        Status ret = IfmrQuantInference(deviceData, length, ifmrParam, scale, offset);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("IfmrQuantInternel inference failed.\n");
        }
        return ret;
    }
}

namespace AmctCommon {
int IfmrQuantGpu(float* data,
                  float* hostDatadata,
                  unsigned int length,
                  IfmrParam ifmrParam,
                  FloatData scale,
                  IntData offset)
{
    return IfmrQuantInternel(data, hostDatadata, length, ifmrParam, scale, offset);
}

int IfmrQuantGpu(double* data,
                  double* hostDatadata,
                  unsigned int length,
                  IfmrParam ifmrParam,
                  FloatData scale,
                  IntData offset)
{
    return IfmrQuantInternel(data, hostDatadata, length, ifmrParam, scale, offset);
}
}