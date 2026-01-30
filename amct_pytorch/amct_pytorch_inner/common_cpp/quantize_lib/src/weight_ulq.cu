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
 * @brief weight ulq cuda implementation
 *
 * @version 1.0
 */

#if GOOGLE_CUDA
#include <math.h>
#include <thrust/reduce.h>

#include "weight_ulq.h"
#include "util.h"

using namespace util;

template <typename T>
__global__ void ScaleArqInitKernel(
    const int size,
    const T* in,
    T* max,
    T* min,
    AmctCommon::WeightUlqParam quantParam)
{
    int index = blockIdx.x;
    max[index] = 0.0;
    min[index] = 0.0;
    for (int i = index; i < size; i += gridDim.x) {
        max[index] = (in[i] > max[index]) ? in[i] : max[index];
        min[index] = (in[i] < min[index]) ? in[i] : min[index];
    }

    int steps = pow(BASE, quantParam.quantBits - 1);
    if (steps > 1) {
        quantParam.scale[index] = (-1 * min[index] > max[index]) ? -1 * min[index] / steps : max[index] / (steps - 1);
        quantParam.scale[index] = (quantParam.scale[index] < EPSILON) ? 1.0 : quantParam.scale[index];
        // when sRecFlag is true, need to save the 1 / scale;
        quantParam.scale[index] = quantParam.sRecFlag ? 1 / quantParam.scale[index] : quantParam.scale[index];
    }
}

template <typename T>
__global__ void ClipAndQuantCuda(AmctCommon::Input<T> input,
                                 T* outputData,
                                 const float* scale,
                                 int quantBitNum,
                                 bool sRecFlag)
{
    int index = blockIdx.x;
    float curScale = scale[index];
    if (fabs(curScale) < EPSILON) {
        curScale += EPSILON;
    }

    curScale = sRecFlag ? 1 / curScale : curScale;
    T minValue = - pow(2.0, quantBitNum - 1);
    T maxValue = pow(2.0, quantBitNum - 1) - 1;

    for (int i = index; i < input.length; i += gridDim.x) {
        T quantValue = round(input.data[i] / curScale);
        if (quantValue < minValue) {
            quantValue = minValue;
        } else if (quantValue > maxValue) {
            quantValue = maxValue;
        }
        outputData[i] = quantValue * curScale;
    }
}

template <typename T>
__global__ void ClipAndQuantCudaPerTensor(const T* intputData,
                                          int inputLength,
                                          T* outputData,
                                          const float* scale,
                                          int quantBitNum,
                                          bool sRecFlag)
{
    float curScale = scale[0];
    if (fabs(curScale) < EPSILON) {
        curScale += EPSILON;
    }
    curScale = sRecFlag ? 1 / curScale : curScale;
    T minValue = - pow(2.0, quantBitNum - 1);
    T maxValue = pow(2.0, quantBitNum - 1) - 1;
    CUDA_KERNEL_LOOP(index, inputLength) {
        T quantValue = round(intputData[index] / curScale);
        if (quantValue < minValue) {
            quantValue = minValue;
        } else if (quantValue > maxValue) {
            quantValue = maxValue;
        }
        outputData[index] = quantValue * curScale;
    }
}

__global__ void ProcessScaleCuda(const float* scaleIn,
                                 float* scaleOut,
                                 int* offsetOut,
                                 int scaleLength,
                                 bool sRecFlag)
{
    CUDA_KERNEL_LOOP(index, scaleLength) {
        float curScale = fabs(scaleIn[index]);
        if (fabs(curScale) < EPSILON) {
            curScale += EPSILON;
        }

        if (sRecFlag) {
            scaleOut[index] = 1 / curScale;
        } else {
            scaleOut[index] = curScale;
        }
        offsetOut[index] = 0;
    }
}

namespace AmctCommon
{
    template <typename T>
    int ScaleArqInitCudaHost(const int size,
                             const T* in,
                             T* max,
                             T* min,
                             AmctCommon::WeightUlqParam quantParam)
    {
        const int blockCount = quantParam.scaleLength;
        const int threadPerBlock = 1;

        ScaleArqInitKernel<T><<<blockCount, threadPerBlock>>>(size, in, max, min, quantParam);
        return AmctCommon::SUCCESS;
    }

    template <typename T>
    int WtsFakeQuantCudaHost(Input<T> input,
                             T* output,
                             const float* scale,
                             int quantBitNum,
                             bool sRecFlag)
    {
        if (input.scaleLength == 1) {
            ClipAndQuantCudaPerTensor<T><<<AMCT_GET_BLOCKS(input.length), AMCT_CUDA_NUM_THREADS>>>(
                input.data, input.length, output, scale, quantBitNum, sRecFlag);
        } else {
            const int blockCount = input.scaleLength;
            const int threadPerBlock = 1;
            ClipAndQuantCuda<T><<<blockCount, threadPerBlock>>>(input, output, scale, quantBitNum, sRecFlag);
        }
        return AmctCommon::SUCCESS;
    }

    void ProcessScaleCudaHost(const float* scaleIn,
                              float* scaleOut,
                              int* offsetOut,
                              int scaleLength,
                              bool sRecFlag)
    {
        ProcessScaleCuda<<<AMCT_GET_BLOCKS(scaleLength), AMCT_CUDA_NUM_THREADS>>>(
            scaleIn, scaleOut, offsetOut, scaleLength, sRecFlag);
    }

    // Explicitly instantiate functors for the types of OpKernels registered.
    template int ScaleArqInitCudaHost(const int, const float*, float*, float*, AmctCommon::WeightUlqParam);
    template int WtsFakeQuantCudaHost(Input<float>, float*, const float*, int, bool);
}

#endif  // GOOGLE_CUDA
