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
 * @brief ULQ algorithm CUDA C++ implementation for GPU.
 *
 * @file ulq.cu
 *
 * @version 1.0
 */

#include "ulq.h"
#include <cfloat>
#include <cmath>
#include <numeric>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <ctime>
#include <thrust/scan.h>

using namespace util;

namespace AmctCommon {
template <typename Dtype>
__global__ void BlobCopy(const int n, const Dtype* in, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = in[index];
    }
}

template <typename Dtype>
__global__ void WeightQuantCudaKernel(const Dtype* in, Dtype* out, const int quantBits, int kernelSize)
{
    int offset = blockIdx.x;
    Dtype scale = 0;
    for (int i = offset * kernelSize; i < (offset + 1) * kernelSize; i += 1) {
        if (abs(scale) < abs(in[i]))
        {
            scale = in[i];
        }
    }
    if (scale < 0) {
        scale = abs(scale) / (pow(2, quantBits - 1));
    } else {
        scale = abs(scale) / (pow(2, quantBits - 1) - 1);
    }
    for (int i = offset * kernelSize; i < (offset + 1) * kernelSize; i += 1) {
        out[i] = ::round(in[i] / scale) * scale;
    }
}

template <typename Dtype>
__global__ void WeightQuantCudaKernelFc(const int n, const Dtype* in, Dtype* out, const int quantBits,
    Dtype* max, Dtype* min)
{
    Dtype scale = 0;
    if (max[0] > abs(min[0])) {
        scale = abs(max[0]) / (pow(2, quantBits - 1));
    } else {
        scale = abs(min[0]) / (pow(2, quantBits - 1) - 1);
    }
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = ::round(in[index] / scale) * scale;
    }
}

//activation quant process
template <class Dtype>
class SharedMem {
public:
    Dtype* getPointer() {
      return NULL;
    };
};

template <>
class SharedMem <int> {
public:
    __device__ int* getPointer() {
      extern __shared__ int s_int[]; return s_int;
    }
};

template <>
class SharedMem <float> {
public:
    __device__ float* getPointer() {
      extern __shared__ float s_float[]; return s_float;
    }
};

template <>
class SharedMem <double> {
public:
    __device__ double* getPointer() {
      extern __shared__ double s_double[]; return s_double;
    }
};


template <typename Dtype>
__global__ void ReduceSum(const int count, Dtype* dataIn)
{
    SharedMem<Dtype> shared;
    Dtype* shareSpace = shared.getPointer();

    Dtype transferVar = 0;
    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < count) {
        transferVar = dataIn[globalIndex];
    }
    shareSpace[tid] = transferVar;
    __syncthreads();

    const int halfNum = 2;
    for (int validThread = blockDim.x / halfNum; validThread > 0; validThread /= halfNum) {
        if (threadIdx.x < validThread) {
            shareSpace[threadIdx.x] += shareSpace[threadIdx.x + validThread];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dataIn[blockIdx.x] = shareSpace[0];
    }
}

template <typename Dtype>
__global__ void QuantForward(const int n, const Dtype* in, Dtype* out,
    Dtype* clipMaxOri, Dtype* clipMinOri, bool fixedMin, int quantBitNum) {
        Dtype scale = (clipMaxOri[0] - clipMinOri[0]) / (pow(2, quantBitNum) - 1);
        Dtype offset = ::round(clipMinOri[0] / scale);

        Dtype clipMax, clipMin;
        clipMax = scale * (offset + pow(2, quantBitNum) - 1);
        clipMin = scale * offset;

  CUDA_KERNEL_LOOP(index, n) {
        out[index] = in[index];
        if (in[index] < clipMin) {
            out[index] =  clipMin;
        } else if(in[index] > clipMax) {
            out[index] =  clipMax;
        }
        out[index] = ::round(out[index] / scale) * scale;
    }
}

template <typename Dtype>
__global__ void QuantBackwardMax(
    const int n,
    const Dtype* bottomData,
    const Dtype* topDiff,
    Dtype* bottomDiff,
    const Dtype* clipMaxOri,
    const Dtype* clipMinOri,
    int quantBitNum) {
        Dtype scale = (clipMaxOri[0] - clipMinOri[0]) / (pow(2, quantBitNum) - 1);
        Dtype offset = ::round(clipMinOri[0] / scale);
        Dtype clipMax, clipMin;
        clipMax = scale * (offset + pow(2, quantBitNum) - 1);
        clipMin = scale * offset;

        Dtype N = pow(2, quantBitNum) - 1;
        Dtype temp = clipMinOri[0] * N / (clipMaxOri[0] - clipMinOri[0]);
        Dtype result = 1 / N * (::round(temp) - temp);

    CUDA_KERNEL_LOOP(index, n) {
        if (bottomData[index] > clipMax) {
            bottomDiff[index] = topDiff[index] * (result + 1);
        } else if(bottomData[index] < clipMin) {
            bottomDiff[index] = topDiff[index] * result;
        } else {
            Dtype temp = bottomData[index] * N / (clipMaxOri[0] - clipMinOri[0]);
            Dtype result = 1 / N * (::round(temp) - temp);
            bottomDiff[index] = topDiff[index] * result;
        }
    }
}

template <typename Dtype>
__global__ void QuantBackwardMin(
    const int n,
    const Dtype* bottomData,
    const Dtype* topDiff,
    Dtype* bottomDiff,
    const Dtype* clipMaxOri,
    const Dtype* clipMinOri,
    int quantBitNum) {
        Dtype scale = (clipMaxOri[0] - clipMinOri[0]) / (pow(2, quantBitNum) - 1);
        Dtype offset = ::round(clipMinOri[0] / scale);
        Dtype clipMax, clipMin;
        clipMax = scale * (offset + pow(2, quantBitNum) - 1);
        clipMin = scale * offset;

        Dtype N = pow(2, quantBitNum) - 1;
        Dtype temp = clipMinOri[0] * N / (clipMaxOri[0] - clipMinOri[0]);
        Dtype result = 1 / N * (::round(temp) - temp);

    CUDA_KERNEL_LOOP(index, n) {
        if (bottomData[index] > clipMax) {
            bottomDiff[index] = topDiff[index] * -result;
        } else if(bottomData[index] < clipMin) {
            bottomDiff[index] = topDiff[index] * (1 - result);
        } else {
            Dtype temp = bottomData[index] * N / (clipMaxOri[0] - clipMinOri[0]);
            Dtype result = 1 / N * (::round(temp) - temp);
            bottomDiff[index] = topDiff[index] * -result;
        }
    }
}

template <typename Dtype>
__global__ void QuantBackward(
    const int n,
    const Dtype* in_data,
    Dtype* in_diff,
    const Dtype* clipMax,
    const Dtype* clipMin) {
    CUDA_KERNEL_LOOP(index, n) {
        if (in_data[index] < clipMin[0]) {
            in_diff[index] = 0;
        } else if(in_data[index] > clipMax[0]) {
            in_diff[index] = 0;
        }
    }
}

template <typename Dtype>
Dtype MultiSum(int inputOriginDataNum, Dtype* inputOriginData) {
    ReduceSum<Dtype><<<AMCT_GET_BLOCKS(inputOriginDataNum), AMCT_CUDA_NUM_THREADS,
        AMCT_CUDA_NUM_THREADS * sizeof(Dtype)>>>(inputOriginDataNum, inputOriginData);

    int origin_data_num = AMCT_GET_BLOCKS(inputOriginDataNum);
    while (origin_data_num > AMCT_CUDA_NUM_THREADS) {
        ReduceSum<Dtype><<<AMCT_GET_BLOCKS(origin_data_num), AMCT_CUDA_NUM_THREADS,
            AMCT_CUDA_NUM_THREADS * sizeof(Dtype)>>>(origin_data_num, inputOriginData);
        origin_data_num = AMCT_GET_BLOCKS(origin_data_num);
    }

    ReduceSum<Dtype><<<1, AMCT_CUDA_NUM_THREADS, AMCT_CUDA_NUM_THREADS * sizeof(Dtype)>>>(
        origin_data_num, inputOriginData);

    Dtype sumDataOut = 0;
    cudaMemcpy(&sumDataOut, inputOriginData, sizeof(Dtype) * (1), cudaMemcpyDeviceToHost);

    return sumDataOut;
}


template <typename Dtype>
int ActQuantForwardGpuInternel(const int count, const Dtype* bottomData, Dtype* topData, Dtype* clipMaxGpu,
    Dtype* clipMinGpu, bool fixedMin, int quantBitNum)
{
        QuantForward<Dtype><<<AMCT_GET_BLOCKS(count), AMCT_CUDA_NUM_THREADS>>>(
            count, bottomData, topData, clipMaxGpu, clipMinGpu, fixedMin, quantBitNum);
        return AmctCommon::SUCCESS;
}

int ActQuantForwardGpu(const int count, const float* bottomData, float* topData, float* clipMaxGpu,
    float* clipMinGpu, bool fixedMin, int quantBitNum)
{
        return ActQuantForwardGpuInternel(count, bottomData, topData, clipMaxGpu,
                                          clipMinGpu, fixedMin, quantBitNum);
}

int ActQuantForwardGpu(const int count, const double* bottomData, double* topData, double* clipMaxGpu,
    double* clipMinGpu, bool fixedMin, int quantBitNum)
{
    return ActQuantForwardGpuInternel(count, bottomData, topData, clipMaxGpu,
                                      clipMinGpu, fixedMin, quantBitNum);
}

template <typename Dtype>
int UlqDiffGpuInternel(const int count, const Dtype* bottomData, Dtype* bottomDiff, const Dtype* topDiff,
    const Dtype* clipMaxGpu, const Dtype* clipMinGpu, Dtype& diffMaxCpuRef, Dtype& diffMinCpuRef, int quantBitNum)
{
    QuantBackwardMax<Dtype><<<AMCT_GET_BLOCKS(count), AMCT_CUDA_NUM_THREADS>>>(
        count, bottomData, topDiff, bottomDiff, clipMaxGpu, clipMinGpu, quantBitNum);
    diffMaxCpuRef = MultiSum(count, bottomDiff);
    QuantBackwardMin<Dtype><<<AMCT_GET_BLOCKS(count), AMCT_CUDA_NUM_THREADS>>>(
        count, bottomData, topDiff, bottomDiff, clipMaxGpu, clipMinGpu, quantBitNum);
    diffMinCpuRef = MultiSum(count, bottomDiff);

     return AmctCommon::SUCCESS;
}

int UlqDiffGpu(const int count, const double* bottomData, double* bottomDiff, const double* topDiff,
    const double* clipMaxGpu, const double* clipMinGpu, double& diffMaxCpuRef, double& diffMinCpuRef,
    int quantBitNum)
{
    return UlqDiffGpuInternel(count, bottomData, bottomDiff, topDiff, clipMaxGpu,
                              clipMinGpu, diffMaxCpuRef, diffMinCpuRef, quantBitNum);
}

int UlqDiffGpu(const int count, const float* bottomData, float* bottomDiff, const float* topDiff,
    const float* clipMaxGpu, const float* clipMinGpu, float& diffMaxCpuRef, float& diffMinCpuRef, int quantBitNum)
{
    return UlqDiffGpuInternel(count, bottomData, bottomDiff, topDiff, clipMaxGpu,
                              clipMinGpu, diffMaxCpuRef, diffMinCpuRef, quantBitNum);
}
}