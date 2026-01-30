/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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
 * @brief ARQ algorithm CUDA C++ implementation for GPU.
 *
 * @file dmq_balance.cu
 *
 * @version 1.0
 */
#if GOOGLE_CUDA

#include "dmq_balance.h"

#include <cfloat>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "cast_util.h"

using namespace util;

namespace AmctCommon {
inline cudaError_t CheckCudaError(cudaError_t error, const char *file, const int line)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
    }
    return error;
}

__global__ void GetScalePerChannelCudaKernel(
    const float* data, unsigned int channelNum, unsigned int channelSize, float* scale)
{
    CUDA_KERNEL_LOOP(i, channelNum) {
        for (unsigned int idx = i * channelSize; idx < (i + 1) * channelSize; ++idx) {
            scale[i] = scale[i] > fabs(data[idx]) ? scale[i] : fabs(data[idx]);
        }
    }
}

__global__ void GetBalanceFactorCudaKernel(float* actScale,
                                           float* wgtScale,
                                           unsigned int channelNum,
                                           float migrationStrength,
                                           float* balanceFactor)
{
    CUDA_KERNEL_LOOP(i, channelNum) {
        if ((actScale[i] < FLT_EPSILON) || (wgtScale[i] < FLT_EPSILON)) {
            balanceFactor[i] = 0;
            continue;
        }
        balanceFactor[i] = pow(actScale[i], migrationStrength) / pow(wgtScale[i], 1 - migrationStrength);
    }
}

__global__ void ProcessZeroScaleCudaKernel(float* actScale,
                                           float* wgtScale,
                                           unsigned int channelNum,
                                           float* balanceFactor)
{
    float actOutlier = 0;
    float wgtOutlier = 0;
    for (unsigned int i = 0; i < channelNum; i++) {
        if (balanceFactor[i] >= FLT_EPSILON) {
            actOutlier =
                actOutlier > fabs(actScale[i] / balanceFactor[i]) ? actOutlier : fabs(actScale[i] / balanceFactor[i]);
            wgtOutlier =
                wgtOutlier > fabs(wgtScale[i] * balanceFactor[i]) ? wgtOutlier : fabs(wgtScale[i] * balanceFactor[i]);
        }
    }
    if (actOutlier < FLT_EPSILON || wgtOutlier < FLT_EPSILON) {
        for (unsigned int i = 0; i < channelNum; i++) {
            balanceFactor[i] = 1;
        }
        return;
    }
    CUDA_KERNEL_LOOP(i, channelNum) {
        if (actScale[i] < FLT_EPSILON) {
            if (wgtScale[i] < FLT_EPSILON) {
                balanceFactor[i] = 1;
            } else {
                balanceFactor[i] = wgtOutlier / wgtScale[i];
            }
        } else {
            if (wgtScale[i] < FLT_EPSILON) {
                balanceFactor[i] = actScale[i] / actOutlier;
            }
        }
    }
}

Status DMQBalanceGpu(const FloatData &act,
                     const FloatData &wts,
                     float migrationStrength,
                     unsigned int channelNum,
                     float* balanceFactor)
{
    auto checkRes = CheckDMQBParam(act, wts, migrationStrength, channelNum, balanceFactor);
    if (checkRes != AmctCommon::SUCCESS) {
        return checkRes;
    }

    // get activation scale
    thrust::device_vector<float> actScale(channelNum, 0);
    float* actScalePtr = thrust::raw_pointer_cast(actScale.data());
    unsigned int actChannelSize = act.length / channelNum;
    GetScalePerChannelCudaKernel<<<channelNum, 1>>>(act.data, channelNum, actChannelSize, actScalePtr);
    auto actMinMax = thrust::minmax_element(actScale.begin(), actScale.end());
    if (*actMinMax.second < FLT_EPSILON) {
        thrust::device_ptr<float> balanceFactorDev(balanceFactor);
        thrust::fill(balanceFactorDev, balanceFactorDev + channelNum, 1);
        return AmctCommon::SUCCESS;
    }
    // get weight scale
    thrust::device_vector<float> wgtScale(channelNum, 0);
    float* wgtScalePtr = thrust::raw_pointer_cast(wgtScale.data());
    unsigned int wgtChannelSize = wts.length / channelNum;
    GetScalePerChannelCudaKernel<<<channelNum, 1>>>(wts.data, channelNum, wgtChannelSize, wgtScalePtr);
    auto wgtMinMax = thrust::minmax_element(wgtScale.begin(), wgtScale.end());
    if (*wgtMinMax.second < FLT_EPSILON) {
        thrust::device_ptr<float> balanceFactorDev(balanceFactor);
        thrust::fill(balanceFactorDev, balanceFactorDev + channelNum, 1);
        return AmctCommon::SUCCESS;
    }
    // compute tensor balance factor
    GetBalanceFactorCudaKernel<<<channelNum, 1>>>(
        actScalePtr, wgtScalePtr, channelNum, migrationStrength, balanceFactor);
    // if zero value in activation scale or weight scale, assign balanceFactor according to outliers.
    // when actScale[i] is zero, wgtScale[i] * balanceFactor[i] = wgtOutlier
    // when wgtScale[i] is zero, actScale[i] / balanceFactor[i] = actOutlier
    if (*actMinMax.first < FLT_EPSILON || *wgtMinMax.first < FLT_EPSILON) {
        ProcessZeroScaleCudaKernel<<<channelNum, 1>>>(actScalePtr, wgtScalePtr, channelNum, balanceFactor);
    }
    return AmctCommon::SUCCESS;
}

Status DMQBalanceGpuMemCopy(InputDataParam inputAct,
                            InputDataParam inputWts,
                            float migrationStrength,
                            unsigned int channelNum,
                            float* balanceFactor)
{
    FloatData act;
    thrust::device_vector<float> actData(inputAct.length, 0);
    float* actDataPtr = thrust::raw_pointer_cast(actData.data());
    if (inputAct.inType == 10) { // data type is float16
        DataCastToFloat32Functor<GPUDevice, uint16_t>()(reinterpret_cast<uint16_t*>(inputAct.in), actDataPtr,
            inputAct.length);
        act = {static_cast<unsigned int>(inputAct.length), actDataPtr};
    } else if (inputAct.inType == 1) { // data type is float32
        act = {static_cast<unsigned int>(inputAct.length), reinterpret_cast<float*>(inputAct.in)};
    } else {
        return AmctCommon::NOT_SUPPORT_ERROR;
    }

    FloatData wts;
    thrust::device_vector<float> wtsData(inputWts.length, 0);
    float* wtsDataPtr = thrust::raw_pointer_cast(wtsData.data());
    if (inputWts.inType == 10) { // data type is float16
        DataCastToFloat32Functor<GPUDevice, uint16_t>()(reinterpret_cast<uint16_t*>(inputWts.in), wtsDataPtr,
            inputWts.length);
        wts = {static_cast<unsigned int>(inputWts.length), wtsDataPtr};
    } else if (inputWts.inType == 1) { // data type is float32
        wts = {static_cast<unsigned int>(inputWts.length), reinterpret_cast<float*>(inputWts.in)};
    } else {
        return AmctCommon::NOT_SUPPORT_ERROR;
    }

    float* devBalanceFactor = nullptr;
    if (CheckCudaError(cudaMalloc((void**)&devBalanceFactor, sizeof(float) * channelNum), __FILE__, __LINE__)) {
        return AmctCommon::CUDA_ERROR;
    }

    Status ret = DMQBalanceGpu(act, wts, migrationStrength, channelNum, devBalanceFactor);
    if (ret != AmctCommon::SUCCESS) {
        cudaFree(devBalanceFactor);
        return ret;
    }

    if (CheckCudaError(cudaMemcpy(balanceFactor, devBalanceFactor, sizeof(float) * channelNum, cudaMemcpyDeviceToHost),
        __FILE__, __LINE__)) {
        cudaFree(devBalanceFactor);
        return AmctCommon::CUDA_ERROR;
    }
    cudaFree(devBalanceFactor);

    return AmctCommon::SUCCESS;
}
}

#endif // GOOGLE_CUDA