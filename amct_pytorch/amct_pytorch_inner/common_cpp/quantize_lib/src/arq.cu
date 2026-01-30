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
 * @brief ARQ algorithm CUDA C++ implementation for GPU.
 *
 * @file arq.cu
 *
 * @version 1.0
 */

#include "arq.h"

#include <cfloat>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace util;

// add for cuda
namespace AmctCommon {
#define CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__)

inline cudaError_t checkCudaError(cudaError_t error, const char *file, const int line)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
    }
    return error;
}

template <class T>
void FreeGpuMemory(T* devData, char* devInt8Data, float* scaleGpu, int* offsetGpu)
{
    if (devData != nullptr) {
        cudaFree(devData);
        devData = nullptr;
    }
    if (devInt8Data != nullptr) {
        cudaFree(devInt8Data);
        devInt8Data = nullptr;
    }
    if (scaleGpu != nullptr) {
        cudaFree(scaleGpu);
        scaleGpu = nullptr;
    }
    if (offsetGpu != nullptr) {
        cudaFree(offsetGpu);
        offsetGpu = nullptr;
    }
}

template <class T>
__device__ void CalculateScaleWithOffset(T maxVal, T minVal, float& scale, int &offset, const int& N)
{
    // include zero
    maxVal = maxVal < 0.0 ? 0.0 : maxVal;
    minVal = minVal > 0.0 ? 0.0 : minVal;
    // get scale
    if (fabs(maxVal - minVal) < FLT_EPSILON) {
        scale = 1.0;
    } else {
        scale = (maxVal - minVal) / (pow(2, N) - 1 + EPS);
    }
    // get offset
    offset = ::round(-minVal / scale) - pow(2, N - 1);
}

template <class T>
__device__ void ArqFakeQuant(T& value, const float scale, const int offset, const int N)
{
        float min_limit = -(pow(2, N - 1));
        float max_limit = pow(2, N - 1) - 1;
        int temp_data = ::round(value / scale) + offset;
        if (temp_data > max_limit) {
            temp_data = max_limit;
        }
        if (temp_data < min_limit) {
            temp_data = min_limit;
        }
        value = (temp_data - offset) * scale;
}

template <class T>
__device__ void ArqRealQuant(T& value, float& scale, int &offset, const int N, char& data)
{
        float max_limit = pow(2, N - 1) - 1;
        float min_limit = -(pow(2, N - 1));
        int temp_data = ::round(value / scale) + offset;
        // clip into int8
        if (temp_data > max_limit) {
            temp_data = max_limit;
        }
        if (temp_data < min_limit) {
            temp_data = min_limit;
        }
        data = temp_data;
}

template <class T>
__global__ void WtsCalibrationKernelPerChannel(const T* dIn, const int size, float* scale, int* offset, const int N,
    int kernelSize, WeightFormat format)
{
    int channelId = blockIdx.x;
    if (format == CO_CI_KH_KW) {
        offset[channelId] = 0;
        scale[channelId] = 0;
        for (int i = channelId * kernelSize; i < (channelId + 1) * kernelSize; i += 1) {
            if (abs(scale[channelId]) < abs(dIn[i]))
            {
                scale[channelId] = dIn[i];
            }
        }
    } else {
        // KH_KW_CI_CO
        offset[channelId] = 0;
        scale[channelId] = 0;
        for (int i = channelId; i < size; i += gridDim.x) {
            if (abs(scale[channelId]) < abs(dIn[i]))
                scale[channelId] = dIn[i];
            }
    }
    if (scale[channelId] < 0) {
        scale[channelId] = abs(scale[channelId]) / (pow(2, N - 1) + EPS);
    } else {
        scale[channelId] = abs(scale[channelId]) / (pow(2, N - 1) - 1 + EPS);
    }
    if (scale[channelId] < EPSILON) {
        scale[channelId] = 1.0;
    }
}


template <class T>
__global__ void WtsFakeQuantKernelPerChannel(T* dIn, float* scale, int* offset,
    const int N, int kernelSize, WeightFormat format)
{
    int offsetData = blockIdx.x;
    // fake quant the weight data
    if (format == CO_CI_KH_KW) {
        for (int i = offsetData * kernelSize; i < (offsetData + 1) * kernelSize; i += 1) {
            ArqFakeQuant(dIn[i], scale[offsetData], offset[offsetData], N);
        }
    }

}

template <class T>
__global__ void CalculateScaleOffset(const T* min, const T* max, float *scale, int *offset, const int N,
    bool withOffset)
{
    T maxVal = max[0];
    T minVal = min[0];
    if (not withOffset) {
        // without offset
        if (maxVal > abs(minVal)) {
            scale[0] = abs(maxVal) / (pow(2, N - 1) - 1 + EPS);
        } else {
            scale[0] = abs(minVal) / (pow(2, N - 1) + EPS);
        }
        offset[0] = 0;
    } else {
        // with offset
        CalculateScaleWithOffset(maxVal, minVal, scale[0], offset[0], N);
    }
}


template <class T>
__global__ void WtsFakeQuantKernelPerTensor(const int dataSize, T* dIn, const int N, float* scale, int* offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // fake quant
    if (index < dataSize) {
        ArqFakeQuant(dIn[index], scale[0], offset[0], N);
    }
}


template <class T>
__global__ void WeightQuantRealCudaKernel(T* dIn, char* devInt8Data, float* scale, int* offset,
    const int N, int kernelSize, int length)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        int scaleIndex = index / kernelSize;
        ArqRealQuant(dIn[index], scale[scaleIndex], offset[scaleIndex], N, devInt8Data[index]);
    }
}


template <class T>
Status WtsArqCalibrationCudaKernel(const T* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, WeightFormat format)
{
    unsigned int kernelSize = length / scale.length;
    if (scale.length == 1) {
        thrust::pair<const T*, const T*> result = thrust::minmax_element(thrust::device, data, data + kernelSize);
        CalculateScaleOffset<<<1, 16>>>(
            result.first, result.second, scale.data, offset.data, arqParam.numBits, arqParam.withOffset);
    } else {
        WtsCalibrationKernelPerChannel<<<scale.length, 1>>>(
            data, length, scale.data, offset.data, arqParam.numBits, kernelSize, format);
    }
    // process scale
    std::unique_ptr<float[]> scaleHostUptr(new(std::nothrow) float[scale.length]);
    NULLPTR_CHECK(scaleHostUptr);
    float* scaleHost = scaleHostUptr.get();
    CHECK_ERROR(cudaMemcpy(scaleHost, scale.data, scale.length * sizeof(float), cudaMemcpyDeviceToHost));
    int ret;
    for (unsigned int index = 0; index < scale.length; index++) {
        ret = util::ProcessScale(scaleHost[index]);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("WtsArqCalibrationCudaKernel scale is illegal.\n");
            return ret;
        }
    }
    CHECK_ERROR(cudaMemcpy(scale.data, scaleHost, scale.length * sizeof(float), cudaMemcpyHostToDevice));
    return AmctCommon::SUCCESS;
}


template <class T>
Status ArqQuantKernelGPU(T* data, unsigned int length, ArqParam arqParam, FloatData scale, IntData offset)
{
    Status ret = CheckArqQuantParams(data, length, arqParam, {scale, offset});
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("CheckArqQuantParams failed.");
        return ret;
    }

    double* devData = nullptr;
    if (CHECK_ERROR(cudaMalloc((void**)&devData, sizeof(T) * length))) {
        return AmctCommon::CUDA_ERROR;
    }

    if (CHECK_ERROR(cudaMemcpy(devData, data, length * sizeof(T), cudaMemcpyHostToDevice))) {
        FreeGpuMemory(devData, nullptr, nullptr, nullptr);
        return AmctCommon::CUDA_ERROR;
    }

    float* scaleGpu = nullptr;
    if ( CHECK_ERROR(cudaMalloc((void**)&scaleGpu, scale.length * sizeof(float)))) {
        FreeGpuMemory(devData, nullptr, nullptr, nullptr);
        return AmctCommon::CUDA_ERROR;
    }

    int* offsetGpu = nullptr;
    if ( CHECK_ERROR(cudaMalloc((void**)&offsetGpu, offset.length * sizeof(int)))) {
        FreeGpuMemory(devData, nullptr, scaleGpu, nullptr);
        return AmctCommon::CUDA_ERROR;
    }

    unsigned int kernelSize = length / scale.length;
    if (scale.length == 1) {
        thrust::pair<double*, double*> result = thrust::minmax_element(thrust::device, devData, devData + kernelSize);
        CalculateScaleOffset<<<1, 16>>>(result.first, result.second, scaleGpu, offsetGpu,
                                        arqParam.numBits, arqParam.withOffset);
        int blockNum = (length / AMCT_CUDA_NUM_THREADS) + ((length % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
        WtsFakeQuantKernelPerTensor<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(length, devData, arqParam.numBits, scaleGpu, offsetGpu);
    } else {
        WtsCalibrationKernelPerChannel<<<scale.length, 1>>>(
            devData, length, scaleGpu, offsetGpu, arqParam.numBits, kernelSize, CO_CI_KH_KW);
        WtsFakeQuantKernelPerChannel<<<scale.length, 1>>>(
            devData, scaleGpu, offsetGpu, arqParam.numBits, kernelSize, CO_CI_KH_KW);
    }

    if (CHECK_ERROR(cudaMemcpy(scale.data, scaleGpu, scale.length * sizeof(float), cudaMemcpyDeviceToHost))) {
        FreeGpuMemory(devData, nullptr, scaleGpu, offsetGpu);
        return AmctCommon::CUDA_ERROR;
    }

    if (CHECK_ERROR(cudaMemcpy(offset.data, offsetGpu, offset.length * sizeof(int), cudaMemcpyDeviceToHost))) {
        FreeGpuMemory(devData, nullptr, scaleGpu, offsetGpu);
        return AmctCommon::CUDA_ERROR;
    }

    if (CHECK_ERROR(cudaMemcpy(data, devData, length * sizeof(T), cudaMemcpyDeviceToHost))) {
        FreeGpuMemory(devData, nullptr, scaleGpu, offsetGpu);
        return AmctCommon::CUDA_ERROR;
    }

    FreeGpuMemory(devData, nullptr, scaleGpu, offsetGpu);

    for (uint i = 0; i < scale.length; i++){
        ret = util::ProcessScale(scale.data[i]);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("ArqQuantGPU scale is illegal.\n");
            return ret;
        }
    }

    return AmctCommon::SUCCESS;
}

template <class T>
Status ArqQuantRealKernelGPU(T* data, unsigned int length, ArqParam arqParam,
    FloatData scale, IntData offset, char* int8Data)
{
    Status ret = CheckArqQuantParams(data, length, arqParam, {scale, offset});
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("CheckArqQuantParams failed.");
        return ret;
    }

    for (uint i = 0; i < scale.length; i++){
        ret = util::ProcessScale(scale.data[i]);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("ArqQuantRealGPU scale is illegal\n");
            return ret;
        }
    }

    T* devData = nullptr;
    if (CHECK_ERROR(cudaMalloc((void**)&devData, sizeof(T) * length))) {
        return AmctCommon::CUDA_ERROR;
    }

    if (CHECK_ERROR(cudaMemcpy(devData, data, length * sizeof(T), cudaMemcpyHostToDevice))) {
        FreeGpuMemory(devData, nullptr, nullptr, nullptr);
        return AmctCommon::CUDA_ERROR;
    }

    char* devInt8Data = nullptr;
    if (CHECK_ERROR(cudaMalloc((void**)&devInt8Data, sizeof(char) * (length)))) {
        FreeGpuMemory(devData, nullptr, nullptr, nullptr);
        return AmctCommon::CUDA_ERROR;
    }

    float* scaleGpu = nullptr;
    if (CHECK_ERROR(cudaMalloc((void**)&scaleGpu, scale.length * sizeof(float)))) {
        FreeGpuMemory(devData, devInt8Data, nullptr, nullptr);
        return AmctCommon::CUDA_ERROR;
    }

    if (CHECK_ERROR(cudaMemcpy(scaleGpu, scale.data, scale.length * sizeof(float), cudaMemcpyHostToDevice))) {
        FreeGpuMemory(devData, devInt8Data, scaleGpu, nullptr);
        return AmctCommon::CUDA_ERROR;
    }

    int* offsetGpu = nullptr;
    if (CHECK_ERROR(cudaMalloc((void**)&offsetGpu, offset.length * sizeof(int)))) {
        FreeGpuMemory(devData, devInt8Data, scaleGpu, nullptr);
        return AmctCommon::CUDA_ERROR;
    }

    if (CHECK_ERROR(cudaMemcpy(offsetGpu, offset.data, offset.length * sizeof(int), cudaMemcpyHostToDevice))) {
        FreeGpuMemory(devData, devInt8Data, scaleGpu, offsetGpu);
        return AmctCommon::CUDA_ERROR;
    }

    unsigned int kernelSize = length / scale.length;
    int blockNum = (length / AMCT_CUDA_NUM_THREADS) + ((length % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
    WeightQuantRealCudaKernel<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(devData, devInt8Data, scaleGpu, offsetGpu,
                                                               arqParam.numBits, kernelSize, length);

    if (CHECK_ERROR(cudaMemcpy(data, devData, length * sizeof(T), cudaMemcpyDeviceToHost))) {
        FreeGpuMemory(devData, devInt8Data, scaleGpu, offsetGpu);
        return AmctCommon::CUDA_ERROR;
    }

    if (CHECK_ERROR(cudaMemcpy(int8Data, devInt8Data, length * sizeof(char), cudaMemcpyDeviceToHost))) {
        FreeGpuMemory(devData, devInt8Data, scaleGpu, offsetGpu);
        return AmctCommon::CUDA_ERROR;
    }

    FreeGpuMemory(devData, devInt8Data, scaleGpu, offsetGpu);
    return AmctCommon::SUCCESS;
}

template <typename Dtype>
__global__ void QatWtsFakeQuantKernelCoCiKhKw(Dtype* in, const int quantBits, int kernelSize)
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
         scale = abs(scale) / (pow(2, quantBits - 1)) + 1e-6;
     } else {
         scale = abs(scale) / (pow(2, quantBits - 1) - 1) + 1e-6;
     }
     for (int i = offset * kernelSize; i < (offset + 1) * kernelSize; i += 1) {
        in[i] = ::round(in[i] / scale) * scale;
     }
 }

 template <typename T>
__global__ void QatWtsFakeQuantKernelKhKwCiCo(T* data, const unsigned int length, float* scaleValue,
    int* offsetValue, const int numBits)
{
    const int baseNum = 2;
    const int minLimit = -static_cast<int>(pow(baseNum, numBits - 1));
    const int maxLimit = static_cast<int>(pow(baseNum, numBits - 1) - 1);

    int index = blockIdx.x;
    for (int i = index; i < length; i += gridDim.x) {
        int dataTmp = round(data[i] / scaleValue[index]) + offsetValue[index];
        dataTmp = dataTmp < minLimit ? minLimit : dataTmp;
        dataTmp = dataTmp > maxLimit ? maxLimit : dataTmp;
        data[i] = (dataTmp - offsetValue[index]) * scaleValue[index];
    }
}

template <typename Dtype>
__global__ void QatWtsFakeQuantCudaKernelPerTensor(const int n, Dtype* in, const int quantBits, Dtype* min, Dtype* max)
 {
     Dtype scale = 0;
     if (abs(max[0]) > abs(min[0])) {
         scale = abs(max[0]) / (pow(2, quantBits - 1) - 1) + 1e-6;
     } else {
         scale = abs(min[0]) / (pow(2, quantBits - 1)) + 1e-6;
     }
     CUDA_KERNEL_LOOP(index, n) {
        in[index] = ::round(in[index] / scale) * scale;
    }
}

template <class Dtype>
Status ArqQuantRetrainGPUInternel(Dtype* devData, unsigned int length, ArqParam arqParam,
    FloatData scale, IntData offset, WeightFormat format)
{
    unsigned int kernelSize = length / scale.length;
    if (scale.length == 1) {
        thrust::pair<Dtype*, Dtype*> result = thrust::minmax_element(
            thrust::device, devData, devData + kernelSize);
        int blockNum = (length / AMCT_CUDA_NUM_THREADS) + ((length % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
        QatWtsFakeQuantCudaKernelPerTensor<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(length, devData, arqParam.numBits,
                                                                        result.first, result.second);
    } else {
        if (format == CO_CI_KH_KW) {
            QatWtsFakeQuantKernelCoCiKhKw<<<scale.length, 1>>>(devData, arqParam.numBits, kernelSize);
        } else {
            QatWtsFakeQuantKernelKhKwCiCo<<<scale.length, 1>>>(devData, length, scale.data, offset.data,
                arqParam.numBits);
        }
    }
    return AmctCommon::SUCCESS;
}

// C++ API
template <class T>
int ArqQuantGPU(T* data, unsigned int length, ArqParam arqParam, FloatData scale, IntData offset)
{
    return ArqQuantKernelGPU(data, length, arqParam, scale, offset);
}

template <class T>
int ArqQuantRetrainGPU(T* data, unsigned int length, ArqParam arqParam, FloatData scale, IntData offset)
{
    return ArqQuantRetrainGPUInternel(data, length, arqParam, scale, offset, CO_CI_KH_KW);
}

template <class T>
int ArqQuantRealGPU(T* data, unsigned int length, ArqParam arqParam, FloatData scale, IntData offset, char* int8Data)
{
    return ArqQuantRealKernelGPU(data, length, arqParam, scale, offset, int8Data);
}

template int ArqQuantRetrainGPU(float* data, unsigned int length, ArqParam arqParam, FloatData scale, IntData offset);

template int ArqQuantRetrainGPU(double* data, unsigned int length, ArqParam arqParam, FloatData scale, IntData offset);

template int ArqQuantGPU(float* data, unsigned int length, ArqParam arqParam, FloatData scale, IntData offset);

template int ArqQuantGPU(double* data, unsigned int length, ArqParam arqParam, FloatData scale, IntData offset);

template int ArqQuantRealGPU(float* data, unsigned int length, ArqParam arqParam,
    FloatData scale, IntData offset, char* int8Data);

template int ArqQuantRealGPU(double* data, unsigned int length, ArqParam arqParam,
    FloatData scale, IntData offset, char* int8Data);

template int WtsArqCalibrationCudaKernel(const float* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, WeightFormat format);

template int WtsArqCalibrationCudaKernel(const double* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, WeightFormat format);

template Status ArqQuantRetrainGPUInternel(float* devData, unsigned int length,
    ArqParam arqParam, FloatData scale, IntData offset, WeightFormat format);
}

// Python API
int ArqQuantDoublePythonGPU(double* data, unsigned int length, AmctCommon::ArqParam arqParam,
    FloatData scale, IntData offset)
{
    return AmctCommon::ArqQuantKernelGPU(data, length, arqParam, scale, offset);
}

int QuantRealDoublePythonGPU(double* data, unsigned int length, FloatData scale, IntData offset,
    unsigned int numBits, char* int8Data)
{
    AmctCommon::ArqParam arqParam {
        numBits: numBits,
        channelWise: (scale.length > 1)? true : false,
        withOffset: false,
    };

    return AmctCommon::ArqQuantRealKernelGPU(data, length, arqParam, scale, offset, int8Data);
}
