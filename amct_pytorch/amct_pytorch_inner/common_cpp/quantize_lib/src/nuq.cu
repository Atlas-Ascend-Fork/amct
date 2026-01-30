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
 * @brief NUQ algorithm CUDA C++ implementation for GPU.
 *
 * @file nuq.cu
 *
 * @version 1.0
 */

#include "nuq.h"
#include <cfloat>
#include <cmath>
#include <ctime>
#include <numeric>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

using namespace util;

namespace AmctCommon {
#define CHECK_ERROR(ret)                                   \
{                                                          \
    if (ret != AmctCommon::SUCCESS) {                                  \
        LOG_ERROR("Init failed!\n");                       \
        return ret;                                        \
    }                                                      \
}                                                          \

#define CUDA_CHECK_ERROR(error)                            \
{                                                          \
    int ret = checkCudaError(error, __FILE__, __LINE__);\
    if (ret != AmctCommon::SUCCESS) {                                  \
        return ret;                                        \
    }                                                      \
}                                                          \

inline int checkCudaError(cudaError_t error, const char *file, const int line)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
        return AmctCommon::CUDA_ASYNC_ERROR;
    }
    return AmctCommon::SUCCESS;
}

template <typename T>
class NuqSharedMem {
public:
    T* getSharedPointer()
    {
        return NULL;
    };
};

template <>
class NuqSharedMem <int> {
public:
    __device__ int* getSharedPointer()
    {
        extern __shared__ int s_int[];
        return s_int;
    }
};

template <>
class NuqSharedMem <float> {
public:
    __device__ float* getSharedPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
class NuqSharedMem <double> {
public:
    __device__ double* getSharedPointer() {
        extern __shared__ double s_double[];
        return s_double;
    }
};

template <typename T>
__global__ void ReduceSum(const int count, T* dataIn)
{
    NuqSharedMem<T> shared;
    T* shareSpace = shared.getSharedPointer();

    T transferVar = 0;
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

template <typename T>
__global__ void ReduceSumIndex(const int count, T* dataIn, T* dataOut, unsigned int* indexData, unsigned int indexId)
{
    NuqSharedMem<T> shared;
    T* shareSpace = shared.getSharedPointer();

    T transferVar = 0;
    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < count && indexData[globalIndex] == indexId) {
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
        dataOut[blockIdx.x] = shareSpace[0];
    }
}


template <typename T>
T MultiSumG(int inputOriginDataNum, T* inputOriginData, T* tmpValueDataPtr, unsigned int* indexData,
    unsigned int indexId)
{
    ReduceSumIndex<T><<<AMCT_GET_BLOCKS(inputOriginDataNum),
                                        AMCT_CUDA_NUM_THREADS,
                                        AMCT_CUDA_NUM_THREADS * sizeof(T)>>>(
        inputOriginDataNum, inputOriginData, tmpValueDataPtr, indexData, indexId);
    int originDataNum = AMCT_GET_BLOCKS(inputOriginDataNum);
    while (originDataNum > AMCT_CUDA_NUM_THREADS) {
        ReduceSum<T><<<AMCT_GET_BLOCKS(originDataNum),
                                        AMCT_CUDA_NUM_THREADS,
                                        AMCT_CUDA_NUM_THREADS * sizeof(T)>>>(originDataNum, tmpValueDataPtr);

        originDataNum = AMCT_GET_BLOCKS(originDataNum);
    }

    ReduceSum<T><<<1, AMCT_CUDA_NUM_THREADS, AMCT_CUDA_NUM_THREADS * sizeof(T)>>>(originDataNum, tmpValueDataPtr);

    T sumDataOut = 0;
    cudaMemcpy(&sumDataOut, tmpValueDataPtr, sizeof(T) * (1), cudaMemcpyDeviceToHost);

    return sumDataOut;
}


template <typename T>
__global__ void MultiplyKernel(int dataLength, T* valueData, T* weightData, T* weightDataPtr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dataLength) {
        weightDataPtr[tid] = valueData[tid] * weightData[tid];
    }
}

template <typename T>
__global__ void ClipAndQuantCuda(T* data, bool withOffset, int perChannelLength, float currScale, int currOffset,
    unsigned int baseOffset, int minLimit, int maxLimit)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < perChannelLength) {
        data[baseOffset + tid] = data[baseOffset + tid] / currScale + currOffset;
        if (data[baseOffset + tid] < minLimit) {
            data[baseOffset + tid] = minLimit;
        }
        if (data[baseOffset + tid] > maxLimit) {
            data[baseOffset + tid] = maxLimit;
        }
    }
}

template <typename T>
int ClipAndQuantG(thrust::device_ptr<T>& data, bool withOffset, float* scale, int* offset, int scaleLength,
    const unsigned int perChannelLength)
{
    int minLimit, maxLimit;
    unsigned int baseOffset;

    if (withOffset) {
        minLimit = 0;
        maxLimit = pow(BINARY, NUM_BITS_QUANT) - 1;
    } else {
        minLimit = -pow(BINARY, NUM_BITS_QUANT - 1);
        maxLimit = pow(BINARY, NUM_BITS_QUANT - 1) - 1;
    }

    T* dataPtr = thrust::raw_pointer_cast(data);
    int blockNum = (perChannelLength / AMCT_CUDA_NUM_THREADS) + ((perChannelLength % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
    for (unsigned int channelIndex = 0; channelIndex < scaleLength; channelIndex++) {
        baseOffset = perChannelLength * channelIndex;
        float currScale = scale[channelIndex];
        int currOffset = offset[channelIndex];
        ClipAndQuantCuda<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(dataPtr, withOffset, perChannelLength,
                                                              currScale, currOffset,
                                                              baseOffset, minLimit, maxLimit);
        cudaError_t errSync = cudaGetLastError();
        CUDA_CHECK_ERROR(errSync);
        cudaError_t errAsync = cudaDeviceSynchronize();
        CUDA_CHECK_ERROR(errAsync);
    }
    return AmctCommon::SUCCESS;
}

template <typename T>
T MedianG(std::vector<T> data)
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

template <typename T>
__global__ void CalDistanceCuda(T* dataPtr, T* distance, T* candidateCenter, T* canidateDistance,
    int snCandidate, int dataLength)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dataLength) {
        canidateDistance[tid] = (distance[tid] < powf(dataPtr[tid] - candidateCenter[snCandidate], EVEN)) ?
                    distance[tid] : powf(dataPtr[tid] - candidateCenter[snCandidate], EVEN);
    }
}

template <typename T>
__global__ void PowDistanceCuda(T* dataPtr, T* distance, T hclusterCenterValue, int dataLength)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dataLength) {
        distance[tid] = powf(dataPtr[tid] - hclusterCenterValue, EVEN);
    }
}

template <typename T>
__global__ void PowDistanceCuda2(T* dataPtr, T* distancePtr, T* clusterCenterPtr, int snCenter, int dataLength)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dataLength) {
        distancePtr[tid] = (distancePtr[tid] < powf(dataPtr[tid] - clusterCenterPtr[snCenter], EVEN)) ?
            distancePtr[tid] : powf(dataPtr[tid] - clusterCenterPtr[snCenter], EVEN);
    }
}

template <typename T>
__global__ void BreakCuda(T* cumulativeDistancePtr, T* dataPtr, int* canData,
    T equidistant, int snCandidate, int dataLength)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dataLength) {
        if (cumulativeDistancePtr[tid] > equidistant * (snCandidate + 1)) {
            canData[tid] = tid;
        }
    }
}

template <typename T>
__global__ void scan(T* dInData, T* dOutData, int n)
{
    NuqSharedMem<T> shared;
    T* temp = shared.getSharedPointer();
    int tid = threadIdx.x;
    int pout = 0, pin = 1;
    if (tid == 0) {
        temp[pout * n + tid] = dOutData[tid];
        temp[pin * n + tid] = dOutData[tid];
    } else {
        temp[pout * n + tid] = dInData[tid];
        temp[pin * n + tid] = dInData[tid];
    }
    __syncthreads();
    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout;
        pin = 1 - pout;
        if (tid >= offset) {
            temp[pout * n + tid] = temp[pin * n + tid - offset] + temp[pin * n + tid];
        } else {
            temp[pout * n + tid] = temp[pin * n + tid];
        }
        __syncthreads();
    }
    dOutData[tid] = temp[pout * n + tid];
}

template <typename T>
void MultiScan(T* dInData, T* dOutData, int length)
{
    int currentPosi = 0;
    while ((length - currentPosi) > AMCT_CUDA_NUM_THREADS) {
        scan<<<1, AMCT_CUDA_NUM_THREADS, (AMCT_CUDA_NUM_THREADS*2)*sizeof(T)>>>(dInData + currentPosi,
                                                                                dOutData + currentPosi,
                                                                                AMCT_CUDA_NUM_THREADS);
        currentPosi += AMCT_CUDA_NUM_THREADS - 1;
    }
    scan<<<1, (length - currentPosi), ((length - currentPosi)*2)*sizeof(T)>>>(dInData + currentPosi,
                                                                                dOutData + currentPosi,
                                                                                (length - currentPosi));
}

template <typename T>
int PowDistanceCudaFun(T* dataPtr, T* distancePtr,
    thrust::host_vector<T>& hclusterCenter, thrust::host_vector<T>& hdata)
{
    int blockNum = (hdata.size() / AMCT_CUDA_NUM_THREADS) + ((hdata.size() % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
    PowDistanceCuda<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(dataPtr, distancePtr, hclusterCenter[0], hdata.size());
    cudaError_t errSync = cudaGetLastError();
    CUDA_CHECK_ERROR(errSync);
    cudaError_t errAsync = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(errAsync);
    return AmctCommon::SUCCESS;
}

template <typename T>
int BreakCudaFun(struct ThrustParam<T>& thrustParam, T* dataPtr, T& equidistant, int snCandidate,
    thrust::host_vector<T>& hdata)
{
    T* cumulativeDistancePtr = thrustParam.dcumulativeDistancePtr;
    int* canDataPtr = thrustParam.canDataPtr;
    int blockNum = (hdata.size() / AMCT_CUDA_NUM_THREADS) + ((hdata.size() % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
    BreakCuda<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(cumulativeDistancePtr, dataPtr, canDataPtr,
                                                   equidistant, snCandidate, hdata.size());
    cudaError_t errSync = cudaGetLastError();
    CUDA_CHECK_ERROR(errSync);
    cudaError_t errAsync = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(errAsync);
    return AmctCommon::SUCCESS;
}

template <typename T>
int CalDistanCuFun(T* dataPtr, struct ThrustParam<T>& thrustParam, T* candidateCenterPtr,
    int snCandidate, thrust::host_vector<T>& hdata)
{
    T* candiDistPtr = thrustParam.dcandiDistPtr;
    T* distancePtr = thrustParam.ddistancePtr;
    int blockNum = (hdata.size() / AMCT_CUDA_NUM_THREADS) + ((hdata.size() % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
    CalDistanceCuda<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(dataPtr, distancePtr, candidateCenterPtr,
                                                         candiDistPtr, snCandidate, hdata.size());
    cudaError_t errSync = cudaGetLastError();
    CUDA_CHECK_ERROR(errSync);
    cudaError_t errAsync = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(errAsync);
    return AmctCommon::SUCCESS;
}

template <typename T>
int PowDistanceCuda2Fun(struct ThrustParam<T>& thrustParam, int snCenter, thrust::host_vector<T>& hdata)
{
    T* dataPtr = thrustParam.clusterDataPtr;
    T* distancePtr = thrustParam.ddistancePtr;
    T* clusterCenterPtr = thrustParam.clusterCenterPtr;
    int blockNum = (hdata.size() / AMCT_CUDA_NUM_THREADS) + ((hdata.size() % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
    PowDistanceCuda2<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(dataPtr, distancePtr, clusterCenterPtr,
                                                               snCenter, hdata.size());
    cudaError_t errSync = cudaGetLastError();
    CUDA_CHECK_ERROR(errSync);
    cudaError_t errAsync = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(errAsync);
    return AmctCommon::SUCCESS;
}

template <typename T>
void DealCandidateCenter(int snCandidate,
    thrust::device_ptr<int>& canData, thrust::host_vector<T>& hdata,
    thrust::host_vector<T>& candidateCenter, thrust::device_vector<T>& dcandidateCenter)
{
    int tempPos = thrust::min_element(canData, canData + hdata.size()) - canData;
    if (tempPos < hdata.size()) {
        candidateCenter[snCandidate] = hdata[tempPos];
        dcandidateCenter[snCandidate] = hdata[tempPos];
    }

    dcandidateCenter = candidateCenter;
}

template <typename T>
void CalCumuDistance(thrust::host_vector<T>& cumulativeDistance, thrust::host_vector<T>& distance,
    struct ThrustParam<T>& thrustParam, thrust::host_vector<T>& hdata)
{
    // Calculate the cumulative distance distribution
    T* distancePtr = thrustParam.ddistancePtr;
    T* cumulativeDistancePtr = thrustParam.dcumulativeDistancePtr;
    cumulativeDistance[0] = distance[0];
    cudaMemcpy(cumulativeDistancePtr, cumulativeDistance.data(), sizeof(T) * hdata.size(), cudaMemcpyHostToDevice);
    MultiScan(distancePtr, cumulativeDistancePtr, hdata.size());
    cudaMemcpy(cumulativeDistance.data(), cumulativeDistancePtr, sizeof(T) * hdata.size(), cudaMemcpyDeviceToHost);
}

template <typename T>
void TakeNextCenter(thrust::host_vector<T>& canidateTotalDistance, int snCenter, thrust::host_vector<T>& hclusterCenter,
    thrust::host_vector<T>& candidateCenter, thrust::device_vector<T>& clusterCenter)
{
    // The candidate cluster center with the minimum sum of distances is taken as the next cluster center
    int minPosition = thrust::min_element(canidateTotalDistance.begin(), canidateTotalDistance.end())
                      - canidateTotalDistance.begin();
    hclusterCenter[snCenter] = candidateCenter[thrust::distance(
            canidateTotalDistance.begin(), canidateTotalDistance.begin() + minPosition)];
    clusterCenter = hclusterCenter;
}

template <typename T>
int KmeansInitG(std::vector<T>& stdData, struct ThrustParam<T>& thrustParam, thrust::device_vector<T>& clusterCenter)
{
    thrust::host_vector<T> hdata(thrustParam.dataLength);
    T* dataPtr = thrustParam.clusterDataPtr;
    cudaMemcpy(hdata.data(), dataPtr, sizeof(T) * hdata.size(), cudaMemcpyDeviceToHost);
    int numCandidate = thrustParam.numCandidate;
    clusterCenter[0] = MedianG(stdData);
    thrust::host_vector<T> hclusterCenter = clusterCenter;
    thrust::host_vector<T> distance(hdata.size());
    thrust::host_vector<T> cumulativeDistance(hdata.size());
    T equidistant;
    thrust::host_vector<T> candidateCenter(numCandidate);
    thrust::host_vector<T> candidateDistance(hdata.size());
    thrust::host_vector<T> canidateTotalDistance(numCandidate);
    thrust::device_vector<T> dcandidateCenter = candidateCenter;
    T* candiCenterPtr = thrust::raw_pointer_cast(dcandidateCenter.data());
    cudaMemcpy(thrustParam.dcandiDistPtr, candidateDistance.data(), sizeof(T) * hdata.size(), cudaMemcpyHostToDevice);
    CHECK_ERROR(PowDistanceCudaFun(dataPtr, thrustParam.ddistancePtr, hclusterCenter, hdata));
    cudaMemcpy(distance.data(), thrustParam.ddistancePtr, sizeof(T) * hdata.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(thrustParam.dcumulativeDistancePtr, thrustParam.ddistancePtr, sizeof(T) * hdata.size(),
               cudaMemcpyDeviceToDevice);
    thrust::device_ptr<int> canDataDevPtr = thrust::device_pointer_cast(thrustParam.canDataPtr);
    for (size_t snCenter = 1; snCenter < hclusterCenter.size(); snCenter++) {
        CalCumuDistance(cumulativeDistance, distance, thrustParam, hdata);
        // Five equally spaced distribution points as candidate points
        equidistant = cumulativeDistance.back() / (numCandidate + 1);
        for (int candIdx = 0; candIdx < numCandidate; candIdx++) {
            thrust::host_vector<int> canDataHost(hdata.size(), hdata.size() + 1);
            cudaMemcpy(thrustParam.canDataPtr, canDataHost.data(), sizeof(int) * canDataHost.size(),
                       cudaMemcpyHostToDevice);
            CHECK_ERROR(BreakCudaFun(thrustParam, dataPtr, equidistant, candIdx, hdata));
            DealCandidateCenter(candIdx, canDataDevPtr, hdata, candidateCenter, dcandidateCenter);
            CHECK_ERROR(CalDistanCuFun(dataPtr, thrustParam, candiCenterPtr, candIdx, hdata));
            cudaMemcpy(candidateDistance.data(), thrustParam.dcandiDistPtr, sizeof(T) * hdata.size(),
                       cudaMemcpyDeviceToHost);
            canidateTotalDistance[candIdx] = thrust::reduce(thrust::device, thrustParam.dcandiDistPtr,
                                                            thrustParam.dcandiDistPtr + hdata.size(), 0);
        }
        TakeNextCenter(canidateTotalDistance, snCenter, hclusterCenter, candidateCenter, clusterCenter);
        // Upadte distance, unless the last iteration
        if (snCenter + 1 == hclusterCenter.size()) {
            break;
        } else {
            CHECK_ERROR(PowDistanceCuda2Fun(thrustParam, snCenter, hdata));
            cudaMemcpy(distance.data(), thrustParam.ddistancePtr, sizeof(T) * hdata.size(), cudaMemcpyDeviceToHost);
        }
    }
    clusterCenter = hclusterCenter;
    return AmctCommon::SUCCESS;
}

template <typename T>
__global__ void ClassifyCuda(unsigned  int* dataClusterPtr, const T* dataPtr, const T* clusterCenterPtr,
    int dataLength, int snClusterLength)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dataLength) {
        T minValue = FLT_MAX;
        for (int snCluster = 0; snCluster < snClusterLength; snCluster++) {
            if (minValue > powf(dataPtr[tid] - clusterCenterPtr[snCluster], EVEN)) {
                minValue = powf(dataPtr[tid] - clusterCenterPtr[snCluster], EVEN);
                dataClusterPtr[tid] = snCluster;
            }
        }
    }
}

template <typename T>
int ClassifyG(struct ThrustParam<T>& thrustParam)
{
    int dataLength = thrustParam.dataLength;
    const T* dataPtr = thrustParam.clusterDataPtr;
    const T* clusterCenterPtr = thrustParam.clusterCenterPtr;
    unsigned int* dataClusterPtr = thrustParam.dataClusterPtr;

    int blockNum = (dataLength / AMCT_CUDA_NUM_THREADS) + ((dataLength % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
    ClassifyCuda<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(dataClusterPtr, dataPtr,
                                                      clusterCenterPtr, dataLength, thrustParam.numSteps);
    cudaError_t errSync = cudaGetLastError();
    CUDA_CHECK_ERROR(errSync);
    cudaError_t errAsync = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(errAsync);

    return AmctCommon::SUCCESS;
}

template <typename T>
__global__ void CalSumClusterCuda(T* dataPtr, T* weightPtr, unsigned int* dataClusterPtr,
    T* weightClusterPtr, T* sumClusterPtr, int dataSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dataSize) {
        weightClusterPtr[dataClusterPtr[tid]] += weightPtr[tid];
        sumClusterPtr[dataClusterPtr[tid]] += dataPtr[tid] * weightPtr[tid];
    }
}

template <typename T>
__global__ void DataCluCenterCuda(T* dataPtr, T* clusterCenterPtr, unsigned int* dataClusterPtr, int dataSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dataSize) {
        dataPtr[tid] = clusterCenterPtr[dataClusterPtr[tid]];
    }
}

template <typename T>
__global__ void CalSumClusterCuda(T* weightClusterPtr, unsigned int* dataClusterPtr, T* weightPtr,
    T* sumClusterPtr, T* dataPtr, int dataLength)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dataLength) {
        weightClusterPtr[dataClusterPtr[tid]] += weightPtr[tid];
        sumClusterPtr[dataClusterPtr[tid]] += dataPtr[tid] * weightPtr[tid];
    }
}

template <typename T>
int CalClusterCenter(struct ThrustParam<T>& thrustParam, thrust::device_vector<T>& clusterCenter,
    T* dataPtr, int numCluster)
{
    for (unsigned int snCluster = 0; snCluster < numCluster; snCluster++) {
        clusterCenter[snCluster] = round(clusterCenter[snCluster]);
    }
    ClassifyG(thrustParam);

    T* clusterCenterPtr = thrustParam.clusterCenterPtr;
    unsigned int* dataClusterPtr = thrustParam.dataClusterPtr;
    DataCluCenterCuda<<<AMCT_GET_BLOCKS(thrustParam.dataLength), AMCT_CUDA_NUM_THREADS>>>(dataPtr, clusterCenterPtr,
                                                                               dataClusterPtr, thrustParam.dataLength);
    cudaError_t errSync = cudaGetLastError();
    CUDA_CHECK_ERROR(errSync);
    cudaError_t errAsync = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(errAsync);

    return AmctCommon::SUCCESS;
}

template <typename T>
void calLoss(T& loss, int& numCluster,
    thrust::device_vector<T>& clusterCenter, thrust::device_vector<T>& clusterCenterLast)
{
    loss = 0;
    for (unsigned int snCluster = 0; snCluster < numCluster; snCluster++) {
        loss += fabs(clusterCenter[snCluster] - clusterCenterLast[snCluster]);
    }
}

template <typename T>
void UpdateClusterCenter(thrust::host_vector<T>& hsumCluster, thrust::host_vector<T>& hweightCluster,
    thrust::host_vector<T>& hclusterCenter, unsigned int index)
{
    if (hweightCluster[index] < EPS) {
        hweightCluster[index] = 1.0;
    }
    hclusterCenter[index] = hsumCluster[index] / hweightCluster[index];
}

// Kmeans++ Algorithm
template <typename T>
int KmeansG(std::vector<T>& stdData, struct ThrustParam<T>& thrustParam, thrust::device_ptr<T>& data,
    int numCluster, std::vector<T>& clusterToStore)
{
    int dataLength = stdData.size();
    thrust::device_vector<T> clusterCenter(numCluster);
    thrust::host_vector<T> hclusterCenter(numCluster);
    thrustParam.clusterCenterPtr = thrust::raw_pointer_cast(clusterCenter.data());
    T* dataPtr = thrust::raw_pointer_cast(data);
    CHECK_ERROR(KmeansInitG(stdData, thrustParam, clusterCenter));
    thrust::device_vector<T> clusterCenterLast(numCluster, 0);
    thrust::host_vector<T> hdata(dataLength);
    cudaMemcpy(hdata.data(), thrustParam.clusterDataPtr, sizeof(T) * dataLength, cudaMemcpyDeviceToHost);
    thrust::host_vector<T> hweight(dataLength);
    cudaMemcpy(hweight.data(), thrustParam.weightPtr, sizeof(T) * dataLength, cudaMemcpyDeviceToHost);
    T* tmpValueDataPtr = thrustParam.tmpValueDataPtr;
    T* weightDataPtr = thrustParam.weightDataPtr;
    unsigned int* dataClusterPtr = thrustParam.dataClusterPtr;
    T loss;
    T* weightPtr = thrustParam.weightPtr;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        ClassifyG(thrustParam);
        hclusterCenter = clusterCenter;
        calLoss(loss, numCluster, clusterCenter, clusterCenterLast);
        if (loss < EPS) {
            break;
        } else {
            clusterCenterLast = clusterCenter;
            thrust::host_vector<T> hweightCluster(numCluster);
            thrust::host_vector<T> hsumCluster(numCluster);
            cudaMemcpy(hweightCluster.data(), thrustParam.weightClusterPtr, sizeof(T) * numCluster,
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(hsumCluster.data(), thrustParam.sumClusterPtr, sizeof(T) * numCluster, cudaMemcpyDeviceToHost);
            MultiplyKernel<<<AMCT_GET_BLOCKS(dataLength), AMCT_CUDA_NUM_THREADS>>>(dataLength, dataPtr,
                                                                                   weightPtr, weightDataPtr);
            for (unsigned int index = 0; index < numCluster; index++) {
                hsumCluster[index] = MultiSumG(dataLength, weightDataPtr, tmpValueDataPtr, dataClusterPtr, index);
                hweightCluster[index] = MultiSumG(dataLength, weightPtr, tmpValueDataPtr, dataClusterPtr, index);
                UpdateClusterCenter(hsumCluster, hweightCluster, hclusterCenter, index);
            }
            clusterCenter = hclusterCenter;
        }
    }
    CalClusterCenter(thrustParam, clusterCenter, dataPtr, numCluster);
    thrust::copy(clusterCenter.begin(), clusterCenter.end(), clusterToStore.begin());
    return AmctCommon::SUCCESS;
}

template <typename T>
int NuqInitG(T* data, thrust::device_ptr<T>& deviceData, bool withOffset,
    float* scale, int* offset, int scaleLength, unsigned int perChannelLength)
{
    T maxValue, minValue;

    for (int channelIndex = 0; channelIndex < scaleLength; channelIndex++) {
        auto start = deviceData + channelIndex * perChannelLength;
        auto stop = start + perChannelLength;
        maxValue = *thrust::max_element(start, stop);
        minValue = *thrust::min_element(start, stop);

        if (withOffset) {
            scale[channelIndex] = (maxValue - minValue) / (pow(BINARY, NUM_BITS_QUANT) - 1) + EPS;
            offset[channelIndex] = round(-minValue / scale[channelIndex]);
        } else {
            if (fabs(maxValue) >= fabs(minValue)) {
                scale[channelIndex] = fabs(maxValue) / (pow(BINARY, NUM_BITS_QUANT - 1) - 1) + EPS;
            } else {
                scale[channelIndex] = fabs(minValue) / pow(BINARY, NUM_BITS_QUANT - 1) + EPS;
            }
            offset[channelIndex] = 0;
        }
    }

    return AmctCommon::SUCCESS;
}

template <typename T>
__global__ void WeightCuda(T* weightPtr, T* expandWeightPtr, unsigned int length, unsigned int perChannelLength)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        expandWeightPtr[tid] = weightPtr[tid / perChannelLength];
    }
}

template<typename T>
int WeightG(unsigned int length, float* scale, T* expandWeightPtr, int scaleLength, unsigned int perChannelLength)
{
    T sumWeight = 0;
    thrust::device_vector<T> weight(scale, scale + scaleLength);
    T* weightPtr = thrust::raw_pointer_cast(weight.data());
    T minWeight = *thrust::min_element(scale, scale + scaleLength);

    for (unsigned int channelIndex = 0; channelIndex < scaleLength; channelIndex++) {
        weight[channelIndex] = pow(weight[channelIndex] / minWeight, EVEN);
        sumWeight += weight[channelIndex];
    }

    for (unsigned int channelIndex = 0; channelIndex < scaleLength; channelIndex++) {
        weight[channelIndex] = weight[channelIndex] / sumWeight;
    }

    int blockNum = (length / AMCT_CUDA_NUM_THREADS) + ((length % AMCT_CUDA_NUM_THREADS) ? 1 : 0);
    WeightCuda<<<blockNum, AMCT_CUDA_NUM_THREADS>>>(weightPtr, expandWeightPtr, length, perChannelLength);
    cudaError_t errSync = cudaGetLastError();
    CUDA_CHECK_ERROR(errSync);
    cudaError_t errAsync = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(errAsync);

    return AmctCommon::SUCCESS;
}

template <typename T>
__global__ void FakeQuantCuda(T* data, int perChannelLength, float scale, int offset, unsigned int baseOffset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < perChannelLength) {
        data[baseOffset + tid] = (data[baseOffset + tid] - offset) * scale;
    }
}

template <typename T>
void CalScaleOffset(struct NuqAlgoParam& quantParam, int perChannelLength, T& numerator, T& denominator,
    float* scale, int* offset, unsigned int& baseOffset, std::vector<T>& Data, thrust::host_vector<T>& hclusterData)
{
    for (unsigned int channelIndex = 0; channelIndex < quantParam.scaleLength; channelIndex++) {
        baseOffset = channelIndex * perChannelLength;
        numerator = 0.0;
        denominator = 0.0;
        for (unsigned int i = 0; i < perChannelLength; i++) {
            numerator += Data[baseOffset + i] * (hclusterData[baseOffset + i] - offset[channelIndex]);
            denominator += pow(hclusterData[baseOffset + i] - offset[channelIndex], EVEN);
        }
        scale[channelIndex] = numerator / denominator;
        if (std::isnan(scale[channelIndex])) {
            scale[channelIndex] = 1.0;
        }
        if (quantParam.withOffset) {
            numerator = 0.0;
            for (unsigned int i = 0; i < perChannelLength; i++) {
                numerator += Data[baseOffset + i] / scale[channelIndex] - hclusterData[baseOffset + i];
            }
            offset[channelIndex] = -round(numerator / perChannelLength);
        }
    }
}

template <typename T>
Status NuqCalibrationCudaKernel(T* data, int length, struct NuqAlgoParam& quantParam,
    float* scale, int* offset, struct ThrustParam<T>& thrustParam, std::vector<T>& clusterToStore)
{
    const unsigned int perChannelLength = length / quantParam.scaleLength;
    cudaMemcpy(thrustParam.clusterDataPtr, data, sizeof(T) * length, cudaMemcpyDeviceToDevice);
    thrust::device_ptr<T> clusterData = thrust::device_pointer_cast(thrustParam.clusterDataPtr);
    std::vector<T> stdData(length);
    std::vector<T> Data(length);
    cudaMemcpy(Data.data(), thrustParam.clusterDataPtr, sizeof(T) * length, cudaMemcpyDeviceToHost);
    NuqInitG(data, clusterData, quantParam.withOffset, scale, offset, quantParam.scaleLength, perChannelLength);
    ClipAndQuantG(clusterData, quantParam.withOffset, scale, offset, quantParam.scaleLength, perChannelLength);
    unsigned int baseOffset;
    T numerator;
    T denominator;
    WeightG(length, scale, thrustParam.weightPtr, quantParam.scaleLength, perChannelLength);
    cudaMemcpy(stdData.data(), thrustParam.clusterDataPtr, sizeof(T) * length, cudaMemcpyDeviceToHost);
    KmeansG(stdData, thrustParam, clusterData, quantParam.numSteps, clusterToStore);
    thrust::host_vector<T> hclusterData(length);
    cudaMemcpy(hclusterData.data(), thrustParam.clusterDataPtr, sizeof(T) * length, cudaMemcpyDeviceToHost);
    for (unsigned int iter = 0; iter < quantParam.numOfIteration; iter++) {
        CalScaleOffset(quantParam, perChannelLength, numerator, denominator, scale,
                       offset, baseOffset, Data, hclusterData);
        if (iter + 1 == quantParam.numOfIteration) {
            break;
        } else {
            cudaMemcpy(thrustParam.clusterDataPtr, data, sizeof(T) * length, cudaMemcpyDeviceToDevice);
            ClipAndQuantG(clusterData, quantParam.withOffset, scale, offset, quantParam.scaleLength, perChannelLength);
            WeightG(length, scale, thrustParam.weightPtr, quantParam.scaleLength, perChannelLength);
            KmeansG(stdData, thrustParam, clusterData, quantParam.numSteps, clusterToStore);
        }

    }
    cudaDeviceSynchronize();
    // process scale
    int ret;
    for (unsigned int index = 0; index < quantParam.scaleLength; index++) {
        ret = util::ProcessScale(scale[index]);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("NuqCalibrationCudaKernel scale is illegal.\n");
            return ret;
        }
    }
    return AmctCommon::SUCCESS;
}

template<class T>
static Status CheckNuqQuantParams(T* data, unsigned int length, FloatData scale, IntData offset)
{
    if (data == NULL) {
        LOG_ERROR("Empty data pointer!\n");
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

template <class T>
Status NuqQuantKernel(T* data, unsigned int length, NuqParam nuqParam, FloatData scale, IntData offset)
{
    Status ret = CheckNuqQuantParams(data, length, scale, offset);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("Check NUQ parameters failed!\n");
        return ret;
    }
    int inputSize = length;
    int numCandidate = BINARY + static_cast<int>(log(nuqParam.numSteps));

    thrust::device_vector<T> clusterData(inputSize);
    thrust::device_vector<T> weightCluster(nuqParam.numSteps);
    thrust::device_vector<T> clusterCenter(nuqParam.numSteps);
    thrust::device_vector<T> sumCluster(nuqParam.numSteps);
    thrust::device_vector<T> tmpValueData(AMCT_GET_BLOCKS(inputSize));
    thrust::device_vector<T> weightData(inputSize);
    thrust::device_vector<unsigned int> dataCluster(inputSize);
    thrust::device_vector<T> weightTensor(inputSize);
    thrust::device_vector<T> ddistance(inputSize);
    thrust::device_vector<T> dcandiDist(inputSize);
    thrust::device_vector<T> dcandidateDistance(inputSize);
    thrust::device_vector<T> dcumulativeDistance(inputSize);
    thrust::device_vector<int> canData(inputSize);
    thrust::device_vector<T> dcandidateCenter(numCandidate);
    thrust::device_vector<T> dcanidateTotalDistance(numCandidate);

    struct ThrustParam<T> thrustParam;
    thrustParam.clusterDataPtr = thrust::raw_pointer_cast(clusterData.data());
    thrustParam.clusterCenterPtr = thrust::raw_pointer_cast(clusterCenter.data());
    thrustParam.weightPtr = thrust::raw_pointer_cast(weightTensor.data());
    thrustParam.dataClusterPtr = thrust::raw_pointer_cast(dataCluster.data());
    thrustParam.sumClusterPtr = thrust::raw_pointer_cast(sumCluster.data());
    thrustParam.weightClusterPtr = thrust::raw_pointer_cast(weightCluster.data());
    thrustParam.weightDataPtr = thrust::raw_pointer_cast(weightData.data());
    thrustParam.tmpValueDataPtr = thrust::raw_pointer_cast(tmpValueData.data());
    thrustParam.ddistancePtr = thrust::raw_pointer_cast(ddistance.data());
    thrustParam.dcandiDistPtr = thrust::raw_pointer_cast(dcandiDist.data());
    thrustParam.dcumulativeDistancePtr = thrust::raw_pointer_cast(dcumulativeDistance.data());
    thrustParam.canDataPtr = thrust::raw_pointer_cast(canData.data());
    thrustParam.numCandidate = numCandidate;
    thrustParam.dataLength = length;
    thrustParam.numSteps = nuqParam.numSteps;

    NuqAlgoParam nuqAlgoParam{
        static_cast<int>(scale.length),
        static_cast<int>(nuqParam.numSteps),
        static_cast<int>(nuqParam.numIter),
        nuqParam.withOffset};
    std::vector<T> clusterToStore(nuqAlgoParam.numSteps);
    thrust::device_vector<T> deviceData(data, data + length);
    NuqCalibrationCudaKernel(thrust::raw_pointer_cast(deviceData.data()), length,
        nuqAlgoParam, scale.data, offset.data, thrustParam, clusterToStore);
    cudaDeviceSynchronize();

    const unsigned int perChannelLength = length / scale.length;
    thrust::host_vector<T> hclusterData(data, data + length);
    unsigned int baseOffset;
    T* dataPtr = thrust::raw_pointer_cast(clusterData.data());
    // Do fakequant
    for (unsigned int channelIndex = 0; channelIndex < scale.length; channelIndex++) {
        baseOffset = channelIndex * perChannelLength;
        float currScale = scale.data[channelIndex];
        int currOffset = offset.data[channelIndex];
        FakeQuantCuda<<<AMCT_GET_BLOCKS(perChannelLength), AMCT_CUDA_NUM_THREADS>>>(
            dataPtr, perChannelLength, currScale, currOffset, baseOffset);
        cudaError_t errSync = cudaGetLastError();
        CUDA_CHECK_ERROR(errSync);
        cudaError_t errAsync = cudaDeviceSynchronize();
        CUDA_CHECK_ERROR(errAsync);
        thrust::copy(clusterData.begin() + baseOffset, clusterData.begin() + baseOffset + perChannelLength,
                     hclusterData.begin() + baseOffset);
        for (unsigned int i = 0; i < perChannelLength; i++) {
            data[baseOffset + i] = hclusterData[baseOffset + i];
        }
    }

    for (unsigned int i = 0; i < scale.length; i++) {
        ret = util::ProcessScale(scale.data[i]);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("Nuq gpu scale is illegal.\n");
            return AmctCommon::NOT_SUPPORT_ERROR;
        }
    }

    return AmctCommon::SUCCESS;
}


Status NuqQuantReal(double* data, unsigned int length, FloatData scale, IntData offset, char* int8Data)
{
    Status ret = CheckNuqQuantParams(data, length, scale, offset);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("Check NUQ parameters failed!\n");
        return ret;
    }
    unsigned int perChannelLength = length / scale.length;
    unsigned int baseOffset;
    double* offsetData = data;
    char* offsetChar = int8Data;

    for (unsigned int channelIndex = 0; channelIndex < scale.length; channelIndex++) {
        baseOffset = channelIndex * perChannelLength;
        offsetData = data + baseOffset;
        offsetChar = int8Data + baseOffset;
        for (unsigned int i = 0; i < perChannelLength; i++) {
            offsetChar[i] = static_cast<char>(
                round(offsetData[i] / static_cast<double>(scale.data[channelIndex]))
                + static_cast<double>(offset.data[channelIndex]));
        }
    }

    return AmctCommon::SUCCESS;
}

template Status NuqCalibrationCudaKernel(float* data, int length, struct NuqAlgoParam& quantParam,
    float* scale, int* offset, struct ThrustParam<float>& thrustParam, std::vector<float>& clusterToStore);

template Status NuqCalibrationCudaKernel( double* data, int length, struct NuqAlgoParam& quantParam,
    float* scale, int* offset, struct ThrustParam<double>& thrustParam, std::vector<double>& clusterToStore);
}


int NuqQuantPythonGPU(double* data, unsigned int length, AmctCommon::NuqParam nuqParam, FloatData scale, IntData offset)
{
    return AmctCommon::NuqQuantKernel(data, length, nuqParam, scale, offset);
}


int NuqQuantRealPythonGPU(double* data, unsigned int length, FloatData scale, IntData offset, char* int8Data)
{
    return AmctCommon::NuqQuantReal(data, length, scale, offset, int8Data);
}
