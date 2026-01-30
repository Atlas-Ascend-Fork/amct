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
 * @brief HFMG algorithm C++ implementation for CPU.
 *
 * @file hfmg.cu
 *
 * @version 1.0
 */

#include <thrust/reduce.h>
#include "hfmg.h"

using namespace util;

namespace AmctCommon {
template <typename T>
__global__ void DistributeDataKernel(
    T dataMin,
    T binWidth,
    unsigned int nbins,
    const T* in,
    int size,
    int* binCounts)
{
    CUDA_KERNEL_LOOP(index, size) {
        unsigned int binIndex = static_cast<unsigned int>(floor((in[index] - dataMin) / binWidth));
        if (binIndex >= nbins) {
            binIndex = nbins - 1;
        }
        atomicAdd(&(binCounts[binIndex]), 1);
    }
}

template <typename T>
__device__ void getNormDevice(T deltaBegin, T deltaEnd, T density, T& norm)
{
    norm = (pow(deltaEnd, HFMG_THREE) - pow(deltaBegin, HFMG_THREE)) / HFMG_THREE;
    norm = norm * density;
}

template <typename T>
__global__ void CalculateL2LossKernel(
    float* l2lossDevicePtr,
    int* binCountsDevicePtr,
    int nbins,
    int dstNumOfBins,
    T binWidth,
    T dstBinWidth,
    unsigned int binIndexLeft,
    unsigned int binIndexRight)
{
    CUDA_KERNEL_LOOP(index, nbins) {
        T srcBinBegin = (index - binIndexLeft) * binWidth;
        T srcBinEnd = srcBinBegin + binWidth;
        T deltaBegin;
        T deltaEnd;
        int tempSrcBinBegin = static_cast<int>(floor(srcBinBegin / dstBinWidth));
        int tempSrcBinEnd = static_cast<int>(floor(srcBinEnd / dstBinWidth));
        int temp1 = tempSrcBinBegin > 0 ? tempSrcBinBegin : 0;
        int temp2 = tempSrcBinEnd > 0 ? tempSrcBinEnd : 0;
        int dstBinBeginIndex = dstNumOfBins - 1 < temp1 ? dstNumOfBins - 1 : temp1;
        int dstBinEndIndex = dstNumOfBins - 1 < temp2 ? dstNumOfBins - 1 : temp2;
        T dstBinBeginCenter = dstBinBeginIndex * dstBinWidth + dstBinWidth / BASE;
        T density = binCountsDevicePtr[index] / binWidth;
        T norm;
        T normSum = 0;
        if (dstBinBeginIndex == dstBinEndIndex) {
            // if src_bin is entirely within 1 dst_bin
            deltaBegin = srcBinBegin - dstBinBeginCenter;
            deltaEnd = srcBinEnd - dstBinBeginCenter;
            getNormDevice(deltaBegin, deltaEnd, density, norm);
            normSum += norm;
        } else {
            // the left dst bin and src bin overlapping parts
            deltaBegin = srcBinBegin - dstBinBeginCenter;
            deltaEnd = dstBinWidth / BASE;
            getNormDevice(deltaBegin, deltaEnd, density, norm);
            normSum += norm;
            // middle full dst bin loss
            getNormDevice(-dstBinWidth / BASE, dstBinWidth / BASE, density, norm);
            normSum += (dstBinEndIndex - dstBinBeginIndex - 1) * norm;
            // the right dst bin and src bin overlapping parts
            T dstBinEndCenter = dstBinEndIndex * dstBinWidth + dstBinWidth / BASE;
            deltaBegin = -dstBinWidth / BASE;
            deltaEnd = srcBinEnd - dstBinEndCenter;
            getNormDevice(deltaBegin, deltaEnd, density, norm);
            normSum += norm;
        }
        l2lossDevicePtr[index] = normSum;
    }
}


template <typename T>
int HfmgDistributeInterCuda(
    int nbins,
    T dataMin,
    T binWidth,
    std::vector<DataBin<T>>& dataBins,
    struct InputData<T>& inputData,
    int* binCountsDevicePtr)
{
    T lowerBound;
    T higherBound;
    unsigned int numOfBins = static_cast<unsigned int>(nbins);
    bool minMaxEqualFlag = false;
    if (binWidth < FLT_EPSILON) {
        minMaxEqualFlag = true;
    }
    // initialize all the bins;
    for (unsigned int index = 0; index < numOfBins; index ++) {
        lowerBound = dataMin + index * binWidth;
        higherBound = dataMin + (index + 1) * binWidth;
        dataBins.push_back(DataBin<T>(0, lowerBound, higherBound));
    }
    // intialize quantParam.binCounts
    cudaMemset(binCountsDevicePtr, 0, sizeof(int) * numOfBins);
    if (!minMaxEqualFlag) {
        DistributeDataKernel<T><<<AMCT_GET_BLOCKS(inputData.size), AMCT_CUDA_NUM_THREADS>>>(
            dataMin, binWidth, numOfBins, inputData.in, inputData.size, binCountsDevicePtr);
        std::unique_ptr<int> binCountsCpu(new(std::nothrow) int[numOfBins]);
        NULLPTR_CHECK(binCountsCpu);
        CUDA_MEMCPY_ERROR_CHECK(cudaMemcpy(
            binCountsCpu.get(), binCountsDevicePtr, sizeof(int) * numOfBins, cudaMemcpyDeviceToHost));
        for (int i = 0 ; i < numOfBins; i++) {
            dataBins[i].count = binCountsCpu.get()[i];
        }
    } else {
        // when min and max arq equal, count all the elements to the middle bin of dataBins;
        unsigned int binIndex = static_cast<unsigned int>(floor(numOfBins / BASE));
        dataBins[binIndex].count += inputData.size;
    }
    return AmctCommon::SUCCESS;
}


template <typename T>
int HfmgDistributeCuda(
    int* binCountsDevicePtr,
    int nbins,
    std::vector<DataBin<T>>& dataBins,
    struct InputData<T>& inputData)
{
    thrust::device_ptr<const T> indataDevicePtr = thrust::device_pointer_cast(inputData.in);
    T dataMin = *thrust::min_element(indataDevicePtr, indataDevicePtr + inputData.size);
    T dataMax = *thrust::max_element(indataDevicePtr, indataDevicePtr + inputData.size);
    T binWidth = (dataMax - dataMin) / nbins;
    HfmgDistributeInterCuda<T>(nbins, dataMin, binWidth, dataBins, inputData, binCountsDevicePtr);
    return AmctCommon::SUCCESS;
}


template <typename T>
int HfmgMergeCuda(
     int* binCountsDevicePtr,
     int nbins,
     std::vector<DataBin<T>>& dataBins,
     struct InputData<T>& inputData)
{
    if (inputData.in == nullptr) {
        return AmctCommon::NULL_PTR_ERROR;
    }
    if (dataBins.size() == 0) {
        HfmgDistributeCuda<T>(binCountsDevicePtr, nbins, dataBins, inputData);
    }
    // merge the new inputData.in into existed dataBins
    bool sameRangeFlag = false;

    thrust::device_ptr<const T> indataDevicePtr = thrust::device_pointer_cast(inputData.in);
    T currentDataMin = *thrust::min_element(indataDevicePtr, indataDevicePtr + inputData.size);
    T currentDataMax = *thrust::max_element(indataDevicePtr, indataDevicePtr + inputData.size);
    T dataMin = dataBins.front().lowerBound;
    T dataMax = dataBins.back().higherBound;
    T mergedDataMin = std::min(currentDataMin, dataMin);
    T mergedDataMax = std::max(currentDataMax, dataMax);
    T mergedBinWidth = (mergedDataMax - mergedDataMin) / nbins;
    std::vector<DataBin<T>> mergedDataBins;

    HfmgDistributeInterCuda<T>(nbins, mergedDataMin, mergedBinWidth, mergedDataBins, inputData, binCountsDevicePtr);
    // when mergedDataMax equal last data max and  mergedDataMin equal last data min;
    // means do not need to change data range;
    if ((mergedDataMax - dataMax) < FLT_EPSILON && \
            (mergedDataMin - dataMin) < FLT_EPSILON) {
        sameRangeFlag = true;
    }
    // merge the existed dataBins and the new bins
    HfmgMergeInter<T>(dataBins, mergedDataBins, sameRangeFlag, mergedDataMin, mergedBinWidth);
    // swap Histogram dataBins;
    dataBins.swap(mergedDataBins);
    return AmctCommon::SUCCESS;
}

template<class T>
int HfmgComputeCuda(
    IntData binCountsDevice,
    FloatData l2LossDevice,
    std::vector<DataBin<T>>& dataBins,
    float& scale,
    int& offset,
    struct HfmgAlgoParam& hfmgParam)
{
    if (dataBins.size() == 0) {
        return AmctCommon::CONTAINER_EMPTY_ERROR;
    }
    int nbins = binCountsDevice.length;
    std::vector<std::pair<unsigned int, unsigned int>> searchRange;
    HfmgGetSearchRange<T>(dataBins, searchRange);

    std::vector<float> allL2Loss(searchRange.size());
    // start to calculate the l2 loss of all candidates
    unsigned int dstNumOfBins = pow(BASE, hfmgParam.quantBitNum);
    T dataMin = dataBins.front().lowerBound;
    T dataMax = dataBins.back().higherBound;
    T binWidth = (dataMax - dataMin) / nbins;
    std::unique_ptr<int> binCountsCpu(new(std::nothrow) int[nbins]);
    NULLPTR_CHECK(binCountsCpu);
    for (int i = 0 ; i < nbins; i++) {
        binCountsCpu.get()[i] = dataBins[i].count;
    }
    CUDA_MEMCPY_ERROR_CHECK(cudaMemcpy(
        binCountsDevice.data, binCountsCpu.get(), sizeof(int) * nbins, cudaMemcpyHostToDevice));
    for (size_t index = 0; index < searchRange.size(); index++) {
        float l2Loss = 0.0;
        auto range = searchRange[index];
        T dstBinWidth = binWidth * (range.second - range.first + 1) / dstNumOfBins;
        if (dstBinWidth < FLT_EPSILON) {
            l2Loss = 0.0;
        } else {
            CalculateL2LossKernel<T><<<AMCT_GET_BLOCKS(nbins), AMCT_CUDA_NUM_THREADS>>>(
                l2LossDevice.data, binCountsDevice.data, nbins, dstNumOfBins,
                binWidth, dstBinWidth, range.first, range.second);
            l2Loss = thrust::reduce(
                thrust::device,
                l2LossDevice.data,
                l2LossDevice.data + l2LossDevice.length);
        }
        allL2Loss[index] = l2Loss;
    }
    auto bestMaxIndex = std::distance(
        std::begin(allL2Loss), std::min_element(std::begin(allL2Loss), std::end(allL2Loss)));
    unsigned int bestLeft = searchRange[bestMaxIndex].first;
    unsigned int bestRight = searchRange[bestMaxIndex].second;
    T clipMin = dataMin + bestLeft * binWidth;
    T clipMax = dataMin + (bestRight + 1) * binWidth;
    if (hfmgParam.withOffset) {
        clipMax = clipMax > 0 ? clipMax : 0;
        clipMin = clipMin < 0 ? clipMin : 0;
    } else {
        clipMax = fabs(clipMax) > fabs(clipMin) ? clipMax : clipMin;
        clipMin = 0;
    }

    CalScaleOffset(clipMax, clipMin, scale, offset, hfmgParam);
    return AmctCommon::SUCCESS;
}

template int HfmgDistributeCuda(int* binCountsDevicePtr, int nbins,
    std::vector<DataBin<float>>& dataBins, struct InputData<float>& inputData);
template int HfmgDistributeCuda(int* binCountsDevicePtr, int nbins,
    std::vector<DataBin<double>>& dataBins, struct InputData<double>& inputData);

template int HfmgMergeCuda(int* binCountsDevicePtr, int nbins,
    std::vector<DataBin<float>>& dataBins, struct InputData<float>& inputData);
template int HfmgMergeCuda(int* binCountsDevicePtr, int nbins,
    std::vector<DataBin<double>>& dataBins, struct InputData<double>& inputData);

template int HfmgComputeCuda(IntData binCountsDevice, FloatData l2LossDevice,
    std::vector<DataBin<float>>& dataBins, float& scale, int& offset, struct HfmgAlgoParam& hfmgParam);
template int HfmgComputeCuda(IntData binCountsDevice, FloatData l2LossDevice,
    std::vector<DataBin<double>>& dataBins, float& scale, int& offset, struct HfmgAlgoParam& hfmgParam);
}
