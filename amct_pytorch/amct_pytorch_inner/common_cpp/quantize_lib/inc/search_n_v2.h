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
 * @brief search_n_v2 header file
 *
 * @file search_n_v2.h in common_cpp
 *
 * @version 1.0
 */

#ifndef COMMON_SEARCH_N_V2_H
#define COMMON_SEARCH_N_V2_H

#include "util.h"

namespace AmctCommon {
struct NchwShapeInfo {
    int channelSize;
    int nFactor;
    int hwFactor;
    int cFactor;
};

struct NhwcShapeInfo {
    int channelSize;
    int cFactor;
};


// Define the structure of search_n_v2 algorithm
template <typename T>
struct SearchnV2AlgoParam {
    // whether reduce error per tensor
    bool isBroadcast;
    // device pointer of quantErrorSum, size is AMCT_GET_BLOCKS(searchNDataSize)
    T* quantErrorSum;
    // size of device pointer quantErrorSum
    int quantErrorSumSize;
    // device pointer of deq scale
    float* deqScale;
    // size of deqScale and deqScaleCpu
    int deqScaleSize;
    // channelNum of target operator of search_n_v2
    int channelNum;
};


template <typename T>
struct SearchnV2InputParam {
    // the vector to store the searchn error
    std::vector<std::vector<T>> storeError;
    // size of input data
    int size;
    // devicePtr
    const T* in;
};

template <typename T>
int SearchNV2FindBestNCpu(
    std::vector<std::vector<T>>& storeError,
    util::IntData& bestN,
    bool isBroadcast);


template <typename T>
int SearchNV2AccumulateError(std::vector<std::vector<T>>& currentData, std::vector<std::vector<T>>& searchNError,
    const util::FloatData &deqScaleCpu, bool isBroadcast);


template <typename T>
int AccumulateErrorPerChannelNchwCuda(
    struct SearchnV2InputParam<T>& inputParam,
    struct SearchnV2AlgoParam<T>& algoParam,
    struct NchwShapeInfo& shapeInfo);

template <typename T>
int AccumulateErrorPerChannelNhwcCuda(
    struct SearchnV2InputParam<T>& inputParam,
    struct SearchnV2AlgoParam<T>& algoParam,
    struct NhwcShapeInfo& shapeInfo);

template <typename T>
int AccumulateErrorPerTensorCuda(
    struct SearchnV2InputParam<T>& inputParam,
    struct SearchnV2AlgoParam<T>& algoParam);
}

#endif // COMMON_SEARCH_N_V2_H
