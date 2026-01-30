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
 * @brief hfmg header file
 *
 * @file hfmg.h in common_cpp
 *
 * @version 1.0
 */

#ifndef COMMON_HFMG_H
#define COMMON_HFMG_H

#include <cmath>
#include "util.h"

namespace AmctCommon {
constexpr unsigned int HFMG_THREE = 3;
constexpr unsigned int HFMG_TWO = 2;
constexpr int NUM_LIMIT = 128;
constexpr unsigned int MIN_BIN_RATIO = 4;
constexpr unsigned int STEP_DIVISOR = 100000;

inline float GetNorm(float deltaBegin, float deltaEnd, float density)
{
    float norm = 0.0f;
    norm = (pow(deltaEnd, HFMG_THREE) - pow(deltaBegin, HFMG_THREE)) / HFMG_THREE;
    norm = norm * density;
    return norm;
}

inline double GetNorm(double deltaBegin, double deltaEnd, double density)
{
    double norm = 0.0;
    norm = (pow(deltaEnd, HFMG_THREE) - pow(deltaBegin, HFMG_THREE)) / HFMG_THREE;
    norm = norm * density;
    return norm;
}

struct HfmgAlgoParam {
    unsigned int quantBitNum;
    bool withOffset;
    unsigned int nbins;
};


template <typename T>
struct InputData {
    InputData<T>(unsigned int sizeIn, const T* inData): size(sizeIn), in(inData) {}
    ~InputData<T>() {}
    unsigned int size;
    const T* in;
};


template<class T>
struct DataBin {
    DataBin<T>(unsigned int countIn, T lowerBoundIn, T higherBoundIn): count(countIn),
        lowerBound(lowerBoundIn),
        higherBound(higherBoundIn) {}
    ~DataBin<T>() {}
    // number of elements inside the bin
    unsigned int count;
    // lowerBound of bin
    T lowerBound;
    // higherBound of bin
    T higherBound;
};

template <class T>
Status ActArqCalibration(T inputMin, T inputMax,
    const util::FloatData &scale, const util::IntData &offset, const HfmgAlgoParam& hfmgParam);

template <typename T>
void CalScaleOffset(T max, T min, float& scaleCpu, int& offsetCpu, const HfmgAlgoParam& hfmgParam);

template <typename T>
void HfmgMergeInter(std::vector<DataBin<T>>& dataBins, std::vector<DataBin<T>>& mergedDataBins,
    bool sameRangeFlag, T mergedDataMin, T mergedBinWidth);

template <typename T>
void HfmgGetSearchRange(std::vector<DataBin<T>>& dataBins,
    std::vector<std::pair<unsigned int, unsigned int>>& searchRange);


/**
  * @ingroup quantize lib
  * @brief: hfmg Quantization kernel Function [CPU].
  * @param [in|out] dataBins: HFMG databins.
  * @param [in] inputData: inputs data.
  * @return succ/fail
  */
template<class T>
int HfmgMerge(int nbins, std::vector<DataBin<T>>& dataBins, const InputData<T>& inputData);

/**
  * @ingroup quantize lib
  * @brief: hfmg Quantization kernel Function [CPU].
  * @param [in|out] dataBins: HFMG databins.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @param [in] hfmgParam: hfmg algorithm params.
  * @return succ/fail
  */
template<class T>
int HfmgCompute(std::vector<DataBin<T>>& dataBins, float& scale, int& offset, const HfmgAlgoParam& hfmgParam);

/**
  * @ingroup quantize lib
  * @brief: hfmg Quantization kernel Function [CUDA].
  * @param [in] binCountsDevicePtr: bincount devicePtr, lenght nbins.
  * @param [in] nbins: nbins.
  * @param [in|out] dataBins: dataBins of hfmg.
  * @param [in] inputData: input data.
  * @return succ/fail
  */
template <typename T>
int HfmgMergeCuda(int* binCountsDevicePtr, int nbins, std::vector<DataBin<T>>& dataBins,
    struct InputData<T>& inputData);

/**
  * @ingroup quantize lib
  * @brief: hfmg Quantization cuda kernel Function [CUDA].
  * @param [in] binCountsDevice: the bincount device data, lenght the same as nbins.
  * @param [in] l2LossDevice: the l2loss device data, lenght the same as nbins.
  * @param [in] dataBins: dataBins data.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @param [in] hfmgParam: hfmg algorithm parms.
  * @return succ/fail
  */
template<class T>
int HfmgComputeCuda(util::IntData binCountsDevice, util::FloatData l2LossDevice,
    std::vector<DataBin<T>>& dataBins, float& scale, int& offset, struct HfmgAlgoParam& hfmgParam);
}

#endif // COMMON_HFMG_H
