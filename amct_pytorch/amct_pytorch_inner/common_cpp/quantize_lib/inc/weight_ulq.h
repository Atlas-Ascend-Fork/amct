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
 * @brief Declare WeightRetrain Forward Operation
 *
 * @file weight_ulq.h
 *
 * @version 1.0
 */

#ifndef WEIGHT_ULQ_H
#define WEIGHT_ULQ_H

namespace AmctCommon {
constexpr int DATA_IN_INDEX = 0;
constexpr int SCALE_INDEX = 1;
constexpr int WEIGHT_ULQ_OUT_INDEX = 0;
constexpr int WEIGHT_ULQ_SCALE_INDEX = 1;
constexpr int WEIGHT_ULQ_OFFSET_INDEX = 2;
constexpr float BINARY_BASE_FLT = 2.0;

template<typename T>
struct Input {
    const T* data;
    int length;
    int scaleLength;
};

template<typename T>
struct Output {
    T* data;
    float* scale;
    int* offset;
};

struct WeightUlqParam {
    float* scale;
    int scaleLength;
    int quantBits;
    bool sRecFlag;
};

template <typename T>
int ScaleArqInit(const int inputDataSize, const T* inputData, T* max, T* min, WeightUlqParam quantParam);

template <typename T>
int ScaleArqInitCudaHost(const int size, const T* in, T* max, T* min, WeightUlqParam quantParam);

template <typename T>
int WtsFakeQuant(const Input<T> &input, T* output, const float* scale, int quantBitNum, bool sRecFlag);

template <typename T>
int WtsFakeQuantCudaHost(Input<T> input, T* output, const float* scale, int quantBitNum, bool sRecFlag);

void ProcessScale(const float* scaleIn, float* scaleOut, int* offsetOut, int scaleLength, bool sRecFlag);

void ProcessScaleCudaHost(const float* scaleIn, float* scaleOut, int* offsetOut, int scaleLength, bool sRecFlag);
}

#endif // WEIGHT_ULQ_H
