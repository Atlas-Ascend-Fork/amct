/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
