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
 * @brief ULQ algorithm C++ implementation for CPU.
 *
 * @file ulq.cpp
 *
 * @version 1.0
 */

#include "ulq.h"
#include <cmath>
#include <cstdio>

using namespace util;

namespace AmctCommon {
template <class T>
void TransposeAB(T* data, int length, std::vector<int>& shape)
{
    int shapeLength = 1;
    const unsigned minLength = 2;

    if (shape.size() < minLength) {
        LOG_ERROR("The shape dimensions less than 2!\n");
        return;
    }

    for (size_t i = 0; i < shape.size(); i++) {
        shapeLength *= shape[i];
    }

    if (shapeLength != length) {
        LOG_ERROR("The length and length calculated by shape are different!\n");
        return;
    }

    int cellSize = 1;
    std::vector<T> dataCopy(data, data + length);

    // Calculate the minimum cell size
    for (size_t i = minLength; i < shape.size(); i++) {
        cellSize *= shape[i];
    }

    // Swap the axis 0 and axis 1
    int trans = shape[0];
    shape[0] = shape[1];
    shape[1] = trans;

    // Rearrangement
    for (int i = 0; i < length; i++) {
        int cIn = i / (shape[0] * cellSize);
        int cOut = i % (shape[0] * cellSize) / cellSize;
        trans = cOut * (shape[1] * cellSize) + cIn * cellSize + i % cellSize;
        data[static_cast<unsigned int>(trans)] = dataCopy[static_cast<unsigned int>(i)];
    }
}


template <class T>
bool ClipCheck(T& clipMax, T& clipMin, T& clipMaxPre, T& clipMinPre)
{
    if (clipMax <= clipMin) {
        clipMax = clipMaxPre;
        clipMin = clipMinPre;
        return true;
    } else if (clipMax < 0) {
        clipMax = clipMaxPre;
        clipMinPre = clipMin;
        return false;
    } else if (clipMin > 0) {
        clipMaxPre = clipMax;
        clipMin = clipMinPre;
        return false;
    } else {
        clipMaxPre = clipMax;
        clipMinPre = clipMin;
        return false;
    }
}


template <class T>
void Ulq(const T* inData, T* outData, int length, const FloatData &scale, const IntData &offset, const int numBits)
{
    T clipMax = scale.data[0] * (offset.data[0] + pow(BINARY_BASE, numBits) - 1);
    T clipMin = scale.data[0] * offset.data[0];

#pragma omp parallel for
    for (int i = 0; i < length; i++) {
        outData[i] = inData[i];
        if (outData[i] < clipMin) {
            outData[i] =  clipMin;
        } else if (outData[i] > clipMax) {
            outData[i] = clipMax;
        }
        outData[i] = rint(outData[i] / scale.data[0]) * scale.data[0];
    }
}


template <class T>
T DiffFunc(T data, T clipMax, T clipMin)
{
    T step = pow(BINARY_BASE, NUM_BITS_QUANT) - 1;
    T temp = data * step / (clipMax - clipMin);
    T result = (1 / step) * (round(temp) - temp);
    return result;
}


template <class T>
std::vector<T> UlqDiff(const T* bottomData, const T* topDiff, const int length, std::vector<T> clip)
{
    T diffTotalClipMax = 0;
    T diffTotalClipMin = 0;
    const int clipMaxIndex = 0;
    const int clipMinIndex = 1;
    const int clipMaxOriIndex = 2;
    const int clipMinOriIndex = 3;

    // Computing the gradient
    for (int i = 0; i < length; i++) {
        if (bottomData[i] > clip[clipMaxIndex]) {
            diffTotalClipMax +=
                topDiff[i] * (DiffFunc(clip[clipMinOriIndex], clip[clipMaxOriIndex], clip[clipMinOriIndex]) + 1);
            diffTotalClipMin +=
                topDiff[i] * (-DiffFunc(clip[clipMinOriIndex], clip[clipMaxOriIndex], clip[clipMinOriIndex]));
        } else if (bottomData[i] < clip[clipMinIndex]) {
            diffTotalClipMax +=
                topDiff[i] * DiffFunc(clip[clipMinOriIndex], clip[clipMaxOriIndex], clip[clipMinOriIndex]);
            diffTotalClipMin +=
                topDiff[i] * (1 - DiffFunc(clip[clipMinOriIndex], clip[clipMaxOriIndex], clip[clipMinOriIndex]));
        } else {
            diffTotalClipMax += topDiff[i] * DiffFunc(bottomData[i], clip[clipMaxOriIndex], clip[clipMinOriIndex]);
            diffTotalClipMin += topDiff[i] * (-DiffFunc(bottomData[i], clip[clipMaxOriIndex], clip[clipMinOriIndex]));
        }
    }

    std::vector<T> diffTotal {diffTotalClipMax, diffTotalClipMin};
    return diffTotal;
}

template void TransposeAB(float* data, int length, std::vector<int>& shape);

template void TransposeAB(double* data, int length, std::vector<int>& shape);

template bool ClipCheck(float& clipMax, float& clipMin, float& clipMaxPre, float& clipMinPre);

template bool ClipCheck(double& clipMax, double& clipMin, double& clipMaxPre, double& clipMinPre);

template void Ulq(const float* inData, float* outData, int length,
    const FloatData &scale, const IntData &offset, const int numBits);

template void Ulq(const double* inData, double* outData, int length,
    const FloatData &scale, const IntData &offset, const int numBits);

template std::vector<float> UlqDiff(const float* bottomData, const float* topDiff,
    const int length, std::vector<float> clip);

template std::vector<double> UlqDiff(const double* bottomData, const double* topDiff,
    const int length, std::vector<double> clip);
}
