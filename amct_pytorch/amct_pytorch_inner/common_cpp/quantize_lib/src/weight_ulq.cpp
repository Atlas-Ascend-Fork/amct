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
 * @brief weight ulq cpp implementation
 *
 * @version 1.0
 */

#include "weight_ulq.h"
#include <cmath>
#include "util.h"

using namespace util;

namespace AmctCommon {
    template <typename T>
    int ScaleArqInit(const int inputDataSize, const T* inputData, T* max, T* min, WeightUlqParam quantParam)
    {
        for (int i = 0; i < quantParam.scaleLength; i++) {
            max[i] = 0.0;
            min[i] = 0.0;
        }
        for (int i = 0; i < inputDataSize; i++) {
            int j = i % quantParam.scaleLength;
            max[j] = (inputData[i] > max[j]) ? inputData[i] : max[j];
            min[j] = (inputData[i] < min[j]) ? inputData[i] : min[j];
        }
        // calculate scale value
        int steps = static_cast<int>(pow(BASE, quantParam.quantBits - 1));
        if (steps > 1) {
            for (int i = 0; i < quantParam.scaleLength; i++) {
                quantParam.scale[i] = (-1 * min[i] > max[i]) ? -1 * min[i] / steps : max[i] / (steps - 1);
                quantParam.scale[i] = (quantParam.scale[i] < EPSILON) ? 1.0 : quantParam.scale[i];
                // when sRecFlag is true, need to save the 1 / scale;
                quantParam.scale[i] = quantParam.sRecFlag ? 1 / quantParam.scale[i] : quantParam.scale[i];
            }
        }
        return AmctCommon::SUCCESS;
    }

    template <typename T>
    int WtsFakeQuant(const Input<T> &input, T* output, const float* scale, int quantBitNum, bool sRecFlag)
    {
        T minValue = static_cast<T>(-pow(AmctCommon::BINARY_BASE_FLT, quantBitNum - 1));
        T maxValue = static_cast<T>(pow(AmctCommon::BINARY_BASE_FLT, quantBitNum - 1) - 1);
        for (int i = 0; i < input.length; i++) {
            int j = i % input.scaleLength;
            float currentScale = scale[j];
            if (currentScale < EPSILON) {
                currentScale += EPSILON;
            }

            currentScale = sRecFlag ? 1 / currentScale : currentScale;
            T quantValue = static_cast<T>(round(input.data[i] / currentScale));
            if (quantValue < minValue) {
                quantValue = minValue;
            } else if (quantValue > maxValue) {
                quantValue = maxValue;
            }
            output[i] = quantValue * currentScale;
        }
        return AmctCommon::SUCCESS;
    }

    void ProcessScale(const float* scaleIn, float* scaleOut, int* offsetOut, int scaleLength, bool sRecFlag)
    {
        for (int i = 0; i < scaleLength; i++) {
            float currentScale = static_cast<float>(fabs(scaleIn[i]));
            if (currentScale < EPSILON) {
                currentScale += EPSILON;
            }
            scaleOut[i] = sRecFlag ? 1 / currentScale : currentScale;
            offsetOut[i] = 0;
        }
    }


    template int ScaleArqInit(const int, const float*, float*, float*, WeightUlqParam);
    template int WtsFakeQuant(const Input<float> &, float*, const float*, int, bool);
}
