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
 * @brief util head file
 *
 * @file util.h
 *
 * @version 1.0
 */

#ifndef UTIL_OP_H
#define UTIL_OP_H

#include <torch/extension.h>

const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int KERNEL_DIM_4D = 4;
const int SQURE = 2;
const int BASENUM = 2;
const int ZERO = 0;
const int ONE = 1;
const float SEARCHSTART = 0.7;
const float SEARCHEND = 1.3;
const float SEARCHSTEP = 0.01;

int CheckCaliParams(torch::Tensor &data, torch::Tensor &deqScale);

void TensorToVector(torch::Tensor input, std::vector<float> &inputVector);

#endif /* UTIL_OP_H */
