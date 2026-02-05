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
