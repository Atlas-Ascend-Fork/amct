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
 * @brief utls for custom op C++ implement
 *
 * @file utils_pytorch.h
 *
 * @version 1.0
 */

#include <sstream>
#include "util.h"
#include "util_op.h"

const int DATA_DIM = 2;
const int CHANNEL_DIM = 1;

int CheckCaliParams(torch::Tensor &data, torch::Tensor &deqScale)
{
    if (data.sizes() == torch::IntArrayRef{0}) {
        LOG_ERROR("Empty input data!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if (data.dim() != DATA_DIM) {
        LOG_ERROR("Parameter data's dim should be 2!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if (deqScale.sizes() == torch::IntArrayRef{0}) {
        LOG_ERROR("Empty deqScale!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if (deqScale.dim() != 1) {
        LOG_ERROR("Parameter deqScale's dim should be 1!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if (deqScale.sizes()[0] != 1 && data.sizes()[CHANNEL_DIM] != deqScale.sizes()[0]) {
        LOG_ERROR("deqScale's length is unequal to input's shape[1] when channel_wise is true!\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    return AmctCommon::SUCCESS;
}


void TensorToVector(torch::Tensor input, std::vector<float> &inputVector)
{
    // Carry data to cpu memorry for data accumulation
    if (input.is_cuda()) input = input.cpu();
    // Insure the input data is stored in continous mermory
    input = input.contiguous();
    // Accumulate the data in current batch to memory
    float* inputPtr = input.data_ptr<float>();
    if (inputPtr == nullptr) {
        return;
    }
    inputVector.insert(inputVector.end(), inputPtr, inputPtr + input.numel());
}
