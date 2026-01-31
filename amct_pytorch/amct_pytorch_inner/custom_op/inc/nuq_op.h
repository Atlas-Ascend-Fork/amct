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
 * @brief nuq_quant algorithm custom op C++ implement
 *
 * @file nuq.h
 *
 * @version 1.0
 */

#ifndef NUQ_OP_H
#define NUQ_OP_H

#include <torch/extension.h>

std::vector<torch::Tensor> NuqCali(
    torch::Tensor input,
    int numBits,
    bool withOffset,
    unsigned int numSteps,
    unsigned int numIter,
    bool channelWise);

#endif /* NUQ_OP_H */
