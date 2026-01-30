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
