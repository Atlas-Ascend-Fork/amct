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
 * @brief NUQ torch c++ functuion
 *
 * @file nuq_torch.cpp
 *
 * @version 1.0
 */
#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <vector>

#include "nuq_op.h"
#include "nuq.h"
#include "util.h"

using namespace util;

std::vector<torch::Tensor> NuqCali(
    torch::Tensor input,
    int numBits = 8,
    bool withOffset = false,
    unsigned int numSteps = 32,
    unsigned int numIter = 1,
    bool channelWise = true)
{
    AmctCommon::NuqParam nuqParam = {static_cast<unsigned int>(numSteps), withOffset,
        static_cast<unsigned int>(numIter)};

    torch::Tensor scale;
    torch::Tensor offset;
    if (channelWise) {
        scale = torch::zeros({input.sizes()[0]});
        offset = torch::zeros({input.sizes()[0]}, torch::kInt32);
    } else {
        scale = torch::zeros({1});
        offset = torch::zeros({1}, torch::kInt32);
    }

    torch::Tensor quantizeData = input.clone();
    if (quantizeData.is_cuda()) {
        quantizeData = quantizeData.cpu();
        scale = scale.cpu();
        offset = offset.cpu();
    }
    quantizeData = quantizeData.contiguous();
    scale = scale.contiguous();
    offset = offset.contiguous();

    FloatData scaleData = {static_cast<uint32_t>(scale.numel()),  scale.data_ptr<float>()};
    IntData offsetData = {static_cast<uint32_t>(offset.numel()),  offset.data_ptr<int>()};

    Status status = NuqKernel(quantizeData.data_ptr<float>(),
        static_cast<uint32_t>(input.numel()), nuqParam, scaleData, offsetData);

    torch::Tensor calibratedFlag = torch::scalar_tensor(status);

    auto channelSize = static_cast<int>(scale.numel());
    scale = torch::from_blob(scaleData.data, {channelSize}, torch::kFloat32).clone();
    offset = torch::from_blob(offsetData.data, {channelSize}, torch::kInt32).clone();

    std::vector<torch::Tensor> tensors{calibratedFlag, scale, offset, quantizeData};
    return tensors;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nuq_cali", &NuqCali, "NUQ calibration to find scale and offset");
}