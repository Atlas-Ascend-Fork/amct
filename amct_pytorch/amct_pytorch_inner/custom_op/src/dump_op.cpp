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
 * @brief torch C++ backend api of ifmr algorithm.
 *
 * @file dump_op.cpp
 *
 * @version 1.0
 */

#include "dump_op.h"
#include "dump.h"

#include "util.h"


Status amct_pytorch::DumpForward
(
    torch::Tensor input,
    const std::string dumpDir,
    const std::string layerName,
    const int batchNum
)
{
    // get the input tensor's dim length.
    auto inputSize = input.sizes();
    unsigned int inputSizeLen = inputSize.size();
    amct_pytorch::DumpParam dumpParam = {dumpDir, layerName, batchNum, std::vector<float>(), inputSizeLen + 1};
    dumpParam.inputShape.push_back(static_cast<float>(inputSizeLen));
    for (unsigned int dimIndex = 0; dimIndex < inputSizeLen; dimIndex++) {
        // get the input tensor's each dim size.
        dumpParam.inputShape.push_back(static_cast<float>(input.size(dimIndex)));
    }
    std::string fileName = dumpDir + '/' + layerName + "_batch" + std::to_string(batchNum) + ".bin";
    struct AmctCommon::DumpParam commonParam = {fileName, dumpParam.inputShape, dumpParam.inputShapeLength};
    // clone the input.
    auto inputClone = input.clone();
    // ONLY support CPU, and GPU is not required actually.
    if (inputClone.is_cuda()) {
        inputClone = inputClone.cpu();
    }
    // memory continuity.
    inputClone = inputClone.contiguous();
    // now only support float and double and int, and can be extended.
    // If extended, make sure common cpp support.
    // now, common support float, double, int.
    if (inputClone.dtype() == torch::kFloat32) {
        AmctCommon::DumpDataWithType(inputClone.data_ptr<float>(), inputClone.numel(), commonParam);
    } else if (inputClone.dtype() == torch::kFloat64) {
        AmctCommon::DumpDataWithType(inputClone.data_ptr<double>(), inputClone.numel(), commonParam);
    } else if (inputClone.dtype() == torch::kInt32) {
        AmctCommon::DumpDataWithType(inputClone.data_ptr<int>(), inputClone.numel(), commonParam);
    } else {
        return AmctCommon::NOT_SUPPORT_ERROR;
    }
    return AmctCommon::SUCCESS;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dump_forward", &amct_pytorch::DumpForward, "DUMP forward");
}
