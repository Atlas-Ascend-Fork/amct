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
 * @brief dump algorithm custom op C++ implement
 *
 * @file dump_op.h
 *
 * @version 1.0
 */

#ifndef DUMP_OP_H
#define DUMP_OP_H

#include <torch/extension.h>

#include <string>
#include <vector>

#include "util.h"

namespace amct_pytorch {
/**
  * @ingroup amct_pytorch custom op.
  * @brief: DumpParam struct.
  * @param [in] dumpDir: data dump's data.
  * @param [in] namePrefix: saved data's name prefix.
  * @param [in] batchCounter: current batch num.
  * @param [in] inputShape: dim, shape.
  * @param [in] inputShapeLength: shape length.
  */
struct DumpParam {
    const std::string dumpDir;
    const std::string namePrefix;
    int batchCounter;
    std::vector<float> inputShape;
    unsigned int inputShapeLength;
};

/**
  * @ingroup python dump forward function.
  * @brief: Dump Function.
  * @param [in] input: input data tensor.
  * @param [in] dump_dir: dir to dump.
  * @param [in] layer_name: string, saved data's prefix.
  * @param [in] batch_num: current batch num.
  * @return succ/fail
  */
Status DumpForward(
    torch::Tensor input,
    const std::string dump_dir,
    const std::string layer_name,
    const int batch_num
);
} // namespace amct_pytorch
#endif /* DUMP_OP_H */
