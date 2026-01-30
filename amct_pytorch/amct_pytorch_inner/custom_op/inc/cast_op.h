/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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
 * @brief torch C++ backend api of hif8 and fp8 cast function.
 *
 * @file cast_op.h
 *
 * @version 1.0
 */
#include "util_op.h"
#include "cast_util.h"
#include "util.h"
#include <torch/extension.h>

#ifndef CAST_OP_H
#define CAST_OP_H
torch::Tensor CastToHiFP8(torch::Tensor input, int roundMode);
torch::Tensor HiFP8CastToFloat(torch::Tensor input, int dataType);
torch::Tensor CastToFP8E4M3FN(torch::Tensor input);
torch::Tensor FP8E4M3FNCastToFloat(torch::Tensor input, int dataType);
torch::Tensor CastToFP4E2M1(torch::Tensor input);
torch::Tensor FP4E2M1CastToFloat(torch::Tensor input, int dataType);
torch::Tensor FP4E1M2CastToFloat(torch::Tensor input, int dataType);
#endif
/* CAST_OP_H */