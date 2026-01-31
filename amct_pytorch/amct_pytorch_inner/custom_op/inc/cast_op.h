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