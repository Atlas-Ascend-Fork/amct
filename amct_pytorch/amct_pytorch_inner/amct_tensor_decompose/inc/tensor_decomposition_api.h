/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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
 * @brief interface of estimate rank for tensor decomposition.
 *
 * @file tensor_decomposition_api.h
 *
 * @version 1.0
 */


#ifndef CPP_TEST_TENSOR_DECOMPOSITION_API_H
#define CPP_TEST_TENSOR_DECOMPOSITION_API_H

#include "tensor_decomposition.h"

extern "C" {
int GetRank(TensorDecompose::ConvInfo info, const double *s, unsigned int length);
TensorDecompose::DecomposeMode FastFilterConv(TensorDecompose::ConvInfo info);
}
#endif // CPP_TEST_TENSOR_DECOMPOSITION_API_H
