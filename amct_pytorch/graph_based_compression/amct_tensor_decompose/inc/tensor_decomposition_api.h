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
