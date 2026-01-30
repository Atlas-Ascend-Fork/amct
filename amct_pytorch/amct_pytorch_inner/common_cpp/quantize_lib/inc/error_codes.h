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
 * @brief error_codes head file
 *
 * @file error_codes.h in common_cpp
 *
 * @version 1.0
 */

#ifndef ERROR_CODES_H
#define ERROR_CODES_H
/**
 * @ingroup quantize lib
 * @brief: error code.
 */

namespace AmctCommon {
constexpr int SUCCESS = 0; // 0x00000000
constexpr int GENERIC_ERROR = -65536; // 0xFFFF0000
constexpr int BAD_FORMAT_ERROR = -65531; // 0xFFFF0005
constexpr int BAD_PARAMETERS_ERROR = -65530; // 0xFFFF0006
constexpr int OUT_OF_MEMORY_ERROR = -65524; // 0xFFFF000C
constexpr int SHORT_BUFFER_ERROR = -65520; // 0xFFFF0010
constexpr int NOT_SUPPORT_ERROR = -65519; // 0xFFFF0011
constexpr int CUDA_ERROR = -65518; // 0xFFFF0012
constexpr int INDEX_OUT_OF_RANGE_ERROR = -65517; // 0xFFFF0013

constexpr int RECORD_FILE_ERROR = -65529; // 0xFFFF0007;
constexpr int RECORD_FACTOR_ERROR = -65528; // 0xFFFF0008;

constexpr int CUDA_MEMCPY_ERROR = 1;
constexpr int DEQ_SCALE_ERROR = 2;
constexpr int INPUT_CHANNEL_ERROR = 3;
constexpr int SCALED_NOT_EXIT_ERROR = 4;
constexpr int RECORD_NOT_EXIT_ERROR = 5;
constexpr int RECORD_FILE_OPEN_ERROR = 6;
constexpr int SCALE_LENGTH_ERROR = 7;
constexpr int ZERO_DIVISION_ERROR = 8;
constexpr int SCALEW_ERROR = 9;
constexpr int SCALED_ERROR = 10;
constexpr int TRAINING_ERROR = 11;
constexpr int CUDA_ASYNC_ERROR = 12;
constexpr int NULL_PTR_ERROR = 13;
constexpr int RECORD_FILE_PARSE_ERROR = 14;
constexpr int CONTAINER_EMPTY_ERROR = 15;
constexpr int NON_INIT_ERROR = 16;
constexpr int TENSOR_BALANCE_FACTOR_ERROR = 17;
}

#define CUDA_MEMCPY_ERROR_CHECK(error)               \
    do {                                             \
        if (error != cudaSuccess) {                  \
            return AmctCommon::CUDA_MEMCPY_ERROR;    \
        }                                            \
    } while (0)

#define CHECK_OK(error)                              \
    do {                                             \
        if (error != 0) {                            \
            return error;                            \
        }                                            \
    } while (0)

#define NULLPTR_CHECK(ptr)                           \
    do {                                             \
        if (ptr == nullptr) {                        \
            return AmctCommon::NULL_PTR_ERROR;       \
        }                                            \
    } while (0)

#endif /* ERROR_CODES_H */
