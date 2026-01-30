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
 * @brief error code and logging tools
 *
 * @file td_log.h
 *
 * @version 1.0
 */

#ifndef TD_LOG_H
#define TD_LOG_H

namespace TensorDecompose {
enum class TdError {
    TD_SUCCESS                   = static_cast<int>(0x00000000),
    TD_GENERIC_ERR               = static_cast<int>(0xFFF00000),
    TD_BAD_FORMAT_ERR            = static_cast<int>(0xFFF00001),
    TD_BAD_PARAMETERS_ERR        = static_cast<int>(0xFFF00002),
    TD_OUT_OF_MEMORY_ERR         = static_cast<int>(0xFFF00003),
    TD_SHORT_BUFFER_ERR          = static_cast<int>(0xFFF00004),
    TD_NOT_SUPPORT_ERR           = static_cast<int>(0xFFF00005),
    TD_CUDA_ERR                  = static_cast<int>(0xFFF00006),
    TD_GENERIC_MATH_ERR          = static_cast<int>(0xFFF00007),
    TD_MEM_OPERATION_ERR         = static_cast<int>(0xFFF00008),
    TD_IDX_OUT_OF_BOUNDS_ERR     = static_cast<int>(0xFFF00009),
    TD_NULL_DATA_ERR             = static_cast<int>(0xFFF0000A),
};

#define TD_FUNC_CHECK(func) do { TdError ret = func; \
                                if (ret != TdError::TD_SUCCESS) {  \
                                return ret; \
                                } \
                            } while (0)

#define TD_NULLPTR_CHECK(ptr) do {if (ptr == nullptr) { \
                                    return TdError::TD_NULL_DATA_ERR; \
                                } \
                            } while (0)

#define TD_NULLPTR_CHECK_DOUBLE(ptr1, ptr2) do { \
                                if (ptr1 == nullptr || ptr2 == nullptr) { \
                                    return TdError::TD_NULL_DATA_ERR; \
                                } \
                            } while (0)

#define TD_NULLPTR_CHECK_TRIPLE(ptr1, ptr2, ptr3) do { \
                                if (ptr1 == nullptr || ptr2 == nullptr || ptr3 == nullptr) { \
                                    return TdError::TD_NULL_DATA_ERR; \
                                } \
                            } while (0)

#define TD_CHECK_NORMAL_ZERO_LENGTH(length) do {if (length == 0) { \
                                    return TdError::TD_SUCCESS; \
                                } \
                            } while (0)
}

#endif /* TD_LOG_H */
