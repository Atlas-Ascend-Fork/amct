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
