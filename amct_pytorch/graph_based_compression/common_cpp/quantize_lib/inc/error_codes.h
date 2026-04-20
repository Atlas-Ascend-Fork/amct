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
constexpr int BAD_PARAMETERS_ERROR = -65530; // 0xFFFF0006
constexpr int NOT_SUPPORT_ERROR = -65519; // 0xFFFF0011
constexpr int RECORD_FACTOR_ERROR = -65528; // 0xFFFF0008;
}

#endif
