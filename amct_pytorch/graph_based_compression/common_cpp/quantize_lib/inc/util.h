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
 * @brief util head file
 *
 * @file util.h in common_cpp
 *
 * @version 1.0
 */

#ifndef UTIL_H
#define UTIL_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdint>

#include "error_codes.h"

using Status = int;

namespace util {
#define CHECK_TRUE_RETURN_WITH_LOG(condition, ...)            \
    do {                                                      \
        if (condition) {                                      \
            LOG_ERROR(__VA_ARGS__);                           \
            return;                                           \
        }                                                     \
    } while (0)

#define RAW_PRINTF (void)printf
#define LOG_ERROR(fmt, arg...) RAW_PRINTF("[ERROR][%s][%d] " fmt, __FUNCTION__, __LINE__, ## arg)
}

#endif /* UTIL_H */
