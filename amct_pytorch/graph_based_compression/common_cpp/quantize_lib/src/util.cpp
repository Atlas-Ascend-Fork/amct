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
 * @brief arq_quant algorithm custom op C++ implement
 *
 * @file arq_quant.cpp
 *
 * @version 1.0
 */

#include "util.h"
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <climits>
#include <memory>
#include <string>
#include <sstream>
#include <iomanip>
#include <fcntl.h>    /* For O_RDWR */
#include <unistd.h>
#include <sys/stat.h>
#include <linux/limits.h>

namespace util {
    Status ProcessScale(float& currentScale)
    {
        if (currentScale < FLT_EPSILON) {
            currentScale = 1.0;
        }
        if (std::isinf(currentScale)) {
            LOG_ERROR("Not support scale is +inf.\n");
            return AmctCommon::NOT_SUPPORT_ERROR;
        }
        if (std::isnan(currentScale)) {
            LOG_ERROR("Not support scale is nan.\n");
            return AmctCommon::NOT_SUPPORT_ERROR;
        }
        if ((1 / currentScale) < FLT_EPSILON) {
            LOG_ERROR("Not support scale greater than 1 / FLT_EPSILON. currentScale: %f\n", currentScale);
            return AmctCommon::NOT_SUPPORT_ERROR;
        }
        return AmctCommon::SUCCESS;
    }

    Status ProcessScale(double& currentScale)
    {
        if (currentScale < DBL_EPSILON) {
            currentScale = 1.0;
        }
        if (std::isinf(currentScale)) {
            LOG_ERROR("Not support scale is +inf.\n");
            return AmctCommon::NOT_SUPPORT_ERROR;
        }
        if (std::isnan(currentScale)) {
            LOG_ERROR("Not support scale is nan.\n");
            return AmctCommon::NOT_SUPPORT_ERROR;
        }
        if ((1 / currentScale) < DBL_EPSILON) {
            LOG_ERROR("Not support scale greater than 1 / DBL_EPSILON.\n");
            return AmctCommon::NOT_SUPPORT_ERROR;
        }
        return AmctCommon::SUCCESS;
    }
}
