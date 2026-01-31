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
 * @brief dmq_balance header file
 *
 * @file dmq_balance.h in common_cpp
 *
 * @version 1.0
 */

#ifndef DMQ_BALANCE_H
#define DMQ_BALANCE_H

#include "util.h"

namespace AmctCommon {
struct InputDataParam {
    void* in;
    int64_t inType;
    size_t length;
};

Status CheckDMQBParam(const util::FloatData &act, const util::FloatData &wts, float migrationStrength,
    uint32_t channelNum, const float *balanceFactor);

Status DMQBalance(const util::FloatData &act, const util::FloatData &wts, float migrationStrength,
    uint32_t channelNum, float *balanceFactor);

Status DMQBalanceGpu(const util::FloatData &act, const util::FloatData &wts, float migrationStrength,
    unsigned int channelNum, float *balanceFactor);

Status DMQBalanceGpuMemCopy(InputDataParam inputAct, InputDataParam inputWts, float migrationStrength,
    unsigned int channelNum, float *balanceFactor);
}

#endif
