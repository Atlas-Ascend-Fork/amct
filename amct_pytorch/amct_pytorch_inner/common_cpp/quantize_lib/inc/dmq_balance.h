/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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
