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
 * @file tensor_decomposition_api.cpp
 *
 * @version 1.0
 */

#include "tensor_decomposition_api.h"
#include <algorithm>
#include "vector.h"

using namespace std;
using namespace TensorDecompose;

extern "C" {
int GetRank(ConvInfo info, const double *s, unsigned int length)
{
    if (length == 0) {
        return 0;
    }

    Vector vecS;
    TdError vecRet = vecS.Create(s, length);
    if (vecRet !=TdError::TD_SUCCESS) {
        return length;
    }
    int res = TensorDecomposition::Estimation(info, vecS, length);
    return res;
}


DecomposeMode FastFilterConv(ConvInfo info)
{
    int minChannel = 64;
    int channelRound = 16;
    int ksThreshold = 7;
    if (info.dilationH > 1 || info.dilationW > 1 || info.group > 1 || // if dilation or group > 1 or minimum
        min(info.kernelSizeH, info.kernelSizeW) < 3 ||                // kernel_size < 3, or maximum
        max(info.strideH, info.strideW) > 2 ||                        // stride > 2, do not decompose
        min(info.kernelSizeH, info.kernelSizeW) * min(info.inChannel, info.outChannel) < channelRound ||
        (min(info.kernelSizeH, info.kernelSizeW) < ksThreshold && min(info.inChannel, info.outChannel) < minChannel)) {
        return DecomposeMode::DM_UNCHANGE;
    }

    if (info.outChannel < info.inChannel) {
        if (info.kernelSizeH <= info.kernelSizeW) {
            return DecomposeMode::DM_FIRST_CHANNEL_FIRST_KERNEL;
        } else {
            return DecomposeMode::DM_FIRST_CHANNEL_SECOND_KERNEL;
        }
    } else {
        if (info.kernelSizeH <= info.kernelSizeW) {
            return DecomposeMode::DM_SECOND_CHANNEL_FIRST_KERNEL;
        } else {
            return DecomposeMode::DM_SECOND_CHANNEL_SECOND_KERNEL;
        }
    }
}
}
