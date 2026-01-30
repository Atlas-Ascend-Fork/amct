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
 * @brief arq head file
 *
 * @file arq.h
 *
 * @version 1.0
 */

#ifndef DUMP_H
#define DUMP_H

#include <string>
#include <vector>
#include <cstdint>

namespace AmctCommon {
// Define the structure of data quantification
struct DumpParam {
    std::string fileName;
    std::vector<float> dataShape;
    uint dataShapeLength;
};

template <class T>
void DumpData(const T* inputDataArray, int dataLen, struct DumpParam dumpParam);

void DumpDataWithType(const float* inputDataArray, int dataLen, struct DumpParam dumpParam);
void DumpDataWithType(const double* inputDataArray, int dataLen, struct DumpParam dumpParam);
void DumpDataWithType(const int* inputDataArray, int dataLen, struct DumpParam dumpParam);
void DumpDataWithType(const uint16_t* inputDataArray, int dataLen, struct DumpParam dumpParam);
} // namespace AmctCommon

#endif // DUMP_H
