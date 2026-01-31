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
