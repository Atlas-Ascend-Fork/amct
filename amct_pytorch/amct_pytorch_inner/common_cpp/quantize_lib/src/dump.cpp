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

#include "dump.h"
#include <string>
#include <fstream>
#include "util.h"

namespace AmctCommon {
constexpr float FLOAT_INDEX = 0;
constexpr float DOUBLE_INDEX = 1;
constexpr float INT_INDEX = 2;
constexpr float HALF_INDEX = 3;

template <class T>
void DumpData(const T* inputDataArray, int dataLen, struct DumpParam dumpParam)
{
    CHECK_TRUE_RETURN_WITH_LOG(inputDataArray == nullptr, "input ptr is null.\n");
    std::ofstream outShape(dumpParam.fileName.c_str(), std::ios::binary);
    CHECK_TRUE_RETURN_WITH_LOG(!outShape.is_open(), "AmctCommon::dump_data write outShape fail to open file.\n");
    (void)outShape.write(reinterpret_cast<char*>(dumpParam.dataShape.data()),
        static_cast<long>(sizeof(float) * dumpParam.dataShapeLength));
    outShape.close();
    std::ofstream outData(dumpParam.fileName.c_str(), std::ios::binary | std::ios::app);
    CHECK_TRUE_RETURN_WITH_LOG(!outData.is_open(), "AmctCommon::dump_data write outData fail to open file.\n");
    (void)outData.write(reinterpret_cast<const char*>(inputDataArray), static_cast<long>(sizeof(T) * dataLen));
    outData.close();
};

template void DumpData(const float*, int, struct DumpParam);
template void DumpData(const double*, int, struct DumpParam);
template void DumpData(const int*, int, struct DumpParam);
template void DumpData(const uint16_t*, int, struct DumpParam);

void DumpDataWithType(const float* inputDataArray, int dataLen, struct DumpParam dumpParam)
{
    (void)dumpParam.dataShape.insert(dumpParam.dataShape.cbegin(), FLOAT_INDEX);
    dumpParam.dataShapeLength += 1;
    DumpData(inputDataArray, dataLen, dumpParam);
}

void DumpDataWithType(const double* inputDataArray, int dataLen, struct DumpParam dumpParam)
{
    (void)dumpParam.dataShape.insert(dumpParam.dataShape.cbegin(), DOUBLE_INDEX);
    dumpParam.dataShapeLength += 1;
    DumpData(inputDataArray, dataLen, dumpParam);
}

void DumpDataWithType(const int* inputDataArray, int dataLen, struct DumpParam dumpParam)
{
    (void)dumpParam.dataShape.insert(dumpParam.dataShape.cbegin(), INT_INDEX);
    dumpParam.dataShapeLength += 1;
    DumpData(inputDataArray, dataLen, dumpParam);
}

void DumpDataWithType(const uint16_t* inputDataArray, int dataLen, struct DumpParam dumpParam)
{
    (void)dumpParam.dataShape.insert(dumpParam.dataShape.cbegin(), HALF_INDEX);
    dumpParam.dataShapeLength += 1;
    DumpData(inputDataArray, dataLen, dumpParam);
}
} // namespace AmctCommon
