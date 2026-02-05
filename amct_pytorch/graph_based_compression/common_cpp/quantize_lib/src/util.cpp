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

namespace {
constexpr unsigned int NUM_PRECISION = 10;
constexpr int RECORD_RESERVED_NUM = 16;
constexpr int ONNX_FLOAT16_ENUM = 10;

template <typename T>
std::string Format(T data)
{
    std::ostringstream stringStream;
    if (std::is_same<T, float>::value) {
        stringStream << std::setw(RECORD_RESERVED_NUM) << data;
    } else {
        stringStream << data;
    }
    return stringStream.str();
}
}

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

    template <typename T>
    Status RecordScaleOffsetData(int fd, unsigned char* buffer, unsigned long size, const std::string &layerName,
        const RecordData<T> &recordData)
    {
        // Actual function is insert scale and offset after key word
        std::unique_ptr<unsigned char[]> tmpBufUptr(new (std::nothrow) unsigned char[size]);
        if (tmpBufUptr == nullptr) {
            LOG_ERROR("Failed to allocate memory.");
            return AmctCommon::RECORD_FACTOR_ERROR;
        }

        std::string keyWord = "key: \"" + layerName + "\"\n  value {\n";

        // trans float/int data to string
        std::stringstream buf;
        (void)buf.precision(NUM_PRECISION);
        buf << recordData.scale;

        std::string targetWord = "";
        if (recordData.dataType == "data") {
            targetWord += "    scale_d: " + buf.str() + "\n";
            targetWord += "    offset_d: " + std::to_string(recordData.offset) + "\n";
            targetWord += "    act_type: \"INT" + std::to_string(recordData.numBits) + "\"\n";
        } else if (recordData.dataType == "initial_h") {
            targetWord += "    scale_h: " + buf.str() + "\n";
            targetWord += "    offset_h: " + std::to_string(recordData.offset) + "\n";
        } else {
            targetWord += "    scale_w: " + buf.str() + "\n";
            targetWord += "    offset_w: " + std::to_string(recordData.offset) + "\n";
            targetWord += "    wts_type: \"INT" + std::to_string(recordData.numBits) + "\"\n";
        }
        if (recordData.opDtype != 0) {
            std::string opDtypeStr(recordData.opDtype == ONNX_FLOAT16_ENUM ? "FLOAT16" : "FLOAT32");
            targetWord += "    op_data_type: \'" + opDtypeStr + "\'\n";
        }
        if (recordData.fakequantPrecisionMode != "DEFAULT") {
            targetWord += "    fakequant_precision_mode: \'" + recordData.fakequantPrecisionMode + "\'\n";
        }
        std::string inputStr = reinterpret_cast<char*>(buffer);
        std::string::size_type subStrPos = inputStr.find(keyWord);
        if (subStrPos == std::string::npos) {
            LOG_ERROR("Cannot find \"%s\" in record file.\n", layerName.c_str());
            return AmctCommon::RECORD_FACTOR_ERROR;
        }

        auto index = static_cast<unsigned long>(subStrPos);
        unsigned long insertIndex = index + keyWord.size();
        // Locate key world +1 location to read left data to tmp buffer
        (void)lseek(fd, static_cast<long>(insertIndex), SEEK_SET);
        int retVal = read(fd, tmpBufUptr.get(), static_cast<unsigned int>(size - insertIndex));
        CHECK_RECORD_FILE(retVal, "Failed to read file.");

        // write left data back to fd, but left target word length
        (void)lseek(fd, static_cast<long>(insertIndex + targetWord.size()), SEEK_SET);
        retVal = write(fd, tmpBufUptr.get(), static_cast<unsigned int>(size - insertIndex));
        CHECK_RECORD_FILE(retVal, "Failed to write file.");

        // back to insert location, then write target world into fd
        (void)lseek(fd, static_cast<long>(insertIndex), SEEK_SET);
        retVal = write(fd, targetWord.c_str(), static_cast<unsigned int>(targetWord.size()));
        CHECK_RECORD_FILE(retVal, "Failed to write file.");
        return AmctCommon::SUCCESS;
    }

    template <typename T>
    Status RecordScaleOffsetKernel(const std::string &fileName, const std::string &layerName,
        const RecordData<T> &recordData)
    {
        char path[PATH_MAX + 1] = {0x00};
        if ((strlen(fileName.c_str()) > PATH_MAX) || (realpath(fileName.c_str(), path) == nullptr)) {
            LOG_ERROR("Invalid record file path.\n");
            return AmctCommon::RECORD_FILE_ERROR;
        }
        int fd = open(path, O_RDWR);
        if (fd < 0) {
            LOG_ERROR("Failed to open file.\n");
            return AmctCommon::RECORD_FILE_ERROR;
        }
        struct stat sb{};
        int retVal = fstat(fd, &sb);
        if (retVal < 0) {
            LOG_ERROR("Failed to stat file.\n");
            (void)close(fd);
            return AmctCommon::RECORD_FILE_ERROR;
        }

        auto fileSize = static_cast<unsigned long>(sb.st_size);
        if (fileSize == 0) {
            LOG_ERROR("file(%s) size is zero.\n", fileName.c_str());
            (void)close(fd);
            return AmctCommon::RECORD_FILE_ERROR;
        }
        // last byte store '\0'
        std::unique_ptr<unsigned char[]> tmpStartptr(new (std::nothrow) unsigned char[fileSize + 1]);
        if (tmpStartptr == nullptr) {
            LOG_ERROR("Failed to allocate memory.");
            (void)close(fd);
            return AmctCommon::RECORD_FACTOR_ERROR;
        }
        tmpStartptr.get()[fileSize] = 0;
        retVal = read(fd, tmpStartptr.get(), fileSize);
        if (retVal < 0) {
            LOG_ERROR("Failed to read file.\n");
            (void)close(fd);
            return AmctCommon::RECORD_FILE_ERROR;
        }

        Status status = RecordScaleOffsetData(fd, tmpStartptr.get(), fileSize, layerName, recordData);
        retVal = close(fd);
        if (retVal < 0) {
            LOG_ERROR("Failed to close file.\n");
            return AmctCommon::RECORD_FILE_ERROR;
        }
        if (status != 0) {
            LOG_ERROR("Write scale_d and offset_d to \"%s\" failed.\n", layerName.c_str());
            return status;
        }
        return AmctCommon::SUCCESS;
    }

    template <typename T>
    Status RecordScaleOffset(const std::string &fileName, const std::string &layerName,
        const RecordData<T> &recordData)
    {
        return RecordScaleOffsetKernel(fileName, layerName, recordData);
    }

    template Status RecordScaleOffset(const std::string &fileName, const std::string &layerName,
        const RecordData<int> &recordData);

    Status GetLengthByDim(const std::vector<int>& dims, int64_t& dataSize)
    {
        dataSize = 1;
        if (dims.empty()) {
            dataSize = 0;
            return AmctCommon::SUCCESS;
        }

        size_t dimSize = dims.size();
        for (size_t i = 0; i < dimSize; i++) {
            int dim = dims[i];
            if (dataSize > (LONG_MAX / dim)) {
                LOG_ERROR("Overflow is detected during calculating dataSize!");
                return AmctCommon::BAD_PARAMETERS_ERROR;
            }
            dataSize *= dims[i];
        }
        return AmctCommon::SUCCESS;
    }

    template <typename T>
    Status RecordRepeatDataKernel(int fd, const unsigned char* buffer, unsigned long size, const char* layerName,
        const RecordData<T> &recordData)
    {
        // Actual function is insert scale and offset after key word
        std::unique_ptr<unsigned char[]> tmpBufUptr(new (std::nothrow) unsigned char[size]);
        if (tmpBufUptr == nullptr) {
            LOG_ERROR("Failed to allocate memory.");
            return AmctCommon::RECORD_FACTOR_ERROR;
        }

        std::string keyWord = "key: \"";
        (void)keyWord.append(layerName).append("\"\n  value {\n");

        const size_t nSize = recordData.data.size();
        const std::string typeStr = "    " + recordData.dataType + ": ";
        std::string targetWord = "";
        for (size_t i = 0; i < nSize; i++) {
            (void)targetWord.append(typeStr).append(Format(recordData.data[i])).append("\n");
        }

        std::string inputStr = reinterpret_cast<const char*>(buffer);
        std::string::size_type subStrPos = inputStr.find(keyWord);
        if (subStrPos == std::string::npos) {
            LOG_ERROR("Cannot find \"%s\" in record file.\n", layerName);
            return AmctCommon::RECORD_FACTOR_ERROR;
        }

        auto index = static_cast<unsigned long>(subStrPos);
        unsigned long insertIndex = index + keyWord.size();
        // Locate key world +1 location to read left data to tmp buffer
        if (lseek(fd, static_cast<long>(insertIndex), SEEK_SET) < 0) {
            LOG_ERROR("Failed to reposition read file offset.");
            return AmctCommon::RECORD_FACTOR_ERROR;
        }
        int retVal = 0;
        // tensor_balance_factor need to overwrite last data
        if (recordData.dataType == "tensor_balance_factor") {
            std::string overWriteWord = keyWord + "    " + recordData.dataType;
            subStrPos = inputStr.find(overWriteWord);
            if (subStrPos != std::string::npos) {
                retVal = write(fd, targetWord.c_str(), static_cast<unsigned int>(targetWord.size()));
                CHECK_RECORD_FILE(retVal, "Failed to write file.");
                return AmctCommon::SUCCESS;
            }
        }
        retVal = read(fd, tmpBufUptr.get(), static_cast<unsigned int>(size - insertIndex));
        CHECK_RECORD_FILE(retVal, "Failed to read file.");

        // write left data back to fd, but left target word length
        if (lseek(fd, static_cast<long>(insertIndex + targetWord.size()), SEEK_SET) < 0) {
            LOG_ERROR("Failed to reposition write file offset.");
            return AmctCommon::RECORD_FACTOR_ERROR;
        }
        retVal = write(fd, tmpBufUptr.get(), static_cast<unsigned int>(size - insertIndex));
        CHECK_RECORD_FILE(retVal, "Failed to write file.");
        // back to insert location, then write target world into fd
        if (lseek(fd, static_cast<long>(insertIndex), SEEK_SET) < 0) {
            LOG_ERROR("Failed to reposition write file offset.");
            return AmctCommon::RECORD_FACTOR_ERROR;
        }
        retVal = write(fd, targetWord.c_str(), static_cast<unsigned int>(targetWord.size()));
        CHECK_RECORD_FILE(retVal, "Failed to write file.");
        return AmctCommon::SUCCESS;
    }

    template <typename T>
    Status RecordRepeatData(const std::string &fileName, const std::string &layerName, const std::vector<T> &data,
        const std::string &dataType)
    {
        // open file
        char path[PATH_MAX + 1] = {0x00};
        if ((strlen(fileName.c_str()) > PATH_MAX) || (realpath(fileName.c_str(), path) == nullptr)) {
            LOG_ERROR("Invalid record file path.\n");
            return AmctCommon::RECORD_FILE_ERROR;
        }
        int shiftnFd = open(path, O_RDWR);
        if (shiftnFd < 0) {
            LOG_ERROR("Failed to open file.\n");
            return AmctCommon::RECORD_FILE_ERROR;
        }
        struct stat sb{};
        int retVal = fstat(shiftnFd, &sb);
        if (retVal < 0) {
            LOG_ERROR("Failed to stat file.\n");
            (void)close(shiftnFd);
            return AmctCommon::RECORD_FILE_ERROR;
        }

        auto fileSize = static_cast<unsigned long>(sb.st_size);
        // last byte store '\0'
        std::unique_ptr<unsigned char[]> tmpStartptr(new (std::nothrow) unsigned char[fileSize + 1]);
        if (tmpStartptr == nullptr) {
            LOG_ERROR("Failed to allocate memory.");
            (void)close(shiftnFd);
            return AmctCommon::RECORD_FACTOR_ERROR;
        }
        tmpStartptr.get()[fileSize] = 0;
        retVal = read(shiftnFd, tmpStartptr.get(), fileSize);
        if (retVal < 0) {
            LOG_ERROR("Failed to read file.\n");
            (void)close(shiftnFd);
            return AmctCommon::RECORD_FILE_ERROR;
        }

        // Record shift_n
        RecordData<T> recordData {0, 0, data, dataType, 0, 0, "DEFAULT"};
        Status status = RecordRepeatDataKernel(shiftnFd, tmpStartptr.get(), fileSize, layerName.c_str(),
            recordData);
        retVal = close(shiftnFd);
        if (retVal < 0) {
            LOG_ERROR("Failed to close file.\n");
            return AmctCommon::RECORD_FILE_ERROR;
        }
        if (status != 0) {
            LOG_ERROR("Write %s to \"%s\" failed.\n", dataType.c_str(), layerName.c_str());
            return status;
        }
        return AmctCommon::SUCCESS;
    }

    template Status RecordRepeatData(const std::string &fileName, const std::string &layerName,
        const std::vector<int> &data, const std::string &dataType);

    template Status RecordRepeatData(const std::string &fileName, const std::string &layerName,
        const std::vector<float> &data, const std::string &dataType);

    Status CheckBalanceFactor(const float* balanceFactor, unsigned int channelNum)
    {
        for (unsigned int i = 0; i < channelNum; i++) {
            if ((balanceFactor[i] < FLT_EPSILON) || (balanceFactor[i] > (1 / FLT_EPSILON))) {
                LOG_ERROR("tensor balance factor out of range[FLT_EPSILON, 1/FLT_EPSILON]: %f.\n", balanceFactor[i]);
                return AmctCommon::TENSOR_BALANCE_FACTOR_ERROR;
            }
        }
        return AmctCommon::SUCCESS;
    }
}
