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
constexpr unsigned int NUM_BITS_QUANT = 8;
constexpr unsigned int BINARY_BASE = 2;

constexpr unsigned int MIN_SHIFT_BIT = 1;
constexpr unsigned int MAX_SHIFT_BIT = 16;
constexpr unsigned int MAX_INT = 2147483647;

constexpr float EPSILON = 1e-6f;
constexpr int BASE = 2;
constexpr float ONE_HALF = 0.5f;
constexpr int MINUS_ONE = -1;
constexpr int POWER_OF_SQUARE = 2;
constexpr int POWER_OF_CUBE = 3;
constexpr int EIGHT = 8;
constexpr int FOUR = 4;
constexpr int SIXTEEN = 16;
constexpr int SHIFT_POW = 15;
constexpr int SHIFT_BITS = 16;
constexpr int NCHW_N_DIM = 0;
constexpr int NCHW_C_DIM = 1;
constexpr int NCHW_H_DIM = 2;
constexpr int NCHW_W_DIM = 3;
constexpr int CFIRST_N_DIM = 0;
constexpr int CFIRST_C_DIM = 1;
constexpr int CFIRST_H_OFFSET = 2;
constexpr int NHWC_N_DIM = 0;
constexpr int NHWC_H_DIM = 1;
constexpr int NHWC_W_DIM = 2;
constexpr int NHWC_C_DIM = 3;
constexpr int CLAST_H_OFFSET = 3;
constexpr int CLAST_W_OFFSET = 2;
constexpr int HWCN_H_DIM = 0;
constexpr int HWCN_W_DIM = 1;
constexpr int HWCN_C_DIM = 2;
constexpr int HWCN_N_DIM = 3;
constexpr int NC_N_DIM = 0;
constexpr int NC_C_DIM = 1;
constexpr int FOUR_DIM = 4;
constexpr int AMCT_CUDA_NUM_THREADS = 512;
constexpr int DEQ_SCALE_BINS = 32;
constexpr int N_LFET_BINS = 24;
constexpr int N_RIGHT_BINS = 56;
constexpr int WEIGHT_SHAPE_CIN_OFFSET = 2;
constexpr int COUT_ALIGN = 16;
constexpr int BINARY = 2;
constexpr int EVEN = 2;
constexpr float EPS = 1e-6f;
//min fp16 is 2^-14, 1/16384
constexpr float MIN_FP16 = 1.0f/16384.0f;
constexpr float MAX_FP16 = 65504.0f;
constexpr float DENORMAL_FP16 = 1.0f/33554432.0f;
constexpr int FORCE_FP16_QUANT = 1;
constexpr int ROUND = 0;
constexpr int HYBRID = 1;
constexpr int FP32 = 0;
constexpr int FP16 = 1;
constexpr int BF16 = 2;
constexpr int HIF8 = 3;
constexpr int FP8 = 4;

#define CHECK_INT_OVERFLOW(data) \
        if (data > util::MAX_INT) { \
            LOG(ERROR) << "int overflow."; \
        } \

#define CHECK_RECORD_FILE(status, msg) \
        if (status < 0) { \
            LOG_ERROR(msg); \
            return AmctCommon::RECORD_FACTOR_ERROR; \
        } \

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)


#define CHECK_TRUE_RETURN_WITH_LOG(condition, ...)            \
    do {                                                      \
        if (condition) {                                      \
            LOG_ERROR(__VA_ARGS__);                           \
            return;                                           \
        }                                                     \
    } while (0)

// CUDA: number of blocks for threads.
inline int AMCT_GET_BLOCKS(const int num)
{
    return (num + AMCT_CUDA_NUM_THREADS - 1) / AMCT_CUDA_NUM_THREADS;
}

#define RAW_PRINTF (void)printf
#define LOG_ERROR(fmt, arg...) RAW_PRINTF("[ERROR][%s][%d] " fmt, __FUNCTION__, __LINE__, ## arg)

/**
 * @ingroup quantize lib
 * @brief: float data array and length.
 */
struct FloatData {
    unsigned int length;
    float* data;
};

/**
 * @ingroup quantize lib
 * @breif: clip index and quant bits.
 */
struct ClipInfo {
    unsigned int index;
    unsigned int quantBits;
};

/**
 * @ingroup quantize lib
 * @brief: int data array and length.
 */
struct IntData {
    unsigned int length;
    int* data;
};

/**
 * @ingroup quantize lib
 * @brief: quant factors: scale, offset
 */
struct QuantFactors {
    FloatData scale;
    IntData offset;
};

/**
 * @ingroup quantize lib
 * @brief: record data: scale, offset, repeat data, data or weight, opDtype, numBits
 */
template <typename T>
struct RecordData {
    float scale;
    int offset;
    std::vector<T> data;
    std::string dataType;
    int opDtype;
    unsigned int numBits;
    std::string fakequantPrecisionMode;
};

enum WeightFormat {
    CO_CI_KH_KW = 0,
    KH_KW_CI_CO = 1,
};

const std::vector<std::string> FAKE_QUANT_PRECISION_MODES = {
    "DEFAULT",
    "FORCE_FP16_QUANT"
};

// CPUDevice and GPUDevice
struct ThreadPoolDevice {
};

struct GpuDevice {
};

using CPUDevice = ThreadPoolDevice;
using GPUDevice = GpuDevice;

Status ProcessScale(float& currentScale);
Status ProcessScale(double& currentScale);
Status GetLengthByDim(const std::vector<int>& dims, int64_t& dataSize);
Status CheckBalanceFactor(const float* balanceFactor, unsigned int channelNum);


    /**
    * @ingroup quantize lib
    * @brief: record scale offset Function.
    * @param [in] fileName: record file name.
    * @param [in] layerName: target layer name.
    * @param [in|out] recordData: to record data.
    * @return succ/fail
    */
    template <typename T>
    Status RecordScaleOffset(const std::string &fileName, const std::string &layerName,
        const RecordData<T> &recordData);

    template <typename T>
    Status RecordRepeatData(const std::string &fileName, const std::string &layerName, const std::vector<T> &data,
        const std::string &dataType);
}

#endif /* UTIL_H */
