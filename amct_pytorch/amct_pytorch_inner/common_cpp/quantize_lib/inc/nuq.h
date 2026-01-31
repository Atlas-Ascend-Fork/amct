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
 * @brief nuq header file
 *
 * @file nuq.h in common_cpp
 *
 * @version 1.0
 */

#ifndef NUQ_H
#define NUQ_H

#include "util.h"

namespace AmctCommon {
constexpr int MAX_ITER = 200;

/**
 * @ingroup quantize lib
 * @brief Param for NUQ
 */
struct NuqParam {
    unsigned int numSteps;
    bool withOffset;
    unsigned int numIter;
};

struct NuqAlgoParam {
    int scaleLength;
    int numSteps;
    int numOfIteration;
    bool withOffset;
};

// Define the structure of nuq cuda data
template <typename T>
struct ThrustParam {
    T* clusterToStorePtr;
    T* clusterDataPtr;
    T* clusterCenterPtr;
    T* weightPtr;
    unsigned int* dataClusterPtr;
    T* sumClusterPtr;
    T* weightClusterPtr;
    T* weightDataPtr;
    T* tmpValueDataPtr;
    T* deviceInputPtr;
    T* ddistancePtr;
    T* dcandiDistPtr;
    T* dcumulativeDistancePtr;
    int* canDataPtr;
    int numCandidate;
    int dataLength;
    int numSteps;
};


/**
 * @ingroup quantize lib
 * @brief: common c++ Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @param [out] int8Data: output int8 data
 * @return [out] success/fail
 */
template <class T>
Status NuqKernel(T* data, unsigned int length, const NuqParam &nuqParam,
    const util::FloatData &scale, const util::IntData &offset);

/**
 * @ingroup quantize lib
 * @brief: Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] nuqParam: nuq parameters
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @return [out] success/fail
 */
Status NuqQuant(float* data, unsigned int length, const AmctCommon::NuqParam &nuqParam,
    const util::FloatData &scale, const util::IntData &offset);

template <class T>
Status NuqCalibrationCpuKernel(const T* data, std::vector<T>& clusterData, std::vector<T>& clusterToStore,
    unsigned int length, const NuqParam &nuqParam, const util::FloatData& scale, const util::IntData& offset);


template <typename T>
Status NuqCalibrationCudaKernel(T* data, int length, struct NuqAlgoParam& quantParam,
    float* scale, int* offset, struct ThrustParam<T>& thrustParam, std::vector<T>& clusterToStore);

template <class T>
T Median(std::vector<T> data);
}

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @ingroup quantize lib
 * @brief: Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] nuqParam: nuq parameters
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @return [out] success/fail
 */
int NuqQuantPython(double* data, unsigned int length, AmctCommon::NuqParam nuqParam,
    util::FloatData scale, util::IntData offset);

/**
 * @ingroup quantize lib
 * @brief: Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @param [out] int8Data: output int8 data
 * @return [out] success/fail
 */
int NuqQuantRealPython(double* data, unsigned int length,
    util::FloatData scale, util::IntData offset, char* int8Data);

/**
 * @ingroup quantize lib
 * @brief: Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] nuqParam: nuq parameters
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @return [out] success/fail
 */
int NuqQuantPythonGPU(double* data, unsigned int length, AmctCommon::NuqParam nuqParam,
    util::FloatData scale, util::IntData offset);

/**
 * @ingroup quantize lib
 * @brief: Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @param [out] int8Data: output int8 data
 * @return [out] success/fail
 */
int NuqQuantRealPythonGPU(double* data, unsigned int length,
    util::FloatData scale, util::IntData offset, char* int8Data);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* NUQ_H */
