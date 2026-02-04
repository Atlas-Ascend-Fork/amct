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
 * @brief ARQ algorithm C++ implementation for CPU.
 *
 * @file arq.cpp
 *
 * @version 1.0
 */
#include "arq.h"

#include <cstdio>
#include <cfloat>
#include <cmath>

using namespace util;

namespace {
constexpr unsigned int MIN_NUM_BITS = 2;
}

namespace AmctCommon {

template <class T>
static Status CheckArqParams(const T* data, unsigned int length, const ArqParam &arqParam,
    const QuantFactors &factor, uint32_t group)
{
    if (data == nullptr) {
        LOG_ERROR("Empty pointer\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (length == 0) {
        LOG_ERROR("input data length = 0.\n");
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    for (unsigned int i = 0; i < length; ++i) {
        if (std::isnan(data[i])) {
            LOG_ERROR("Exist NaN in input data, cannot do ARQ quantize.\n");
            return AmctCommon::BAD_PARAMETERS_ERROR;
        }
    }
    if (factor.scale.length != factor.offset.length) {
        LOG_ERROR("scale.length = %u, offset.length = %u\n", factor.scale.length, factor.offset.length);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if (arqParam.channelWise && (group > 1)) {
        LOG_ERROR("arqParam.channelWise = %d, group = %u\n", arqParam.channelWise, factor.scale.length);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }
    if ((!arqParam.channelWise) && (factor.scale.length != group)) {
        LOG_ERROR("arqParam.channelWise = %d, scale.length = %u, group = %u\n",
        arqParam.channelWise, factor.scale.length, group);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (factor.scale.length == 0) {
        LOG_ERROR("scale.length = %u should not be 0.\n", factor.scale.length);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (length % factor.scale.length != 0) {
        LOG_ERROR("input data length = %u, scale.length = %u\n", length, factor.scale.length);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    if (arqParam.numBits < MIN_NUM_BITS) {
        LOG_ERROR("arqParam.numBits = %u, it should be greater than 2.\n", arqParam.numBits);
        return AmctCommon::BAD_PARAMETERS_ERROR;
    }

    return AmctCommon::SUCCESS;
}


template <class T>
Status FindMaxAndMinValueCoCiKhKw(const T* data, const unsigned int length, T& maxValue, T& minValue)
{
    for (unsigned int i = 0; i < length; ++i) {
        if (data[i] < minValue) {
            minValue = data[i];
        }
        if (data[i] > maxValue) {
            maxValue = data[i];
        }
    }

    return AmctCommon::SUCCESS;
}


template <class T>
Status FindMaxAndMinValueKhKwCiCo(const T* data, const unsigned int length, T& maxValue, T& minValue,
    unsigned int channelIndex, unsigned int channelNum)
{
    if (channelNum == 0) {
        return AmctCommon::ZERO_DIVISION_ERROR;
    }
    for (unsigned int i = 0; i < length / channelNum; i++) {
        T temp = data[i * channelNum + channelIndex];
        minValue = temp < minValue ? temp : minValue;
        maxValue = temp > maxValue ? temp : maxValue;
    }
    return AmctCommon::SUCCESS;
}


template <class T>
void ArqClipAndQuantKhKwCiCo(const T* input, T* output, const unsigned int length, const int numBits,
    const FloatData &scaleValue, const IntData &offsetValue)
{
    const int baseNum = 2;
    const int minLimit = -static_cast<int>(pow(baseNum, numBits - 1));
    const int maxLimit = static_cast<int>(pow(baseNum, numBits - 1) - 1);

    unsigned int coutIdx = 0;
    for (unsigned int i = 0; i < length; i++) {
        int dataTmp = static_cast<int>(round(input[i] / scaleValue.data[coutIdx]) + offsetValue.data[coutIdx]);
        dataTmp = dataTmp < minLimit ? minLimit : dataTmp;
        dataTmp = dataTmp > maxLimit ? maxLimit : dataTmp;
        output[i] = (dataTmp - offsetValue.data[coutIdx]) * scaleValue.data[coutIdx];

        coutIdx += 1;
        if (coutIdx == scaleValue.length) {
            coutIdx = 0;
        }
    }
}


template <class T>
void ArqClipAndQuant(T* data, const unsigned int length, const int numBits, const float scaleValue,
    const int offsetValue)
{
    const int baseNum = 2;
    const int minLimit = -static_cast<int>(pow(baseNum, numBits - 1));
    const int maxLimit = static_cast<int>(pow(baseNum, numBits - 1) - 1);

    for (unsigned int i = 0; i < length; ++i) {
        int dataTmp = static_cast<int>(round(data[i] / scaleValue)) + offsetValue;
        dataTmp = dataTmp < minLimit ? minLimit : dataTmp;
        dataTmp = dataTmp > maxLimit ? maxLimit : dataTmp;
        data[i] = (dataTmp - offsetValue) * scaleValue;
    }
}

template <class T>
Status WtsArqCalibrationCpuKernel(const T* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, WeightFormat format)
{
    Status ret;
    const float baseNum = 2.0;
    const float minLimit = static_cast<float>(-pow(baseNum, arqParam.numBits - 1));
    const float maxLimit = static_cast<float>(pow(baseNum, arqParam.numBits - 1) - 1);
    for (unsigned int channelIndex = 0; channelIndex < scale.length; ++channelIndex) {
        T maxValue = -FLT_MAX;
        T minValue = FLT_MAX;
        unsigned int perChannelLength = length / scale.length;
        unsigned int baseOffset = channelIndex * perChannelLength;
        const T* kernelData = data + baseOffset;
        if (format == CO_CI_KH_KW) {
            ret = FindMaxAndMinValueCoCiKhKw(kernelData, perChannelLength, maxValue, minValue);
            if (ret != AmctCommon::SUCCESS) {
                LOG_ERROR("FindMaxAndMinValueCoCiKhKw failed.\n");
                return ret;
            }
        } else {
            ret = FindMaxAndMinValueKhKwCiCo(data, length, maxValue, minValue, channelIndex, scale.length);
            if (ret != AmctCommon::SUCCESS) {
                LOG_ERROR("FindMaxAndMinValueKhKwCiCo failed.\n");
                return ret;
            }
        }

        if (!arqParam.withOffset) {
            // without offset
            float tmpMax = (fabs(maxValue) > fabs(minValue)) ?
                static_cast<float>(maxValue) : static_cast<float>(minValue);
            scale.data[channelIndex] = (tmpMax > 0) ? (tmpMax / maxLimit) : (tmpMax / minLimit);
            Status result = util::ProcessScale(scale.data[channelIndex]);
            if (result != AmctCommon::SUCCESS) {
                LOG_ERROR("ArqQuantCPU scale is illegal.\n");
                return result;
            }
            offset.data[channelIndex] = 0;
        } else {
            // with offset
            maxValue = maxValue < 0.0f ? 0.0f : maxValue;
            minValue = minValue > 0.0f ? 0.0f : minValue;
            // get scale
            scale.data[channelIndex] = static_cast<float>((maxValue - minValue) / (maxLimit - minLimit));
            Status result = util::ProcessScale(scale.data[channelIndex]);
            if (result != AmctCommon::SUCCESS) {
                LOG_ERROR("ArqQuantCPU scale is illegal.\n");
                return result;
            }
            // get offset
            offset.data[channelIndex] = static_cast<int>(round(-minValue / scale.data[channelIndex]) + minLimit);
        }
    }
    return AmctCommon::SUCCESS;
}


template <class T>
void ArqFakeQuantize(T* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset)
{
    for (unsigned int channelIndex = 0; channelIndex < scale.length; ++channelIndex) {
        unsigned int perChannelLength = length / scale.length;
        unsigned int baseOffset = channelIndex * perChannelLength;

        T* kernelData = data + baseOffset;
        ArqClipAndQuant(kernelData, perChannelLength, static_cast<int>(arqParam.numBits),
            scale.data[channelIndex], offset.data[channelIndex]);
    }
}


template <class T>
Status ArqQuantRealKernel(T* data, unsigned int length, const ArqParam &arqParam, const FloatData &scale,
    const IntData &offset, char* int8Data)
{
    Status ret = CheckArqQuantParams(data, length, arqParam, {scale, offset});
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("CheckArqQuantParams failed.\n");
        return ret;
    }

    const int baseNum = 2;
    const int minLimit = -static_cast<int>(pow(baseNum, arqParam.numBits - 1));
    const int maxLimit = static_cast<int>(pow(baseNum, arqParam.numBits - 1) - 1);

    for (unsigned int channelIndex = 0; channelIndex < scale.length; ++channelIndex) {
        unsigned int perChannelLength = length / scale.length;
        unsigned int baseOffset = channelIndex * perChannelLength;

        T* kernelData = data + baseOffset;
        char* int8DataTmp = int8Data + baseOffset;
        float scaleValue = scale.data[channelIndex];
        int offsetValue = offset.data[channelIndex];

        Status result = util::ProcessScale(scaleValue);
        if (result != AmctCommon::SUCCESS) {
            LOG_ERROR("ArqQuantRealCPU scale is illegal.\n");
            return result;
        }

        for (unsigned int index = 0; index < perChannelLength; ++index) {
            int dataTmp = static_cast<int>(round(kernelData[index] / scaleValue) + offsetValue);
            // clip into int8
            dataTmp = dataTmp < minLimit ? minLimit : dataTmp;
            dataTmp = dataTmp > maxLimit ? maxLimit : dataTmp;
            int8DataTmp[index] = static_cast<char>(dataTmp);
        }
    }

    return AmctCommon::SUCCESS;
}


// check function for arq_quznt.cpp and arq_quant.cu
Status CheckArqQuantParams(const float* data, unsigned int length, const ArqParam &arqParam,
    const QuantFactors &factor, uint32_t group)
{
    return CheckArqParams(data, length, arqParam, factor, group);
}


Status CheckArqQuantParams(const double* data, unsigned int length, const ArqParam &arqParam,
    const QuantFactors &factor, uint32_t group)
{
    return CheckArqParams(data, length, arqParam, factor, group);
}


// C++ API
template <class T>
Status ArqQuantReal(T* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, char* int8Data)
{
    return ArqQuantRealKernel(data, length, arqParam, scale, offset, int8Data);
}

template <class T>
Status ArqQuant(T* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, uint32_t group)
{
    Status statusCode = CheckArqQuantParams(data, length, arqParam, {scale, offset}, group);
    if (statusCode != AmctCommon::SUCCESS) {
        LOG_ERROR("CheckArqQuantParams failed.\n");
        return statusCode;
    }
    statusCode = WtsArqCalibrationCpuKernel(data, length, arqParam, scale, offset, CO_CI_KH_KW);
    if (statusCode != AmctCommon::SUCCESS) {
        LOG_ERROR("WtsArqCalibrationCpuKernel of format CO_CI_KH_KW failed.\n");
        return statusCode;
    }
    ArqFakeQuantize(data, length, arqParam, scale, offset);
    return statusCode;
}

template Status ArqQuant(float* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, uint32_t group);
template Status ArqQuant(double* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, uint32_t group);

template Status ArqQuantReal(float* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, char* int8Data);
template Status ArqQuantReal(double* data, unsigned int length, const ArqParam &arqParam,
    const FloatData &scale, const IntData &offset, char* int8Data);

template Status WtsArqCalibrationCpuKernel(const float* data, unsigned int length,
    const ArqParam &arqParam, const FloatData &scale, const IntData &offset, WeightFormat format);
template Status WtsArqCalibrationCpuKernel(const double* data, unsigned int length,
    const ArqParam &arqParam, const FloatData &scale, const IntData &offset, WeightFormat format);

template void ArqClipAndQuantKhKwCiCo(const float* input, float* output, const unsigned int length,
    const int numBits, const FloatData &scaleValue, const IntData &offsetValue);
template void ArqClipAndQuantKhKwCiCo(const double* input, double* output, const unsigned int length,
    const int numBits, const FloatData &scaleValue, const IntData &offsetValue);
}

// Python API
int QuantRealDoublePython(double* data, unsigned int length, FloatData scale,
    IntData offset, unsigned int numBits, char* int8Data)
{
    AmctCommon::ArqParam arqParam {
        numBits,
        (scale.length > 1) ? true : false,
        false
    };
    return AmctCommon::ArqQuantRealKernel(data, length, arqParam, scale, offset, int8Data);
}

int ArqQuantFloatPython(float *data, unsigned int length, AmctCommon::ArqParam arqParam,
    FloatData scale, IntData offset)
{
    Status status = AmctCommon::CheckArqQuantParams(data, length, arqParam, {scale, offset});
    if (status != AmctCommon::SUCCESS) {
        LOG_ERROR("CheckArqQuantParams failed.\n");
        return status;
    }
    status = AmctCommon::WtsArqCalibrationCpuKernel(data, length, arqParam, scale, offset, CO_CI_KH_KW);
    if (status != AmctCommon::SUCCESS) {
        LOG_ERROR("WtsArqCalibrationCpuKernel float of format CO_CI_KH_KW failed.\n");
        return status;
    }
    AmctCommon::ArqFakeQuantize(data, length, arqParam, scale, offset);
    return status;
}

int ArqQuantDoublePython(double* data, unsigned int length, AmctCommon::ArqParam arqParam,
    FloatData scale, IntData offset)
{
    Status ret = AmctCommon::CheckArqQuantParams(data, length, arqParam, {scale, offset});
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("CheckArqQuantParams failed.\n");
        return ret;
    }
    ret = AmctCommon::WtsArqCalibrationCpuKernel(data, length, arqParam, scale, offset, CO_CI_KH_KW);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("WtsArqCalibrationCpuKernel double of format CO_CI_KH_KW failed.\n");
        return ret;
    }
    AmctCommon::ArqFakeQuantize(data, length, arqParam, scale, offset);
    return ret;
}
