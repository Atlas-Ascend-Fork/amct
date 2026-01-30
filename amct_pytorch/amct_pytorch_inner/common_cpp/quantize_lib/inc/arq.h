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
 * @brief arq header file
 *
 * @file arq.h in common_cpp
 *
 * @version 1.0
 */

#ifndef ARQ_H
#define ARQ_H

#include "util.h"

namespace AmctCommon {
/**
 * @ingroup quantize lib
 * @brief: Params for Arq Quantiation.
 */
struct ArqParam {
    unsigned int numBits;
    bool channelWise;
    bool withOffset;
};


template <class T>
Status WtsArqCalibrationCudaKernel(const T* data, unsigned int length, const ArqParam &arqParam,
    const util::FloatData &scale, const util::IntData &offset, util::WeightFormat format);


template <class T>
void ArqClipAndQuantKhKwCiCo(const T* input, T* output, const unsigned int length,
    const int numBits, const util::FloatData &scaleValue, const util::IntData &offsetValue);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization kernel Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
template <class T>
Status WtsArqCalibrationCpuKernel(const T* data, unsigned int length, const ArqParam &arqParam,
    const util::FloatData &scale, const util::IntData &offset, util::WeightFormat format);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
template <class T>
Status ArqQuant(T* data, unsigned int length, const ArqParam &arqParam,
    const util::FloatData &scale, const util::IntData &offset, uint32_t group = 1);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @param [out] int8Data: output int8 data.
  * @return succ/fail
  */
template <class T>
Status ArqQuantReal(T* data, unsigned int length, const ArqParam &arqParam,
    const util::FloatData &scale, const util::IntData &offset, char* int8Data);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
template <class T>
Status ArqQuantGPU(T* data, unsigned int length, ArqParam arqParam,
    util::FloatData scale, util::IntData offset);

template <class Dtype>
Status ArqQuantRetrainGPU(Dtype* devData, unsigned int length, ArqParam arqParam,
    util::FloatData scale, util::IntData offset);

template <class Dtype>
Status ArqQuantRetrainGPUInternel(Dtype* devData, unsigned int length, ArqParam arqParam,
    util::FloatData scale, util::IntData offset, util::WeightFormat format);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @param [out] int8Data: output int8 data.
  * @return succ/fail
  */
template <class T>
Status ArqQuantRealGPU(T* data, unsigned int length, ArqParam arqParam,
    util::FloatData scale, util::IntData offset, char* int8Data);

/**
  * @ingroup quantize lib
  * @brief: Check Arq Quantization Inputs Params.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @return succ/fail
  */
Status CheckArqQuantParams(const float* data, unsigned int length, const ArqParam &arqParam,
    const util::QuantFactors &factor, uint32_t group = 1);

Status CheckArqQuantParams(const double* data, unsigned int length, const ArqParam &arqParam,
    const util::QuantFactors &factor, uint32_t group = 1);
}

#ifdef __cplusplus
extern "C"
{
#endif
/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int ArqQuantFloatPython(float* data, unsigned int length, AmctCommon::ArqParam arqParam,
    util::FloatData scale, util::IntData offset);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int ArqQuantDoublePython(double* data, unsigned int length, AmctCommon::ArqParam arqParam,
    util::FloatData scale, util::IntData offset);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @param [in] numBits: bit number to quantize.
  * @param [out] int8Data: output int8 data.
  * @return succ/fail
  */
int QuantRealDoublePython(double* data, unsigned int length, util::FloatData scale, util::IntData offset,
    unsigned int numBits, char* int8Data);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int ArqQuantDoublePythonGPU(double* data, unsigned int length, AmctCommon::ArqParam arqParam,
    util::FloatData scale, util::IntData offset);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @param [in] numBits: bit number to quantize.
  * @param [out] int8Data: output int8 data.
  * @return succ/fail
  */
int QuantRealDoublePythonGPU(double* data, unsigned int length, util::FloatData scale,
    util::IntData offset, unsigned int numBits, char* int8Data);


#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* ARQ_H */
