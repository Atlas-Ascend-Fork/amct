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
 * @brief vector class implementation
 *
 * @file vector.cpp
 *
 * @version 1.0
 */

#include "vector.h"
#include <new>
#include "securec.h"


namespace TensorDecompose {
bool Vector::CheckValidDataGT(const Vector &vectorIn)
{
    if (vectorIn.data == nullptr) {
        return false;
    }
    double* srcEnd = vectorIn.data + vectorIn.length;
    for (double* src = vectorIn.data; src != srcEnd; src++) {
        if (std::isnan(*src) || std::isinf(*src) || *src <= 0.0) {
            return false;
        }
    }
    return true;
}

bool Vector::CheckValidDataNoZero(const Vector &vectorIn)
{
    if (vectorIn.data == nullptr) {
        return false;
    }
    double* srcEnd = vectorIn.data + vectorIn.length;
    for (double* src = vectorIn.data; src != srcEnd; src++) {
        if (std::isnan(*src) || std::isinf(*src) || *src == 0.0) {
            return false;
        }
    }
    return true;
}

bool Vector::CheckValidDataGTE(const Vector &vectorIn)
{
    if (vectorIn.data == nullptr) {
        return false;
    }
    double* srcEnd = vectorIn.data + vectorIn.length;
    for (double* src = vectorIn.data; src != srcEnd; src++) {
        if (std::isnan(*src) || std::isinf(*src) || *src < 0.0) {
            return false;
        }
    }
    return true;
}

bool Vector::CheckValidDataAny(const Vector &vectorIn)
{
    if (vectorIn.data == nullptr) {
        return false;
    }
    double* srcEnd = vectorIn.data + vectorIn.length;
    for (double* src = vectorIn.data; src != srcEnd; src++) {
        if (std::isnan(*src) || std::isinf(*src)) {
            return false;
        }
    }
    return true;
}

bool Vector::CheckDataGT(const Vector &vectorIn)
{
    if (vectorIn.data == nullptr) {
        return false;
    }
    double* srcEnd = vectorIn.data + vectorIn.length;
    for (double* src = vectorIn.data; src != srcEnd; src++) {
        if (*src <= 0.0) {
            return false;
        }
    }
    return true;
}

bool Vector::CheckDataNoZero(const Vector &vectorIn)
{
    if (vectorIn.data == nullptr) {
        return false;
    }
    double* srcEnd = vectorIn.data + vectorIn.length;
    for (double* src = vectorIn.data; src != srcEnd; src++) {
        if (*src == double(0.0)) {
            return false;
        }
    }
    return true;
}

bool Vector::CheckDataGTE(const Vector &vectorIn)
{
    if (vectorIn.data == nullptr) {
        return false;
    }
    double* srcEnd = vectorIn.data + vectorIn.length;
    for (double* src = vectorIn.data; src != srcEnd; src++) {
        if (*src < 0.0) {
            return false;
        }
    }
    return true;
}

bool Vector::CheckValidData(const Vector &vectorIn, bool checkNoZero, bool checkNonNegative)
{
    if (checkNoZero && checkNonNegative) {
        return Vector::CheckValidDataGT(vectorIn);
    } else if (checkNoZero && !checkNonNegative) {
        return Vector::CheckValidDataNoZero(vectorIn);
    } else if (!checkNoZero && checkNonNegative) {
        return Vector::CheckValidDataGTE(vectorIn);
    }
    return Vector::CheckValidDataAny(vectorIn);
}

bool Vector::CheckData(const Vector &vectorIn, bool checkNonEmpty, bool checkNoZero,
                       bool checkNonNegative, bool checkValid)
{
    if (vectorIn.length == 0) {
        return !checkNonEmpty; // 为空时，若需判断则返回false，若不需判断则作为正常返回true
    }
    if (vectorIn.data == nullptr) { // length!=0但data为空，作为异常
        return false;
    }
    if (checkValid) {
        return Vector::CheckValidData(vectorIn, checkNoZero, checkNonNegative);
    } else {
        if (checkNoZero && checkNonNegative) {
            return Vector::CheckDataGT(vectorIn);
        } else if (checkNoZero && !checkNonNegative) {
            return Vector::CheckDataNoZero(vectorIn);
        } else if (!checkNoZero && checkNonNegative) {
            return Vector::CheckDataGTE(vectorIn);
        }
    }
    return true;
}

Vector::Vector()
    : length(0),
      data(nullptr)
{
}

void Vector::Destroy()
{
    if (this->data != nullptr) {
        delete[] this->data;
        this->data = nullptr;
    }
    this->length = 0;
}

Vector::~Vector()
{
    this->Destroy();
}

TdError Vector::Create(unsigned int lengthIn)
{
    this->Destroy(); // 清理原有数据
    if (lengthIn == 0) {
        return TdError::TD_SUCCESS;
    }
    this->data = new(std::nothrow) double[lengthIn];

    if (this->data == nullptr) {
        this->length = 0;
        return TdError::TD_OUT_OF_MEMORY_ERR;
    }
    errno_t ret = memset_s(this->data, lengthIn * sizeof(double), 0, lengthIn * sizeof(double));
    if (ret != EOK) {
        this->Destroy();
        return TdError::TD_MEM_OPERATION_ERR;
    }
    this->length = lengthIn;
    return TdError::TD_SUCCESS;
}

TdError Vector::Create(const Vector &vectorIn)
{
    if (&vectorIn == this) { // 用自身创建自身，保持内容不变
        return TdError::TD_SUCCESS;
    }
    if (vectorIn.length == 0) {
        this->Destroy();
        return TdError::TD_SUCCESS;
    }
    return this->Create(vectorIn.data, vectorIn.length);
}

TdError Vector::Create(const double* const dataIn, unsigned int lengthIn)
{
    if (dataIn == nullptr || lengthIn == 0) {
        this->Destroy();
        return TdError::TD_BAD_PARAMETERS_ERR;
    }
    if (this->data != nullptr &&
        ((this->data < dataIn + lengthIn && this->data >= dataIn) ||
        (this->data + this->length <= dataIn + lengthIn && this->data + this->length > dataIn))) {
            return TdError::TD_MEM_OPERATION_ERR; // 内存重叠，直接返回，否则后续执行Destroy将销毁部分dataIn指向的内存
    }
    if (lengthIn != this->length) {
        this->Destroy();
        this->data = new(std::nothrow) double[lengthIn];
    }
    if (this->data == nullptr) {
        this->Destroy();
        return TdError::TD_NULL_DATA_ERR;
    }
    errno_t ret = memcpy_s(this->data, lengthIn * sizeof(double), dataIn, lengthIn * sizeof(double));
    if (ret != EOK) {
        this->Destroy();
        return TdError::TD_MEM_OPERATION_ERR;
    }
    this->length = lengthIn;
    return TdError::TD_SUCCESS;
}

unsigned int Vector::GetLength() const
{
    return this->length;
}

TdError Vector::GetValue(double &valueOut, unsigned int idx) const
{
    if (idx >= this->length) {
        return TdError::TD_IDX_OUT_OF_BOUNDS_ERR;
    }
    TD_NULLPTR_CHECK(this->data);
    valueOut = this->data[idx];
    return TdError::TD_SUCCESS;
}

TdError Vector::SetValue(double value, unsigned int idx) const
{
    if (idx >= this->length) {
        return TdError::TD_IDX_OUT_OF_BOUNDS_ERR;
    }
    TD_NULLPTR_CHECK(this->data);
    this->data[idx] = value;
    return TdError::TD_SUCCESS;
}

TdError Vector::CheckVectorCommon(Vector &vectorOut) const
{
    if (&vectorOut != this) {
        TD_FUNC_CHECK(vectorOut.Create(this->length));
    }
    TD_NULLPTR_CHECK_DOUBLE(this->data, vectorOut.data);
    return TdError::TD_SUCCESS;
}

TdError Vector::CheckVector(Vector &vectorOut, bool checkNanInf, bool checkNoZero) const
{
    if (this->length == 0) {
        vectorOut.Destroy();
        return TdError::TD_SUCCESS;
    }
    TD_NULLPTR_CHECK(this->data);
    if (!Vector::CheckData(*this, false, checkNoZero && checkNanInf, checkNanInf, checkNanInf)) {
        return TdError::TD_GENERIC_MATH_ERR;
    }
    return this->CheckVectorCommon(vectorOut);
}

TdError Vector::CheckVectorAndValue(Vector &vectorOut, double value, bool checkNanInf) const
{
    if (checkNanInf && !Vector::CheckValid(value)) {
        return TdError::TD_GENERIC_MATH_ERR;
    }
    if (this->length == 0) {
        vectorOut.Destroy();
        return TdError::TD_SUCCESS;
    }
    TD_NULLPTR_CHECK(this->data);
    if (!Vector::CheckData(*this, false, false, false, checkNanInf)) {
        return TdError::TD_GENERIC_MATH_ERR;
    }
    return this->CheckVectorCommon(vectorOut);
}

TdError Vector::CheckVectorAndVector(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf,
    bool checkVectorInNoZero) const
{
    if (this->length == 0 && vectorIn.length == 0) {
        vectorOut.Destroy();
        return TdError::TD_SUCCESS;
    }
    TD_NULLPTR_CHECK_DOUBLE(this->data, vectorIn.data);
    if (vectorIn.length != this->length ||
        !Vector::CheckData(*this, false, false, false, checkNanInf) ||
        !Vector::CheckData(vectorIn, false, checkVectorInNoZero && checkNanInf, false, checkNanInf)) {
        return TdError::TD_GENERIC_MATH_ERR;
    }
    if (&vectorOut != this && &vectorOut != &vectorIn) {
        TD_FUNC_CHECK(vectorOut.Create(this->length));
    }
    TD_NULLPTR_CHECK(vectorOut.data);
    return TdError::TD_SUCCESS;
}

TdError Vector::Sum(double &resultOut, bool checkNanInf) const
{
    if (this->length == 0) {
        resultOut = 0.0;
        return TdError::TD_SUCCESS;
    }
    if (!Vector::CheckData(*this, false, false, false, checkNanInf)) {
        return TdError::TD_GENERIC_MATH_ERR;
    }
    TD_NULLPTR_CHECK(this->data);
    double sum = 0;
    double* end = this->data + this->length;
    for (double* p = this->data; p != end; p++) {
        sum += *p;
    }
    resultOut = sum;
    return TdError::TD_SUCCESS;
}

TdError Vector::Mean(double &resultOut, bool checkNanInf) const
{
    // 若不开checkNanInf；则允许除0(空向量，返回nan)
    if (!Vector::CheckData(*this, checkNanInf, false, false, checkNanInf)) {
        return TdError::TD_GENERIC_MATH_ERR;
    }
    if (this->length == 0) {
        resultOut = nan("");
        return TdError::TD_SUCCESS;
    }
    TD_NULLPTR_CHECK(this->data);
    double sum = 0;
    double* end = this->data + this->length;
    for (double* p = this->data; p != end; p++) {
        sum += *p;
    }
    resultOut = sum / this->length;
    return TdError::TD_SUCCESS;
}

TdError Vector::Log(Vector &vectorOut, bool checkNanInf) const
{
    TD_FUNC_CHECK(this->CheckVector(vectorOut, checkNanInf, true));
    TD_CHECK_NORMAL_ZERO_LENGTH(this->length);
    TD_NULLPTR_CHECK_DOUBLE(this->data, vectorOut.data);
    double* srcEnd = this->data + this->length;
    double* dst = vectorOut.data;
    for (double* src = this->data; src != srcEnd; src++) {
        *dst = log(*src);
        dst++;
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::Sqrt(Vector &vectorOut, bool checkNanInf) const
{
    TD_FUNC_CHECK(this->CheckVector(vectorOut, checkNanInf, false));
    TD_CHECK_NORMAL_ZERO_LENGTH(this->length);
    TD_NULLPTR_CHECK_DOUBLE(this->data, vectorOut.data);
    double* srcEnd = this->data + this->length;
    double* dst = vectorOut.data;
    for (double* src = this->data; src != srcEnd; src++) {
        *dst = sqrt(*src);
        dst++;
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::Div(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf) const
{
    TD_FUNC_CHECK(this->CheckVectorAndVector(vectorOut, vectorIn, checkNanInf, true));
    TD_CHECK_NORMAL_ZERO_LENGTH(this->length);
    TD_NULLPTR_CHECK_TRIPLE(this->data, vectorIn.data, vectorOut.data);
    double* src1End = this->data + this->length;
    double* src2 = vectorIn.data;
    double* dst = vectorOut.data;
    for (double* src1 = this->data; src1 != src1End; src1++) {
        *dst = *src1 / (*src2);
        src2++;
        dst++;
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::Mul(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf) const
{
    TD_FUNC_CHECK(this->CheckVectorAndVector(vectorOut, vectorIn, checkNanInf, false));
    TD_CHECK_NORMAL_ZERO_LENGTH(this->length);
    TD_NULLPTR_CHECK_TRIPLE(this->data, vectorIn.data, vectorOut.data);
    double* src1End = this->data + this->length;
    double* src2 = vectorIn.data;
    double* dst = vectorOut.data;
    for (double* src1 = this->data; src1 != src1End; src1++) {
        *dst = *src1 * (*src2);
        src2++;
        dst++;
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::Mul(Vector &vectorOut, double value, bool checkNanInf) const
{
    TD_FUNC_CHECK(this->CheckVectorAndValue(vectorOut, value, checkNanInf));
    TD_CHECK_NORMAL_ZERO_LENGTH(this->length);
    TD_NULLPTR_CHECK_DOUBLE(this->data, vectorOut.data);
    double mulValue = value;
    double* srcEnd = this->data + this->length;
    double* dst = vectorOut.data;
    for (double* src = this->data; src != srcEnd; src++) {
        *dst = *src * mulValue;
        dst++;
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::Add(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf) const
{
    TD_FUNC_CHECK(this->CheckVectorAndVector(vectorOut, vectorIn, checkNanInf, false));
    TD_CHECK_NORMAL_ZERO_LENGTH(this->length);
    TD_NULLPTR_CHECK_TRIPLE(this->data, vectorIn.data, vectorOut.data);
    double* src1End = this->data + this->length;
    double* src2 = vectorIn.data;
    double* dst = vectorOut.data;
    for (double* src1 = this->data; src1 != src1End; src1++) {
        *dst = *src1 + (*src2);
        src2++;
        dst++;
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::Add(Vector &vectorOut, double value, bool checkNanInf) const
{
    TD_FUNC_CHECK(this->CheckVectorAndValue(vectorOut, value, checkNanInf));
    TD_CHECK_NORMAL_ZERO_LENGTH(this->length);
    TD_NULLPTR_CHECK_DOUBLE(this->data, vectorOut.data);
    double* srcEnd = this->data + this->length;
    double* dst = vectorOut.data;
    for (double* src = this->data; src != srcEnd; src++) {
        *dst = *src + value;
        dst++;
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::Sub(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf) const
{
    TD_FUNC_CHECK(this->CheckVectorAndVector(vectorOut, vectorIn, checkNanInf, false));
    TD_CHECK_NORMAL_ZERO_LENGTH(this->length);
    TD_NULLPTR_CHECK_TRIPLE(this->data, vectorIn.data, vectorOut.data);
    double* src1End = this->data + this->length;
    double* src2 = vectorIn.data;
    double* dst = vectorOut.data;
    for (double* src1 = this->data; src1 != src1End; src1++) {
        *dst = *src1 - (*src2);
        src2++;
        dst++;
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::Sub(Vector &vectorOut, double value, bool checkNanInf) const
{
    TD_FUNC_CHECK(this->CheckVectorAndValue(vectorOut, value, checkNanInf));
    TD_CHECK_NORMAL_ZERO_LENGTH(this->length);
    TD_NULLPTR_CHECK_DOUBLE(this->data, vectorOut.data);
    double* srcEnd = this->data + this->length;
    double* dst = vectorOut.data;
    for (double* src = this->data; src != srcEnd; src++) {
        *dst = *src - value;
        dst++;
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::SelectToCacheGTE(const Vector &cacheVec, unsigned int &cacheLengthOut, double value) const
{
    TD_NULLPTR_CHECK_DOUBLE(this->data, cacheVec.data);
    double* dst = cacheVec.data;
    double* srcEnd = this->data + this->length;
    unsigned int cacheLength = 0;
    for (double* src = data; src != srcEnd; src++) {
        if (*src >= value) {
            *dst = *src;
            dst++;
            cacheLength += 1;
        }
    }
    cacheLengthOut = cacheLength;
    return TdError::TD_SUCCESS;
}

TdError Vector::SelectToCacheGT(const Vector &cacheVec, unsigned int &cacheLengthOut, double value) const
{
    TD_NULLPTR_CHECK_DOUBLE(this->data, cacheVec.data);
    double* dst = cacheVec.data;
    double* srcEnd = this->data + this->length;
    unsigned int cacheLength = 0;
    for (double* src = data; src != srcEnd; src++) {
        if (*src > value) {
            *dst = *src;
            dst++;
            cacheLength += 1;
        }
    }
    cacheLengthOut = cacheLength;
    return TdError::TD_SUCCESS;
}

TdError Vector::SelectToCacheLTE(const Vector &cacheVec, unsigned int &cacheLengthOut, double value) const
{
    TD_NULLPTR_CHECK_DOUBLE(this->data, cacheVec.data);
    double* dst = cacheVec.data;
    double* srcEnd = this->data + this->length;
    unsigned int cacheLength = 0;
    for (double* src = data; src != srcEnd; src++) {
        if (*src <= value) {
            *dst = *src;
            dst++;
            cacheLength += 1;
        }
    }
    cacheLengthOut = cacheLength;
    return TdError::TD_SUCCESS;
}

TdError Vector::SelectToCacheLT(const Vector &cacheVec, unsigned int &cacheLengthOut, double value) const
{
    TD_NULLPTR_CHECK_DOUBLE(this->data, cacheVec.data);
    double* dst = cacheVec.data;
    double* srcEnd = this->data + this->length;
    unsigned int cacheLength = 0;
    for (double* src = this->data; src != srcEnd; src++) {
        if (*src < value) {
            *dst = *src;
            dst++;
            cacheLength += 1;
        }
    }
    cacheLengthOut = cacheLength;
    return TdError::TD_SUCCESS;
}

TdError Vector::SelectToCache(Vector &cacheVec, unsigned int &cacheLengthOut, double value, bool selectLarger,
    bool allowEqual) const
{
    if (selectLarger && allowEqual) {
        TD_FUNC_CHECK(this->SelectToCacheGTE(cacheVec, cacheLengthOut, value));
    } else if (selectLarger && !allowEqual) {
        TD_FUNC_CHECK(this->SelectToCacheGT(cacheVec, cacheLengthOut, value));
    } else if (!selectLarger && allowEqual) {
        TD_FUNC_CHECK(this->SelectToCacheLTE(cacheVec, cacheLengthOut, value));
    } else {
        TD_FUNC_CHECK(this->SelectToCacheLT(cacheVec, cacheLengthOut, value));
    }
    return TdError::TD_SUCCESS;
}

TdError Vector::Select(Vector &vectorOut, double value, bool selectLarger, bool allowEqual) const
{
    if (this->length == 0) {
        vectorOut.Destroy();
        return TdError::TD_SUCCESS;
    }
    Vector cacheVec;
    TD_FUNC_CHECK(cacheVec.Create(this->length));
    unsigned int cacheLength = 0;
    TD_FUNC_CHECK(this->SelectToCache(cacheVec, cacheLength, value, selectLarger, allowEqual));
    if (cacheLength == 0) {
        vectorOut.Destroy();
        return TdError::TD_SUCCESS;
    }
    TD_NULLPTR_CHECK(cacheVec.data);
    TD_FUNC_CHECK(vectorOut.Create(cacheVec.data, cacheLength));
    return TdError::TD_SUCCESS;
}

TdError Vector::Slice(Vector &vectorOut, unsigned int startIdx, unsigned int endIdx) const
{
    if (this->length == 0 || startIdx >= endIdx || startIdx >= this->length) {
        vectorOut.Destroy();
        return TdError::TD_SUCCESS;
    }
    TD_NULLPTR_CHECK(this->data);
    if (endIdx > this->length) {
        endIdx = this->length;
    }
    unsigned int sliceLength = endIdx - startIdx;
    double* src = this->data + startIdx;
    if (&vectorOut == this) { // 为保存结果到自身，先创建cacheVec临时存储
        Vector cacheVec;
        TD_FUNC_CHECK(cacheVec.Create(src, sliceLength));
        TD_FUNC_CHECK(vectorOut.Create(cacheVec));
    } else {
        TD_FUNC_CHECK(vectorOut.Create(src, sliceLength));
    }
    return TdError::TD_SUCCESS;
}
}