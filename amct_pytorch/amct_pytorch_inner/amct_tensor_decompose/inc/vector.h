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
 * @brief vector class header
 *
 * @file vector.h
 *
 * @version 1.0
 */

#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>
#include "td_log.h"

namespace TensorDecompose {
class Vector {
public:
    Vector();

    ~Vector();

    // 将向量的长度设为lengthIn，值均为0
    // 成功返回TD_SUCCESS，失败返回错误码，失败时清空本向量
    TdError Create(unsigned int lengthIn = 0);

    // 完整拷贝一个向量对象到本向量
    // 成功返回TD_SUCCESS，失败返回错误码，失败时清空本向量
    TdError Create(const Vector &vectorIn);

    // 使用以dataIn为起点的、长度为lengthIn的连续内存数据构建向量(深拷贝)
    // 成功返回TD_SUCCESS，失败返回错误码，失败时清空本向量
    TdError Create(const double* const dataIn, unsigned int lengthIn);

    // 获取向量长度
    unsigned int GetLength() const;

    // 获取向量元素[idx]的值，输出到valueOut中
    // 要求元素索引存在，成功返回TD_SUCCESS，失败返回错误码
    TdError GetValue(double &valueOut, unsigned int idx) const;

    // 将向量元素[idx]的值设为value
    // 要求元素索引存在，成功返回TD_SUCCESS，失败返回错误码
    TdError SetValue(double value, unsigned int idx) const;

    // 求向量所有元素和，将结果存入resultOut中，向量为空则结果为0.0
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求向量无非法值，成功返回TD_SUCCESS，失败返回错误码
    TdError Sum(double &resultOut, bool checkNanInf = false) const;

    // 求向量所有元素的均值，将结果存入resultOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求向量不为空且无非法值，成功返回TD_SUCCESS，失败返回错误码
    TdError Mean(double &resultOut, bool checkNanInf = false) const;

    // 对向量所有元素求自然对数log，将结果存入vectorOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求向量不为空、无非法值，且各元素均需>0，成功返回TD_SUCCESS，失败返回错误码
    TdError Log(Vector &vectorOut, bool checkNanInf = false) const;

    // 对向量所有元素求平方根，将结果存入vectorOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求向量不为空、无非法值，且各元素均需>=0，成功返回TD_SUCCESS，失败返回错误码
    TdError Sqrt(Vector &vectorOut, bool checkNanInf = false) const;

    // 两个向量element-wise的除法，当前向量除以向量vectorIn，输出到vectorOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求两向量不为空、无非法值、长度需一致，且除数不为0，成功返回TD_SUCCESS，失败返回错误码
    TdError Div(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf = false) const;

    // 两个向量element-wise的乘法，当前向量乘以向量vectorIn，输出到vectorOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求两向量不为空、无非法值，且长度需一致，成功返回TD_SUCCESS，失败返回错误码
    TdError Mul(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf = false) const;

    // 对向量中所有元素乘以value，将结果存入vectorOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求向量不为空且无非法值，成功返回TD_SUCCESS，失败返回错误码
    TdError Mul(Vector &vectorOut, double value, bool checkNanInf = false) const;

    // 两个向量element-wise的加法，当前向量乘以向量vectorIn，输出到vectorOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求两向量不为空、无非法值，且长度需一致，成功返回TD_SUCCESS，失败返回错误码
    TdError Add(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf = false) const;

    // 对向量中所有元素加上value，将结果存入vectorOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求向量不为空、无非法值，且value不是非法值，成功返回TD_SUCCESS，失败返回错误码
    TdError Add(Vector &vectorOut, double value, bool checkNanInf = false) const;

    // 两个向量element-wise的减法，当前向量乘以向量vectorIn，输出到vectorOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求两向量不为空、无非法值，且长度需一致，成功返回TD_SUCCESS，失败返回错误码
    TdError Sub(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf = false) const;

    // 对向量中所有元素减去value，将结果存入vectorOut中
    // checkNanInf：是否检查数据中不存在nan或inf，开启则不允许输入/输出nan或inf，否则允许
    // 要求向量不为空、无非法值，且value不是非法值，成功返回TD_SUCCESS，失败返回错误码
    TdError Sub(Vector &vectorOut, double value, bool checkNanInf = false) const;

    // 获取向量中所有与value比较后符合条件的值，输出向量存储于vectorOut中(可能为空)
    // selectLarger：true时选择大于(等于)value的值，否则选择小于(等于)value的值
    // allowEqual：true时允许选择等于value的值，否则不允许
    // 要求向量不为空，成功返回TD_SUCCESS，失败返回错误码
    TdError Select(Vector &vectorOut, double value, bool selectLarger = true, bool allowEqual = false) const;

    // 向量切片，取[startIdx,endIdx)范围的值，输出到vectorOut中
    // 要求向量不为空，且vectorOut不能是自身，成功返回TD_SUCCESS，失败返回错误码
    TdError Slice(Vector &vectorOut, unsigned int startIdx, unsigned int endIdx) const;

protected:
    unsigned int length;      // 向量长度
    double* data;             // 向量数据

    // 回收内存
    void Destroy();

    // 用于CheckData，针对各类比较的情况
    static bool CheckValidDataGT(const Vector &vectorIn);
    static bool CheckValidDataNoZero(const Vector &vectorIn);
    static bool CheckValidDataGTE(const Vector &vectorIn);
    static bool CheckValidDataAny(const Vector &vectorIn);
    static bool CheckDataGT(const Vector &vectorIn);
    static bool CheckDataNoZero(const Vector &vectorIn);
    static bool CheckDataGTE(const Vector &vectorIn);

    // 检查向量数据是否合法，若其中不存在nan或inf且符合下述标志位规则则返回true，否则返回false
    // checkNoZero：是否检查数据中不存在0
    // checkNonNegative：是否检查数据都为非负
    static bool CheckValidData(const Vector &vectorIn, bool checkNoZero, bool checkNonNegative);

    // 检查向量数据是否合法，符合下述标志位规则返回true，否则返回false
    // checkNonEmpty：是否检查数据不为空
    // checkNoZero：是否检查数据中不存在0
    // checkNonNegative：是否检查数据都为非负
    // checkValid：是否检查数据中不存在nan或inf
    static bool CheckData(const Vector &vectorIn, bool checkNonEmpty = false, bool checkNoZero = false,
                          bool checkNonNegative = false, bool checkValid = false);

    // 检查数值是否不存在nan或inf
    inline static bool CheckValid(const double &value)
    {
        return (!std::isnan(value) && !std::isinf(value));
    }

    // 用于SelectToCache，针对各类比较的情况
    TdError SelectToCacheGTE(const Vector &cacheVec, unsigned int &cacheLengthOut, double value) const;
    TdError SelectToCacheGT(const Vector &cacheVec, unsigned int &cacheLengthOut, double value) const;
    TdError SelectToCacheLTE(const Vector &cacheVec, unsigned int &cacheLengthOut, double value) const;
    TdError SelectToCacheLT(const Vector &cacheVec, unsigned int &cacheLengthOut, double value) const;

    // 获取向量data中所有与value比较后符合条件的值，输出到cacheOut中(需确保已分配足够内存)，cacheLengthOut为输出数据的长度
    // selectLarger：true时选择大于(等于)value的值，否则选择小于(等于)value的值
    // allowEqual：true时允许选择等于value的值，否则不允许
    // 成功返回TD_SUCCESS，失败返回错误码
    TdError SelectToCache(Vector &cacheVec, unsigned int &cacheLengthOut, double value, bool selectLarger,
                           bool allowEqual) const;

    // CheckVector与CheckVectorAndValue的公用部分
    TdError CheckVectorCommon(Vector &vectorOut) const;

    // 向量自身在运算前的检查
    TdError CheckVector(Vector &vectorOut, bool checkNanInf, bool checkNoZero) const;

    // 向量与标量在运算前的检查
    TdError CheckVectorAndValue(Vector &vectorOut, double value, bool checkNanInf) const;

    // 向量与向量在运算前的检查
    TdError CheckVectorAndVector(Vector &vectorOut, const Vector &vectorIn, bool checkNanInf,
                                  bool checkVectorInNoZero) const;

private:
    // 禁用拷贝构造和拷贝赋值，因安全规范不允许在构造函数中使用可能失败的操作(如new)
    Vector(const Vector &);
    Vector &operator=(const Vector &);
};
}

#endif /* VECTOR_H */
