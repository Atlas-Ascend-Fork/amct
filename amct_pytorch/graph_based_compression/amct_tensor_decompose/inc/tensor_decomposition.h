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
 * @brief Estimate rank for tensor decomposition.
 *
 * @file tensor_decomposition.h
 *
 * @version 1.0
 */


#ifndef TENSOR_DECOMPOSITION_H
#define TENSOR_DECOMPOSITION_H

#include "vector.h"

namespace TensorDecompose {
enum class DecomposeMode {
    DM_UNCHANGE                         = 0,
    DM_FIRST_CHANNEL_FIRST_KERNEL       = 1,
    DM_FIRST_CHANNEL_SECOND_KERNEL      = 2,
    DM_SECOND_CHANNEL_FIRST_KERNEL      = 3,
    DM_SECOND_CHANNEL_SECOND_KERNEL     = 4,
};

struct ConvInfo {
    int inChannel;
    int outChannel;
    int kernelSizeH;
    int kernelSizeW;
    int strideH;
    int strideW;
    int group;
    int dilationH;
    int dilationW;
};

class TensorDecomposition {
public:
    static TdError Sigma2(double &obj, double sigma2, unsigned int sizeL, unsigned int sizeM, const Vector &vecS);
    static TdError ArgMin(double &resultOut, double lowerBound, double upperBound,
                           unsigned int sizeL, unsigned int sizeM,
                           const Vector &vecS, double xatol = 1e-5, unsigned int maxIter = 500);
    static TdError Tau(Vector &vecRes, const Vector &vecX, double alpha);
    static TdError EVBMF(unsigned int &resultOut, int sizeL, int sizeM, const Vector &vecS);
    static TdError MakeDivisible(unsigned int &newV, int rank, int divisior, int minVal);
    static TdError EstimateRanks(unsigned int &rankResult, const Vector &vecS, int sizeL, int sizeM, int divisor);
    static int Check(TdError ret, int estimateRank, int originRank);
    static unsigned int Estimation(const ConvInfo &info, const Vector &vecS, unsigned int length);
    static bool CheckScalar(int val, bool checkLargerThanZero = false);
    static bool CheckScalar(unsigned int val);
    static bool CheckScalar(double val, bool checkLargerThanZero = false);
    static void Calculation1(const double &p, const double &q, const double &r, const double &a,
                             const double &xf, const double &b, double &rat, double &x, const double &tol2,
                             const double &xm, const double &tol1, bool &golden);
    static void Calculation2(double &rat, const bool &golden, const double &xf, const double &xm,
                             double &e, const double &a, const double &b, const double &goldenMean);
    static void Calculation3(const double &x, double &xg, double &fa, double &fb, double &ngc, double &gulc,
                             double &gngc, double &ggulc, double &gx, const double &gu);
    static void Calculation4(const double &x, const double &xg, double &fa, double &fb, const double &gu, double &gngc,
                             double &ngc, double &gulc, double &ggulc);
    static void UpdateSI(double &si, const double &rat);
    static void UpdateP(double &p, const double &q);
    static TdError LoopCalculation(double &xf, double &xm, double &tol2, double &b, double &a, double &e,
                                    double &tol1, double &nfc, double &fx, double &ffulc, double &fulc, double &fnfc,
                                    double &rat, double &x, const double &goldenMean, unsigned int sizeL,
                                    unsigned int sizeM, const Vector &vecS, unsigned int num, const double &sqrtEps,
                                    const double &xatol, unsigned int maxIter);
};
}

#endif
