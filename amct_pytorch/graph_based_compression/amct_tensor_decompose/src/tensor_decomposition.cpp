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
 * @file tensor_decomposition.cpp
 *
 * @version 1.0
 */

#include "tensor_decomposition.h"
#include <cstdio>
#include <algorithm>
#include "vector.h"
#include "td_log.h"
using namespace std;

namespace TensorDecompose {
int TensorDecomposition::Check(TdError ret, int estimateRank, int originRank)
{
    if (ret == TdError::TD_SUCCESS && TensorDecomposition::CheckScalar(estimateRank)) {
        return min(estimateRank, originRank);
    } else {
        return originRank;
    }
}


bool TensorDecomposition::CheckScalar(int val, bool checkLargerThanZero)
{
    if (isnan(val) || isinf(val)) {
        return false;
    }
    if (checkLargerThanZero && val < 0) {   // check value number 0
        return false;
    }
    return true;
}


bool TensorDecomposition::CheckScalar(double val, bool checkLargerThanZero)
{
    if (isnan(val) || isinf(val)) {
        return false;
    }
    if (checkLargerThanZero && val < 0) {   // check value number 0
        return false;
    }
    return true;
}


bool TensorDecomposition::CheckScalar(unsigned int val)
{
    if (isnan(val) || isinf(val)) {
        return false;
    }
    return true;
}


TdError TensorDecomposition::Tau(Vector &vecRes, const Vector &vecX, double alpha)
{
    // vecRes = 0.5 * (vecX - (1 + alpha) + sqrt((vecX - (1 + alpha)) ** 2 - 4 * alpha))
    Vector vecA, vecB, vecC, vecD, vecE, vecF;           // vecRes
    TD_FUNC_CHECK(vecX.Sub(vecA, 1 + alpha));       // vecX - (1 + alpha)
    TD_FUNC_CHECK(vecX.Sub(vecB, 1 + alpha));       // vecX - (1 + alpha)
    TD_FUNC_CHECK(vecB.Mul(vecC, vecB));            // vecB ** 2
    TD_FUNC_CHECK(vecC.Sub(vecD, 4 * alpha));       // vecB ** 2 - 4 * alpha
    TD_FUNC_CHECK(vecD.Sqrt(vecE));                 // sqrt( vecB ** 2 - 4 * alpha)
    TD_FUNC_CHECK(vecE.Add(vecF, vecA));            // vecA + sqrt( vecB ** 2 - 4 * alpha)
    TD_FUNC_CHECK(vecF.Mul(vecRes, 0.5));           // VecF * 0.5
    return TdError::TD_SUCCESS;
}


/*
 * 单次sigma2的计算，被ArgMin循环调用。
 * 结果存入obj
 */
TdError TensorDecomposition::Sigma2(double &obj, double sigma2, unsigned int sizeL, unsigned int sizeM,
    const Vector &vecS)
{
    if (sizeL < 1 || sizeM < 1 || vecS.GetLength() < 1) {   // check size 1
        return TdError::TD_BAD_PARAMETERS_ERR;
    }
    double alpha = static_cast<double>(sizeL) / static_cast<double>(sizeM);
    double paramX = 2.5129;
    double tauubar = paramX * sqrt(alpha);
    Vector sSliceH;
    double xubar = (1 + tauubar) * (1 + alpha / tauubar);   // calculate xubar 1

    Vector sPower2, vecX;
    TD_FUNC_CHECK(vecS.Mul(sPower2, vecS));
    TD_FUNC_CHECK(sPower2.Mul(vecX, 1.0 / (sizeM * sigma2)));    // calculate power 1.0
    Vector z1, z2;

    TD_FUNC_CHECK(vecX.Select(z1, xubar, true, false));
    TD_FUNC_CHECK(vecX.Select(z2, xubar, false, true));

    Vector tauZ1;
    TD_FUNC_CHECK(Tau(tauZ1, z1, alpha));
    // Function: term1 = sum(z2 - log(z2))
    Vector vecA1, vecB1;
    TD_FUNC_CHECK(z2.Log(vecA1));
    TD_FUNC_CHECK(z2.Sub(vecB1, vecA1));
    double term1;
    TD_FUNC_CHECK(vecB1.Sum(term1));

    // Function: term2 = sum(z1 - tauZ1)
    Vector vecA2, vecB2;
    TD_FUNC_CHECK(z1.Sub(vecB2, tauZ1));
    double term2;
    TD_FUNC_CHECK(vecB2.Sum(term2));

    // Function: term3 = sum(log(divide(tauZ1 + 1, z1)))
    Vector vecA3, vecB3, vecC3;
    TD_FUNC_CHECK(tauZ1.Add(vecA3, 1)); // calculate tau 1
    TD_FUNC_CHECK(vecA3.Div(vecB3, z1));
    TD_FUNC_CHECK(vecB3.Log(vecC3));
    double term3;
    TD_FUNC_CHECK(vecC3.Sum(term3));

    // Function: term4 = alpha * sum(log(tauZ1 / alpha + 1))
    Vector vecA4, vecB4, vecC4;
    TD_FUNC_CHECK(tauZ1.Mul(vecA4, 1.0 / alpha));   // calculate tau 1.0

    TD_FUNC_CHECK(vecA4.Add(vecB4, 1));             // calculate vector 1

    TD_FUNC_CHECK(vecB4.Log(vecC4));

    double tmpSum;
    TD_FUNC_CHECK(vecC4.Sum(tmpSum));
    double term4 = alpha * tmpSum;
    obj = term1 + term2 + term3 + term4;
    return TdError::TD_SUCCESS;
}


void TensorDecomposition::Calculation1(const double &p, const double &q, const double &r, const double &a,
    const double &xf, const double &b, double &rat, double &x, const double &tol2,
    const double &xm, const double &tol1, bool &golden)
{
    // half 0.5 paramter p and q
    if ((abs(p) < abs(0.5 * q * r)) && (p > q * (a - xf)) && (p < q * (b - xf))) {
        rat = p / q;
        x = xf + rat;

        if (((x - a) < tol2) || ((b - x) < tol2)) {
            double si = 1.0;
            if (xm < xf) {
                si = -1.0;
            }
            rat = tol1 * si;
        }
    } else {
        golden = true;
    }
}


void TensorDecomposition::Calculation2(double &rat, const bool &golden, const double &xf, const double &xm,
    double &e, const double &a, const double &b, const double &goldenMean)
{
    if (golden) {
        if (xf >= xm) {
            e = a - xf;
        } else {
            e = b - xf;
        }
        rat = goldenMean * e;
    }
}


void TensorDecomposition::Calculation3(const double &x, double &xg, double &fa, double &fb, double &ngc, double &gulc,
    double &gngc, double &ggulc, double &gx, const double &gu)
{
    if (x < xg) {
        fb = xg;
    } else {
        fa = xg;
    }
    gulc = ngc;
    ggulc = gngc;
    ngc = xg;
    gngc = gx;
    xg = x;
    gx = gu;
}


void TensorDecomposition::Calculation4(const double &x, const double &xg, double &fa, double &fb, const double &gu,
    double &gngc, double &ngc, double &gulc, double &ggulc)
{
    if (gu <= gngc || ngc == xg) {
        gulc = ngc;
        ggulc = gngc;
        gngc = gu;
        ngc = x;
    } else if (gu <= ggulc || gulc == xg || gulc == ngc) {
        ggulc = gu;
        gulc = x;
    }
    if (x < xg) {
        fa = x;
    } else {
        fb = x;
    }
}


void TensorDecomposition::UpdateP(double &p, const double &q)
{
    if (q > 0.0) {  // if q 0.0 higher than zero, reverse p
        p = -p;
    }
}


void TensorDecomposition::UpdateSI(double &si, const double &rat)
{
    if (rat < 0) {  // rat smaller than 0
        si = -1.0;
    }
}


TdError TensorDecomposition::LoopCalculation(double &xf, double &xm, double &tol2, double &b, double &a, double &e,
    double &tol1, double &nfc, double &fx, double &ffulc, double &fulc, double &fnfc, double &rat, double &x,
    const double &goldenMean, unsigned int sizeL, unsigned int sizeM, const Vector &vecS, unsigned int num,
    const double &sqrtEps, const double &xatol, unsigned int maxIter)
{
    while (abs(xf - xm) > (tol2 - 0.5 * (b - a))) { // loop condition 0.5
        bool golden = true;

        if (abs(e) > tol1) {
            golden = false;
            double r = (xf - nfc) * (fx - ffulc);
            double q = (xf - fulc) * (fx - fnfc);
            double p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);  // double factor 2.0
            UpdateP(p, q);
            q = abs(q);
            r = e;
            e = rat;
            TensorDecomposition::Calculation1(p, q, r, a, xf, b, rat, x, tol2, xm, tol1, golden);
        }
        TensorDecomposition::Calculation2(rat, golden, xf, xm, e, a, b, goldenMean);

        double si = 1.0;
        UpdateSI(si, rat);
        x =  xf + si * (abs(rat) > tol1 ? abs(rat) : tol1);
        double fu;
        TD_FUNC_CHECK(Sigma2(fu, x, sizeL, sizeM, vecS));
        num += 1;
        if (fu <= fx) {
            TensorDecomposition::Calculation3(x, xf, a, b, nfc, fulc, fnfc, ffulc, fx, fu);
        } else {
            TensorDecomposition::Calculation4(x, xf, a, b, fu, fnfc, nfc, fulc, ffulc);
        }
        xm = 0.5 * (a + b);                     // half factor 0.5
        tol1 = sqrtEps * abs(xf) + xatol / 3.0; // one in third factor 3.0
        tol2 = 2.0 * tol1;                      // double factor 2.0

        if (num >= maxIter) {
            break;
        }
    }
    return TdError::TD_SUCCESS;
}


/*
 * 在[lowerBound, upperBound]区间内，寻找一个最优的Sigma2让函数EVBSigma2(sigma2, sizeL, sizeM, vecS, xubar)最小，
 * 结果存入resultOut，xatol是可容忍误差，默认值1e-5，maxIter是最大迭代次数，默认值500
 */
TdError TensorDecomposition::ArgMin(double &resultOut, double lowerBound, double upperBound,
    unsigned int sizeL, unsigned int sizeM, const Vector &vecS, double xatol, unsigned int maxIter)
{
    double x1 = lowerBound;
    double x2 = upperBound;
    if (x1 > x2) {
        return TdError::TD_BAD_PARAMETERS_ERR;
    }
    if (!TensorDecomposition::CheckScalar(lowerBound, true) ||
        !TensorDecomposition::CheckScalar(upperBound, true) ||
        !TensorDecomposition::CheckScalar(sizeL) ||
        !TensorDecomposition::CheckScalar(sizeM) ||
        !TensorDecomposition::CheckScalar(xatol) ||
        !TensorDecomposition::CheckScalar(maxIter)) {
        return TdError::TD_BAD_PARAMETERS_ERR;
    }
    double sqrtEps = sqrt(2.2e-16);                 // get sqrt of eps 2.2e-16
    double goldenMean = 0.5 * (3.0 - sqrt(5.0));    // get goldenMean 0.5, 3.0, 5.0
    double a = x1;
    double b = x2;
    double fulc = a + goldenMean * (b - a);
    double nfc = fulc;
    double xf = fulc;
    double rat = 0.0;
    double e = 0.0;
    double x = xf;
    double fx = 0.0;
    TD_FUNC_CHECK(TensorDecomposition::Sigma2(fx, x, sizeL, sizeM, vecS));
    unsigned int num = 1;
    double ffulc = fx;
    double fnfc = fx;
    double xm = 0.5 * (a + b);                      // half factor 0.5
    double tol1 = sqrtEps * abs(xf) + xatol / 3.0;  // one in third facotr 3.0
    double tol2 = 2.0 * tol1;                       // double factor 2.0

    TD_FUNC_CHECK(TensorDecomposition::LoopCalculation(xf, xm, tol2, b, a, e, tol1, nfc, fx, ffulc, fulc, fnfc,
        rat, x, goldenMean, sizeL, sizeM, vecS, num, sqrtEps, xatol, maxIter));
    resultOut = xf;
    return TdError::TD_SUCCESS;
}


/*
 * 基于Empirical-VBMF算法估计阈值，筛选出大于该阈值的奇异值，筛选后奇异值的数量即为估计的秩。
 * 结果存入resultOut，vecS是张量的奇异向量。
 * References
 * ----------
 * [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix
 * factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.
 * [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in
 * Neural Information Processing Systems. 2012.
 */
TdError TensorDecomposition::EVBMF(unsigned int &resultOut, int sizeL, int sizeM, const Vector &vecS)
{
    if (sizeL < 1 || sizeM < 1 || vecS.GetLength() == 0) {  // check size 1 0
        return TdError::TD_BAD_PARAMETERS_ERR;
    }
    double alpha = static_cast<double>(sizeL) / static_cast<double>(sizeM);
    int sizeH = sizeL;
    double paramX = 2.5129;
    double tauubar = paramX * sqrt(alpha);
    double residual = 0.0;

    Vector sSliceH;
    TD_FUNC_CHECK(vecS.Slice(sSliceH, 0, sizeH));           // slice vector 0
    double xubar = (1 + tauubar) * (1 + alpha / tauubar);   // get xubar 1
    int eHub = static_cast<int>(min(static_cast<int>(ceil(static_cast<double>(sizeL) / (1 + alpha)) - 1),
        sizeH)) - 1;    // get eHub 1

    // upperBound = (sum(vecS ** 2) + residual) / (sizeL * sizeM)
    Vector sHpower2;
    TD_FUNC_CHECK(sSliceH.Mul(sHpower2, sSliceH));

    double ss1Sum;
    TD_FUNC_CHECK(sHpower2.Sum(ss1Sum));

    double upperBound = (ss1Sum + residual) / static_cast<double>(sizeL * sizeM);

    // lowerBound = max([vecS[eHub + 1] ** 2 / (sizeM * xubar), mean(vecS[eHub + 1:] ** 2) / sizeM])
    double sHVal;
    TD_FUNC_CHECK(sSliceH.GetValue(sHVal, eHub + 1));  // slice vector 1

    Vector sHSliceEnd;
    TD_FUNC_CHECK(sSliceH.Slice(sHSliceEnd, eHub, sSliceH.GetLength()));

    Vector sHEndPower2;
    TD_FUNC_CHECK(sHSliceEnd.Mul(sHEndPower2, sHSliceEnd));

    double sHMean;
    TD_FUNC_CHECK(sHEndPower2.Mean(sHMean));

    double lowerBound = max(sHVal * sHVal / (static_cast<double>(sizeM) * xubar), sHMean / static_cast<double>(sizeM));

    double sigma2 = (lowerBound + upperBound) / 2;  // mean of lower and upper bound 2
    TensorDecomposition::ArgMin(sigma2, lowerBound, upperBound, sizeL, sizeM, vecS);

    double threshold = sqrt(static_cast<double>(sizeM) * sigma2 * (1 + tauubar) * (1 + alpha / tauubar)); // 1 thr

    Vector sHSelect;
    TD_FUNC_CHECK(sSliceH.Select(sHSelect, threshold, true, false));

    resultOut = sHSelect.GetLength();
    return TdError::TD_SUCCESS;
}


/*
 * 调整EVBMF估计得到的秩，使之能被divisior整除。
 * 结果存入newV。
 */
TdError TensorDecomposition::MakeDivisible(unsigned int &newV, int rank, int divisior, int minVal)
{
    if (TensorDecomposition::CheckScalar(rank, true) && TensorDecomposition::CheckScalar(divisior, true) &&
        TensorDecomposition::CheckScalar(minVal, true) && (divisior != 0)) {
        newV = max(minVal, (static_cast<int>(rank + divisior / 2) / divisior) * divisior);    // get new value 2
        if (newV < 0.9 * rank) {                                                            // divide rule 0.9
            newV += divisior;
        }
        return TdError::TD_SUCCESS;
    } else {
        return TdError::TD_BAD_PARAMETERS_ERR;
    }
}


/*
 * 基于unfold之后张量的长和宽、奇异特征，估计张量的秩。
 * 结果存入rankRes。
 */
TdError TensorDecomposition::EstimateRanks(unsigned int &rankResult, const Vector &vecS,
    int sizeL, int sizeM, int divisor)
{
    if (sizeL != static_cast<int>(vecS.GetLength())) {
        return TdError::TD_BAD_PARAMETERS_ERR;
    }
    int channelStep = 16;
    int minChannel = 16;

    unsigned int rank;
    TD_FUNC_CHECK(EVBMF(rank, sizeL, sizeM, vecS));
    if (rank == 0) {    // rank 0
        (void)printf("[WARNING][%s][%d] Warning: estimate rank is %u, please check if pretrained weight is correct.\n",
            __FUNCTION__, __LINE__, rank);
    }
    if ((divisor == 0) || ((divisor + 1) == 0)) {
        return TdError::TD_BAD_PARAMETERS_ERR;
    }
    rank = min(static_cast<int>(rank), static_cast<int>(sizeL / divisor));
    rank = max(static_cast<int>(rank), static_cast<int>(sizeL / (divisor + 1)));
    TD_FUNC_CHECK(MakeDivisible(rankResult, rank, channelStep, minChannel));
    return TdError::TD_SUCCESS;
}


/*
 * 基于卷积信息（info）和奇异特征（*pS），经过条件判断后，再经由EstimateRanks估计张量的秩。
 * 返回估计得到的秩。
 */
unsigned int TensorDecomposition::Estimation(const ConvInfo &info, const Vector &vecS, unsigned int length)
{
    int sizeL = min(info.kernelSizeH, info.kernelSizeW) * min(info.inChannel, info.outChannel);
    int sizeM = max(info.kernelSizeH, info.kernelSizeW) * max(info.inChannel, info.outChannel);

    unsigned int estRank = length;
    int divisor = 6; // base divisor is 6
    int channelThreshold = 256;
    if (min(info.inChannel, info.outChannel) >= channelThreshold) {
        divisor = 4; // divisor is 4 when channel is more than 256
    }
    TdError ret = TensorDecomposition::EstimateRanks(estRank, vecS, sizeL, sizeM, divisor);
    return TensorDecomposition::Check(ret, estRank, vecS.GetLength());
}

}