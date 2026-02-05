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
 * @brief HFMG algorithm C++ implementation for CPU.
 *
 * @file hfmg.cpp
 *
 * @version 1.0
 */
#include "hfmg.h"
#include <cfloat>

using namespace util;

namespace AmctCommon {
inline unsigned int HfmgDivid(unsigned int dividend, unsigned int divisor)
{
    float fDividend = static_cast<float>(dividend);
    float fDivisor = static_cast<float>(divisor);
    unsigned int result = static_cast<unsigned int>(ceil(fDividend / fDivisor));
    return result;
}

template <typename T>
void CalScaleOffset(T max, T min, float &scaleCpu, int &offsetCpu, const HfmgAlgoParam &hfmgParam)
{
    const int limitBound = static_cast<int>(pow(BASE, hfmgParam.quantBitNum - 1));
    if (hfmgParam.withOffset) {
        scaleCpu = static_cast<float>((max - min) / (pow(BASE, hfmgParam.quantBitNum) - 1));
        if (scaleCpu < DBL_EPSILON) {
            scaleCpu = 1.0f;
        }
        offsetCpu = -limitBound - static_cast<int>(round(min / scaleCpu));
    } else {
        scaleCpu = static_cast<float>(
            max > 0 ? max / ((limitBound + MINUS_ONE) + DBL_EPSILON) : -max / (limitBound + DBL_EPSILON));
        if (scaleCpu < DBL_EPSILON) {
            scaleCpu = 1.0f;
        }
        offsetCpu = 0;
    }
}


template <typename T>
void HfmgDistributeInter(int nbins, T dataMin, T binWidth, std::vector<DataBin<T>> &dataBins,
    const InputData<T> &inputData)
{
    T lowerBound;
    T higherBound;
    unsigned int numOfBins = static_cast<unsigned int>(nbins);
    bool minMaxEqualFlag = false;
    if (binWidth < FLT_EPSILON) {
        minMaxEqualFlag = true;
    }
    // initialize all the bins;
    for (unsigned int index = 0; index < numOfBins; index++) {
        lowerBound = dataMin + index * binWidth;
        higherBound = dataMin + (index + 1) * binWidth;
        dataBins.emplace_back(DataBin<T>(0, lowerBound, higherBound));
    }
    if (!minMaxEqualFlag) {
        for (unsigned int index = 0; index < inputData.size; index++) {
            unsigned int binIndex = static_cast<unsigned int>(floor((inputData.in[index] - dataMin) / binWidth));
            if (binIndex >= numOfBins) {
                dataBins[numOfBins - 1].count += 1;
            } else {
                dataBins[binIndex].count += 1;
            }
        }
    } else {
        // when min and max arq equal, count all the elements to the middle bin of dataBins;
        unsigned int binIndex = static_cast<unsigned int>(numOfBins / BASE);
        dataBins[binIndex].count += inputData.size;
    }
}

template<class T>
int HfmgDistribute(int nbins, std::vector<DataBin<T>> &dataBins, const InputData<T> &inputData)
{
    if (inputData.in == nullptr) {
        return AmctCommon::NULL_PTR_ERROR;
    }
    T dataMin = *std::min_element(inputData.in, inputData.in + inputData.size);
    T dataMax = *std::max_element(inputData.in, inputData.in + inputData.size);
    if (nbins == 0) {
        return AmctCommon::ZERO_DIVISION_ERROR;
    } else {
        T binWidth = (dataMax - dataMin) / nbins;
        HfmgDistributeInter<T>(nbins, dataMin, binWidth, dataBins, inputData);
    }
    return AmctCommon::SUCCESS;
}

template <typename T>
void HfmgMergeInter(std::vector<DataBin<T>>& dataBins, std::vector<DataBin<T>>& mergedDataBins,
    bool sameRangeFlag, T mergedDataMin, T mergedBinWidth)
{
    unsigned int numOfBins = dataBins.size();
    if (sameRangeFlag) {
        // do not need to change data range;
        for (unsigned int binIndex = 0; binIndex < numOfBins; binIndex++) {
            mergedDataBins[binIndex].count += dataBins[binIndex].count;
        }
    } else {
        // reallocate the count of existed bins into the new mergedDataBins
        // here assumes that data are uniform differential distributed in the bins
        for (unsigned int binIndex = 0; binIndex < numOfBins; binIndex++) {
            T lowerBound;
            T higherBound;
            unsigned int binIndexOfLower;
            unsigned int binIndexOfHigher;
            unsigned int count = dataBins[binIndex].count;
            lowerBound = dataBins[binIndex].lowerBound;
            higherBound = dataBins[binIndex].higherBound;
            binIndexOfLower = static_cast<unsigned int>(floor((lowerBound - mergedDataMin) / mergedBinWidth));
            binIndexOfHigher = static_cast<unsigned int>(floor((higherBound - mergedDataMin) / mergedBinWidth));
            binIndexOfHigher = (binIndexOfHigher >= numOfBins) ? numOfBins - 1 : binIndexOfHigher;
            if (binIndexOfLower == binIndexOfHigher) {
                // original bin just include by a merged bin
                mergedDataBins[binIndexOfLower].count += count;
            } else {
                // original bin across two merged bin, assume that data is uniformly distributed in bin；
                T scaleOfBin = (mergedDataBins[binIndexOfLower].higherBound - dataBins[binIndex].lowerBound) / \
                    (dataBins[binIndex].higherBound - dataBins[binIndex].lowerBound);
                unsigned int countOfLower = static_cast<unsigned int>(floor(scaleOfBin * dataBins[binIndex].count));
                unsigned int countOfHigher = dataBins[binIndex].count - countOfLower;
                mergedDataBins[binIndexOfLower].count += countOfLower;
                mergedDataBins[binIndexOfHigher].count += countOfHigher;
            }
        }
    }
}


template<class T>
int HfmgMerge(int nbins, std::vector<DataBin<T>>& dataBins, const InputData<T>& inputData)
{
    if (inputData.in == nullptr) {
        return AmctCommon::NULL_PTR_ERROR;
    }
    if (dataBins.size() == 0) {
        return HfmgDistribute<T>(nbins, dataBins, inputData);
    }
    // merge the new data into existed dataBins
    bool sameRangeFlag = false;
    T currentDataMin = *std::min_element(inputData.in, inputData.in + inputData.size);
    T currentDataMax = *std::max_element(inputData.in, inputData.in + inputData.size);
    T dataMin = dataBins.front().lowerBound;
    T dataMax = dataBins.back().higherBound;
    T mergedDataMin = std::min(currentDataMin, dataMin);
    T mergedDataMax = std::max(currentDataMax, dataMax);
    T mergedBinWidth;
    if (nbins == 0) {
        return AmctCommon::ZERO_DIVISION_ERROR;
    } else {
        mergedBinWidth = (mergedDataMax - mergedDataMin) / nbins;
    }
    std::vector<DataBin<T>> mergedDataBins;
    // when mergedDataMax equal last data max and mergedDataMin equal last data min;
    // means do not need to change data range;
    if (fabs(mergedDataMax - dataMax) < FLT_EPSILON && fabs(mergedDataMin - dataMin) < FLT_EPSILON) {
        sameRangeFlag = true;
    }
    // generate the Histogram of current batch data
    HfmgDistributeInter<T>(nbins, mergedDataMin, mergedBinWidth, mergedDataBins, inputData);
    // merge the existed dataBins and the new bins
    HfmgMergeInter<T>(dataBins, mergedDataBins, sameRangeFlag, mergedDataMin, mergedBinWidth);
    dataBins.swap(mergedDataBins);

    return AmctCommon::SUCCESS;
}


template<class T>
int HfmgCalculateLoss(std::vector<DataBin<T>>& dataBins, unsigned int dstNumOfBins, unsigned int binIndexLeft,
    unsigned int binIndexRight, float& l2Loss)
{
    unsigned int numOfBins = static_cast<unsigned int>(dataBins.size());
    T dataMin = dataBins.front().lowerBound;
    T dataMax = dataBins.back().higherBound;
    T binWidth = (dataMax - dataMin) / numOfBins;
    if (dstNumOfBins == 0) {
        return AmctCommon::ZERO_DIVISION_ERROR;
    }
    T dstBinWidth = (binWidth * ((binIndexRight - binIndexLeft) + 1)) / dstNumOfBins;
    if (dstBinWidth < FLT_EPSILON) {
        l2Loss = 0.0;
        return AmctCommon::SUCCESS;
    }
    for (unsigned int index = 0; index < numOfBins; index++) {
        // distances from the beginning of first dst_bin to the beginning and end of src_bin
        T srcBinBegin = (index - binIndexLeft) * binWidth;
        T srcBinEnd = srcBinBegin + binWidth;
        T deltaBegin;
        T deltaEnd;
        unsigned int temp1 = static_cast<unsigned int>(
            std::max(0, static_cast<int>(floor(srcBinBegin / dstBinWidth))));
        unsigned int temp2 = static_cast<unsigned int>(
            std::max(0, static_cast<int>(floor(srcBinEnd / dstBinWidth))));
        unsigned int dstBinBeginIndex = std::min(dstNumOfBins - 1, temp1);
        unsigned int dstBinEndIndex = std::min(dstNumOfBins - 1, temp2);
        T dstBinBeginCenter = dstBinBeginIndex * dstBinWidth + dstBinWidth / BASE;
        T density = dataBins[index].count / binWidth;
        if (dstBinBeginIndex == dstBinEndIndex) {
            // if src_bin is entirely within 1 dst_bin
            deltaBegin = srcBinBegin - dstBinBeginCenter;
            deltaEnd = srcBinEnd - dstBinBeginCenter;
            l2Loss += static_cast<float>(GetNorm(deltaBegin, deltaEnd, density));
        } else {
            // the left dst bin and src bin overlapping parts
            deltaBegin = srcBinBegin - dstBinBeginCenter;
            deltaEnd = dstBinWidth / BASE;
            l2Loss += static_cast<float>(GetNorm(deltaBegin, deltaEnd, density));
            // middle full dst bin loss
            l2Loss += ((dstBinEndIndex - dstBinBeginIndex) - 1) * static_cast<float>(GetNorm(
                -dstBinWidth / BASE, dstBinWidth / BASE, density));
            // the right dst bin and src bin overlapping parts
            T dstBinEndCenter = dstBinEndIndex * dstBinWidth + dstBinWidth / BASE;
            deltaBegin = -dstBinWidth / BASE;
            deltaEnd = srcBinEnd - dstBinEndCenter;
            l2Loss += static_cast<float>(GetNorm(deltaBegin, deltaEnd, density));
        }
    }
    return AmctCommon::SUCCESS;
}


template <typename T>
void HfmgGetSearchRange(std::vector<DataBin<T>>& dataBins,
    std::vector<std::pair<unsigned int, unsigned int>>& searchRange)
{
    unsigned int left = 0;
    unsigned int right = static_cast<unsigned int>(dataBins.size() - 1);

    unsigned int totalNum = 0;
    for (auto item : dataBins) {
        totalNum += item.count;
    }
    // stepSize must be >= 1
    unsigned int tmpStepSize = HfmgDivid(totalNum, STEP_DIVISOR);
    unsigned int stepSize = tmpStepSize > 1 ? tmpStepSize : 1;
    unsigned int minBins = HfmgDivid(static_cast<unsigned int>(dataBins.size()), MIN_BIN_RATIO);
    // specify the search range first
    while (right - left > minBins) {
        unsigned int tempLeft = left;
        unsigned int tempRight = right;
        unsigned int leftSum = 0;
        unsigned int rightSum = 0;
        while (tempLeft < tempRight && leftSum < stepSize) {
            leftSum += dataBins[tempLeft].count;
            tempLeft += 1;
        }
        while (tempLeft < tempRight && rightSum < stepSize) {
            rightSum += dataBins[tempRight].count;
            tempRight -= 1;
        }
        if ((right - tempRight) >= (tempLeft - left)) {
            right = tempRight;
        } else {
            left = tempLeft;
        }
        searchRange.push_back(std::make_pair(left, right));
    }
}


template<class T>
int HfmgCompute(std::vector<DataBin<T>>& dataBins, float& scale, int& offset, const HfmgAlgoParam& hfmgParam)
{
    if (dataBins.size() == 0) {
        return AmctCommon::CONTAINER_EMPTY_ERROR;
    }
    std::vector<std::pair<unsigned int, unsigned int>> searchRange;
    HfmgGetSearchRange<T>(dataBins, searchRange);

    std::vector<float> allL2Loss(searchRange.size());
    // start to calculate the l2 loss of all candidates
    unsigned int dstNumOfBins = static_cast<unsigned int>(pow(BASE, hfmgParam.quantBitNum));
    for (size_t index = 0; index < searchRange.size(); index++) {
        float l2Loss = 0.0;
        auto range = searchRange[index];
        (void)HfmgCalculateLoss(dataBins, dstNumOfBins, range.first, range.second, l2Loss);
        allL2Loss[index] = l2Loss;
    }
    size_t bestMaxIndex = static_cast<size_t>(std::distance(
        std::begin(allL2Loss), std::min_element(std::begin(allL2Loss), std::end(allL2Loss))));
    unsigned int bestLeft = searchRange[bestMaxIndex].first;
    unsigned int bestRight = searchRange[bestMaxIndex].second;
    T dataMin = dataBins.front().lowerBound;
    T dataMax = dataBins.back().higherBound;
    T binWidth = (dataMax - dataMin) / dataBins.size();
    T clipMin = dataMin + bestLeft * binWidth;
    T clipMax = dataMin + (bestRight + 1) * binWidth;

    if (hfmgParam.withOffset) {
        clipMax = clipMax > 0 ? clipMax : 0;
        clipMin = clipMin < 0 ? clipMin : 0;
    } else {
        clipMax = fabs(clipMax) > fabs(clipMin) ? clipMax : clipMin;
        clipMin = 0;
    }

    CalScaleOffset(clipMax, clipMin, scale, offset, hfmgParam);
    return AmctCommon::SUCCESS;
}


template <class T>
Status ActArqCalibration(T inputMin, T inputMax, const FloatData &scale, const IntData &offset,
    const HfmgAlgoParam& hfmgParam)
{
    float scaleTemp;
    int offsetTemp;
    if (hfmgParam.withOffset) {
        inputMax = inputMax > 0 ? inputMax : 0;
        inputMin = inputMin < 0 ? inputMin : 0;
    } else {
        inputMax = fabs(inputMax) > fabs(inputMin) ? static_cast<T>(fabs(inputMax)) : static_cast<T>(fabs(inputMin));
        inputMin = 0;
    }
    CalScaleOffset(inputMax, inputMin, scaleTemp, offsetTemp, hfmgParam);
    int ret = util::ProcessScale(scaleTemp);
    if (ret != AmctCommon::SUCCESS) {
        return ret;
    }
    scale.data[0] = scaleTemp;
    offset.data[0] = offsetTemp;

    return AmctCommon::SUCCESS;
}

template Status ActArqCalibration(float inputMin, float inputMax, const FloatData &scale, const IntData &offset,
    const HfmgAlgoParam& hfmgParam);

template Status ActArqCalibration(double inputMin, double inputMax, const FloatData &scale, const IntData &offset,
    const HfmgAlgoParam& hfmgParam);

template void CalScaleOffset(float max, float min, float& scaleCpu, int& offsetCpu, const HfmgAlgoParam& hfmgParam);
template void CalScaleOffset(double max, double min, float& scaleCpu, int& offsetCpu, const HfmgAlgoParam& hfmgParam);

template int HfmgDistribute(int nbins, std::vector<DataBin<float>>& dataBins,
    const InputData<float>& inputData);
template int HfmgDistribute(int nbins, std::vector<DataBin<double>>& dataBins,
    const InputData<double>& inputData);

template int HfmgMerge(int nbins, std::vector<DataBin<float>>& dataBins,
    const InputData<float>& inputData);
template int HfmgMerge(int nbins, std::vector<DataBin<double>>& dataBins,
    const InputData<double>& inputData);

template int HfmgCompute(std::vector<DataBin<float>>& dataBins, float& scale,
    int& offset, const HfmgAlgoParam& hfmgParam);
template int HfmgCompute(std::vector<DataBin<double>>& dataBins, float& scale,
    int& offset, const HfmgAlgoParam& hfmgParam);

template void HfmgMergeInter(std::vector<DataBin<float>>& dataBins, std::vector<DataBin<float>>& mergedDataBins,
    bool sameRangeFlag, float mergedDataMin, float mergedBinWidth);
template void HfmgMergeInter(std::vector<DataBin<double>>& dataBins, std::vector<DataBin<double>>& mergedDataBins,
    bool sameRangeFlag, double mergedDataMin, double mergedBinWidth);
}
