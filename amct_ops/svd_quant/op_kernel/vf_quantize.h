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
 */

/*!
 * \file vf_quantize.h
 * \brief
 */

#ifndef VF_QUANTIZE_H_
#define VF_QUANTIZE_H_

#include "kernel_operator.h"

#define ALIGN_DOWN(x, y) ((x) / (y) * (y))
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

using namespace AscendC;

static constexpr int32_t BUFFER_NUM = 2;
static constexpr uint16_t MAX_EXP_BF16 = 0x7F80;
static constexpr int16_t BF16_MANTISSA_LEN = 7;
static constexpr uint16_t MAX_CLIP_VAL = 0x7F00;
static constexpr uint16_t MIN_CLIP_VAL = 0x0100;
static constexpr uint32_t DIGIT_TWO = 2;

static constexpr MicroAPI::CastTrait castTraitBF16ToFP4 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

template <typename T>
class QuantizeFP4 {
public:
    __aicore__ inline QuantizeFP4() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR scale, uint32_t inputSize, TPipe *pipe);
    __aicore__ inline void Process();

private:
    TQue<TPosition::VECIN, BUFFER_NUM> inQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQue_, scaleQue_;

    GlobalTensor<T> inputGm_;
    GlobalTensor<uint8_t> outGm_, scaleGm_;

    TPipe *tpipe_;
    uint32_t groupSize_ = 32;
    uint32_t coreBlocks_ = 0;
    uint32_t ubElements_ = 0;
    uint32_t numIters_ = 0;
    uint32_t ubBlocks_ = 0;

    struct VecRegisters {
        MicroAPI::UnalignReg u0;
        MicroAPI::RegTensor<T> inReg;
        MicroAPI::RegTensor<uint16_t> infReg;
        MicroAPI::RegTensor<uint16_t> maxAbsReg;
        MicroAPI::RegTensor<uint16_t> outScaleReg;
        MicroAPI::RegTensor<uint16_t> maxReg;
        MicroAPI::RegTensor<uint16_t> minReg;
        MicroAPI::RegTensor<uint16_t> realScaleReg;
        MicroAPI::RegTensor<uint8_t> outReg;
        MicroAPI::RegTensor<int16_t> idxReg;
        MicroAPI::RegTensor<int16_t> evenIdxReg;
        MicroAPI::RegTensor<int16_t> oddIdxReg;
        MicroAPI::RegTensor<uint16_t> evenMaxReg;
        MicroAPI::RegTensor<uint16_t> oddMaxReg;

        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg scaleMask;
        MicroAPI::MaskReg outMask;
    };

    __aicore__ inline void CopyIn(uint32_t blockCount, uint32_t offset);
    __aicore__ inline void Compute(uint32_t blockCount);
    __aicore__ inline void CopyOut(uint32_t blockCount, uint32_t outOffset, uint32_t scaleOffset);

    __aicore__ inline void VFCompute(__local_mem__ T *inUb, __local_mem__ uint8_t *outUb,
        __local_mem__ uint8_t *scaleUb, uint16_t vecLen, uint16_t size, uint16_t loopCount, uint16_t outSize,
        uint16_t scaleNum);
    __aicore__ inline void InitConstants(VecRegisters &regs, uint32_t sreg2);
};

template <typename T>
inline __aicore__ void QuantizeFP4<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR scale, uint32_t inputSize, TPipe *pipe) {
    SetAtomicNone();
    tpipe_ = pipe;
    uint32_t numCores = static_cast<uint32_t>(GetBlockNum() * GetSubBlockNum());
    uint32_t coreId = static_cast<uint32_t>(GetBlockIdx());
    uint32_t numBlocks = inputSize / groupSize_;
    if (coreId >= numBlocks) {
        return;
    }
    coreBlocks_ = numBlocks / numCores;
    uint32_t tailBlocks = numBlocks % numCores;
    uint32_t inOffset = 0;
    uint32_t outOffset = 0;
    uint32_t scaleOffset = 0;
    if (coreId < tailBlocks) {
        coreBlocks_++;
        scaleOffset = coreId * coreBlocks_;
        inOffset = scaleOffset * groupSize_;
        outOffset = inOffset / DIGIT_TWO;
    } else {
        scaleOffset = tailBlocks * (coreBlocks_ + 1) + (coreId - tailBlocks) * coreBlocks_;
        inOffset = scaleOffset * groupSize_;
        outOffset = inOffset / DIGIT_TWO;
    }

    uint32_t numElements = coreBlocks_ * groupSize_;
    inputGm_.SetGlobalBuffer((__gm__ T *)input + inOffset, numElements);
    outGm_.SetGlobalBuffer((__gm__ uint8_t *)output + outOffset, numElements / DIGIT_TWO);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t *)scale + scaleOffset, coreBlocks_);

    outGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    scaleGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

    uint32_t bufCoeff = groupSize_ * static_cast<uint32_t>(sizeof(T)) * BUFFER_NUM + groupSize_ + DIGIT_TWO;
    ubBlocks_ = ALIGN_DOWN((TOTAL_UB_SIZE / bufCoeff), ONE_BLOCK_SIZE);
    ubElements_ = ubBlocks_ * groupSize_;
    tpipe_->InitBuffer(inQue_, BUFFER_NUM, ubElements_ * sizeof(T));
    tpipe_->InitBuffer(outQue_, BUFFER_NUM, ubElements_ / DIGIT_TWO);
    tpipe_->InitBuffer(scaleQue_, BUFFER_NUM, ubBlocks_);
    numIters_ = CEIL_DIV(coreBlocks_, ubBlocks_);
}

template <typename T>
inline __aicore__ void QuantizeFP4<T>::Process() {
    for (uint32_t iter = 0; iter < numIters_; iter++, coreBlocks_ -= ubBlocks_) {
        uint32_t procBlocks = ubBlocks_ < coreBlocks_ ? ubBlocks_ : coreBlocks_;
        uint32_t inOffset = ubElements_ * iter;
        uint32_t outOffset = inOffset / DIGIT_TWO;
        uint32_t scaleOffset = ubBlocks_ * iter;
        CopyIn(procBlocks, inOffset);
        Compute(procBlocks);
        CopyOut(procBlocks, outOffset, scaleOffset);
    }
}

template <typename T>
inline __aicore__ void QuantizeFP4<T>::CopyIn(uint32_t blockCount, uint32_t offset) {
    LocalTensor<T> inTensor = inQue_.AllocTensor<T>();
    DataCopy(inTensor, inputGm_[offset], blockCount * groupSize_);
    inQue_.EnQue(inTensor);
}

template <typename T>
inline __aicore__ void QuantizeFP4<T>::Compute(uint32_t blockCount) {
    LocalTensor<T> inTensor = inQue_.DeQue<T>();
    LocalTensor<uint8_t> outTensor = outQue_.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> scaleTensor = scaleQue_.AllocTensor<uint8_t>();

    __local_mem__ T *inUb = (__local_mem__ T *)inTensor.GetPhyAddr();
    __local_mem__ uint8_t *outUb = (__local_mem__ uint8_t *)outTensor.GetPhyAddr();
    __local_mem__ uint8_t *scaleUb = (__local_mem__ uint8_t *)scaleTensor.GetPhyAddr();

    uint16_t vecLen = static_cast<uint16_t>(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t size = static_cast<uint16_t>(blockCount * groupSize_);
    uint16_t loopCount = CEIL_DIV(size, vecLen);
    uint16_t outSize = vecLen / static_cast<uint16_t>(DIGIT_TWO);
    uint16_t scaleNum = vecLen / static_cast<uint16_t>(groupSize_);

    VFCompute(inUb, outUb, scaleUb, vecLen, size, loopCount, outSize, scaleNum);

    inQue_.FreeTensor(inTensor);
    outQue_.EnQue(outTensor);
    scaleQue_.EnQue(scaleTensor);
}

template <typename T>
inline __aicore__ void QuantizeFP4<T>::CopyOut(uint32_t blockCount, uint32_t outOffset, uint32_t scaleOffset) {
    LocalTensor<uint8_t> outTensor = outQue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> scaleTensor = scaleQue_.DeQue<uint8_t>();
    DataCopyParams copyOutParams = {1, static_cast<uint16_t>(blockCount * groupSize_ / DIGIT_TWO), 0, 0};
    DataCopyParams copyScaleParams = {1, static_cast<uint16_t>(blockCount), 0, 0};
    DataCopyPad(outGm_[outOffset], outTensor, copyOutParams);
    DataCopyPad(scaleGm_[scaleOffset], scaleTensor, copyScaleParams);

    outQue_.FreeTensor(outTensor);
    scaleQue_.FreeTensor(scaleTensor);
}

template <typename T>
inline __aicore__ void QuantizeFP4<T>::VFCompute(__local_mem__ T *inUb, __local_mem__ uint8_t *outUb,
    __local_mem__ uint8_t *scaleUb, uint16_t vecLen, uint16_t size, uint16_t loopCount, uint16_t outSize,
    uint16_t scaleNum) {
    __VEC_SCOPE__ {
        uint32_t sreg1 = static_cast<uint32_t>(size);
        uint32_t sreg2 = static_cast<uint32_t>(scaleNum);
        VecRegisters regs;
        InitConstants(regs, sreg2);
        for (uint16_t i = 0; i < loopCount; i++) {
            regs.mask = MicroAPI::UpdateMask<uint16_t>(sreg1);
            MicroAPI::DataCopy(regs.inReg, inUb + i * vecLen);
            MicroAPI::And(regs.maxAbsReg, (MicroAPI::RegTensor<uint16_t> &)(regs.inReg), regs.infReg, regs.mask);
            MicroAPI::ReduceMaxWithDataBlock(regs.maxAbsReg, regs.maxAbsReg, regs.mask);
            MicroAPI::Gather(regs.evenMaxReg, regs.maxAbsReg, (MicroAPI::RegTensor<uint16_t> &)regs.evenIdxReg);
            MicroAPI::Gather(regs.oddMaxReg, regs.maxAbsReg, (MicroAPI::RegTensor<uint16_t> &)regs.oddIdxReg);
            MicroAPI::Max(regs.maxAbsReg, regs.evenMaxReg, regs.oddMaxReg, regs.mask);

            MicroAPI::Max(regs.maxAbsReg, regs.maxAbsReg, regs.minReg, regs.mask);
            MicroAPI::Min(regs.maxAbsReg, regs.maxAbsReg, regs.maxReg, regs.mask);

            MicroAPI::Sub(regs.maxAbsReg, regs.maxAbsReg, regs.minReg, regs.mask);
            MicroAPI::ShiftRights(regs.outScaleReg, regs.maxAbsReg, BF16_MANTISSA_LEN, regs.mask);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint8_t> &)regs.outScaleReg, regs.outScaleReg);

            MicroAPI::Sub(regs.maxAbsReg, regs.maxReg, regs.maxAbsReg, regs.mask);
            MicroAPI::Gather(regs.realScaleReg, regs.maxAbsReg, (MicroAPI::RegTensor<uint16_t> &)regs.idxReg);
            MicroAPI::Mul(regs.inReg, regs.inReg, (MicroAPI::RegTensor<T> &)regs.realScaleReg, regs.mask);
            MicroAPI::Cast<fp4x2_e2m1_t, T, castTraitBF16ToFP4>(
                (MicroAPI::RegTensor<fp4x2_e2m1_t> &)regs.outReg, regs.inReg, regs.mask);

            __local_mem__ uint8_t *scaleAddr = scaleUb + i * scaleNum;
            MicroAPI::DataCopyUnAlign(
                scaleAddr, (MicroAPI::RegTensor<uint8_t> &)regs.outScaleReg, regs.u0, (uint32_t)scaleNum);
            MicroAPI::DataCopyUnAlignPost(scaleAddr, regs.u0, 0);
            MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                outUb + i * outSize, regs.outReg, regs.outMask);
        }
    }
}

template <typename T>
inline __aicore__ void QuantizeFP4<T>::InitConstants(VecRegisters &regs, uint32_t sreg2) {
    regs.outMask = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    regs.scaleMask = MicroAPI::UpdateMask<uint16_t>(sreg2);

    MicroAPI::Duplicate(regs.infReg, MAX_EXP_BF16);
    MicroAPI::Duplicate(regs.minReg, MIN_CLIP_VAL);
    MicroAPI::Duplicate(regs.maxReg, MAX_CLIP_VAL);

    MicroAPI::Arange(regs.idxReg, (int16_t)0);                                           // 0 1 2 3
    MicroAPI::Interleave(regs.evenIdxReg, regs.oddIdxReg, regs.idxReg, regs.idxReg);     // 00 11 22 33
    MicroAPI::Interleave(regs.idxReg, regs.oddIdxReg, regs.evenIdxReg, regs.evenIdxReg); // 4
    MicroAPI::Interleave(regs.evenIdxReg, regs.oddIdxReg, regs.idxReg, regs.idxReg);     // 8
    MicroAPI::Interleave(regs.oddIdxReg, regs.idxReg, regs.evenIdxReg, regs.evenIdxReg); // 16
    MicroAPI::Interleave(regs.idxReg, regs.evenIdxReg, regs.oddIdxReg, regs.oddIdxReg); // 32 -> 0...0 1...1 2...2 3...3

    MicroAPI::Arange(regs.evenIdxReg, (int16_t)0);
    MicroAPI::Muls(regs.evenIdxReg, regs.evenIdxReg, (int16_t)2, regs.scaleMask);
    MicroAPI::Adds(regs.oddIdxReg, regs.evenIdxReg, (int16_t)1, regs.scaleMask);
}

#endif // VF_QUANTIZE_H_
