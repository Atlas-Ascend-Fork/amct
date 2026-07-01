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

/*
 * KernelHiFloat4Cast — FP -> HiFloat4 -> FP fake-quant (block-scaling, no LUT)
 *
 * This is the Ascend port of the pure-python reference `quant_hifx`
 * (amct_pytorch/common/utils/hifx4_quant/quant_dequant.py). It reproduces the exact
 * math, step by step, so the kernel output matches the CPU reference to ~0 error
 * (the acceptance red-line, P1.6). Unlike hifloat8_cast it uses NO lookup table:
 * HiF4's scale is a per-64-element block reduction (arithmetic), and the S1P2 element
 * mantissa has only 8 magnitudes, so a LUT buys nothing (see P1.7).
 *
 * Layout: the host permutes the quant dim to the last axis, pads it to a multiple of 64,
 * and flattens, so the device sees a contiguous buffer whose length is a multiple of 64
 * and every consecutive 64 elements form one independent block. Within a block, element
 * index e (0..63) maps to: L3 sub-group = e / 4 (16 groups), L2 sub-group = e / 8 (8 groups).
 * This matches the reference reshape unflatten(qdim, (-1, 8, 2, 4)).
 *
 * Compute is done in fp32 (reference uses force_fp32=True) with bf16 rounding replicated at
 * the exact points the reference rounds (div7, scale_factor, reciprocal). The whole tile is
 * Cast to fp32 once, processed block-by-block with scalar ops, then Cast back to the output
 * dtype. Correctness-first: a scalar inner loop is fine because MTE dominates and the format
 * is a simulation aid, not a perf path.
 *
 * UB budget (tileLength = T elements, sizeof(T) = 2 for fp16/bf16):
 *   inQueue  : T * 2B      (input tile)
 *   outQueue : T * 2B      (output tile)
 *   f32Buf   : T * 4B      (fp32 work buffer)
 *   total    : T * 8B      + small per-block stack arrays (blk[64] + max/scale[16] fp32)
 * Single-buffered (TQue depth = 1): the inner loop is scalar, so double-buffering the queues
 * would not overlap with MTE. Host aligns T DOWN to a multiple of 64 (a block is never split).
 *
 * NOTE (must validate on device): the float bit-tricks below (Pow2i / FloorLog2Pos / Bf16Round)
 * assume IEEE-754 binary32 and are used instead of libm (floorf/log2f/exp2f) which is not
 * guaranteed on the AI Core scalar unit. Cross-check against reference.py on the 910C; if any
 * intrinsic differs, adjust these helpers only.
 */

#include "kernel_operator.h"
#include "hifloat4_cast_tiling.h"

namespace {
// ── scalar fp32 helpers (no libm; IEEE-754 binary32 bit tricks) ──────────────

union F32Bits {
    float f;
    uint32_t u;
};

// 2^k for integer k within the float normal range (k in ~[-126, 127]).
__aicore__ inline float Pow2i(int32_t k)
{
    F32Bits v;
    v.u = static_cast<uint32_t>(static_cast<int32_t>(127) + k) << 23;  // 127: fp32 exponent bias, 23: mantissa bits
    return v.f;
}

// floor(log2(x)) for x > 0: the unbiased exponent field of a normalized binary32.
__aicore__ inline int32_t FloorLog2Pos(float x)
{
    F32Bits v;
    v.f = x;
    return static_cast<int32_t>((v.u >> 23) & 0xFFu) - 127;  // 23: mantissa bits, 127: exponent bias
}

// x rounded to bfloat16 precision then back to fp32 (round-to-nearest-even on the low 16 bits).
// Mirrors the reference's `.to(torch.bfloat16).to(fp32)` steps.
__aicore__ inline float Bf16Round(float x)
{
    F32Bits v;
    v.f = x;
    uint32_t lsb = (v.u >> 16) & 1u;
    v.u += 0x7FFFu + lsb;
    v.u &= 0xFFFF0000u;
    return v.f;
}

// floor(v) for v >= 0.
__aicore__ inline float FloorNonNeg(float v)
{
    return static_cast<float>(static_cast<int32_t>(v));
}

// round-half-to-even for v >= 0 (matches torch.round).
__aicore__ inline float RoundRNE(float v)
{
    int32_t fl = static_cast<int32_t>(v);
    float d = v - static_cast<float>(fl);
    if (d < 0.5f) {
        return static_cast<float>(fl);
    }
    if (d > 0.5f) {
        return static_cast<float>(fl + 1);
    }
    return (fl % 2 == 0) ? static_cast<float>(fl) : static_cast<float>(fl + 1);  // tie -> even
}

// scale_factor: max_lv1/7 rounded to bf16, clipped, then to a 7-bit and finally E6M2 (2-bit) mantissa.
__aicore__ inline float ComputeScaleFactor(float maxLv1)
{
    float div7 = Bf16Round(1.0f / 7.0f);
    float sf = Bf16Round(maxLv1 * div7);
    float lo = Pow2i(-48);      // 2^-48 lower clip
    float hi = 49152.0f;        // E6M2 upper clip
    if (sf < lo) {
        sf = lo;
    }
    if (sf > hi) {
        sf = hi;
    }
    // round to 7-bit mantissa (reference: tobf16(rint))
    int32_t e = FloorLog2Pos(sf);
    float mant = sf * Pow2i(-e) * 128.0f;                 // sf / 2^e * 2^7
    sf = RoundRNE(mant) / 128.0f * Pow2i(e);
    // round to E6M2 (2-bit mantissa)
    e = FloorLog2Pos(sf);
    sf = RoundRNE(sf * Pow2i(2 - e)) * Pow2i(e - 2);
    return sf;
}
}  // namespace

template <typename T>
class KernelHiFloat4Cast {
public:
    __aicore__ inline KernelHiFloat4Cast(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const __gm__ HiFloat4CastTilingData *tiling)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        tileLength_ = tiling->tileLength;
        total_ = (blockIdx < tiling->blockNum - 1) ? tiling->numPerCore : tiling->tailNumLastCore;
        tileNum_ = (total_ + tileLength_ - 1) / tileLength_;
        tailTile_ = static_cast<uint32_t>(total_ - static_cast<uint64_t>(tileLength_) * (tileNum_ - 1));

        uint64_t offset = static_cast<uint64_t>(blockIdx) * tiling->numPerCore;
        inputGm_.SetGlobalBuffer((__gm__ T *)input + offset, total_);
        outputGm_.SetGlobalBuffer((__gm__ T *)output + offset, total_);

        pipe_->InitBuffer(inQueue_, 1, tileLength_ * sizeof(T));
        pipe_->InitBuffer(outQueue_, 1, tileLength_ * sizeof(T));
        pipe_->InitBuffer(f32Buf_, tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < tileNum_; i++) {
            uint32_t count = (i == tileNum_ - 1) ? tailTile_ : tileLength_;
            CopyIn(i, count);
            Compute(count);
            CopyOut(i, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t tileIdx, uint32_t count)
    {
        auto xLocal = inQueue_.AllocTensor<T>();
        AscendC::DataCopyExtParams cp{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, inputGm_[tileIdx * static_cast<uint64_t>(tileLength_)], cp, pad);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t tileIdx, uint32_t count)
    {
        auto yLocal = outQueue_.DeQue<T>();
        AscendC::DataCopyExtParams cp{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm_[tileIdx * static_cast<uint64_t>(tileLength_)], yLocal, cp);
        outQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto xLocal = inQueue_.DeQue<T>();
        auto yLocal = outQueue_.AllocTensor<T>();
        auto f32 = f32Buf_.Get<float>();

        // Whole tile up to fp32 once (reference computes in fp32).
        AscendC::Cast(f32, xLocal, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();

        // count is a multiple of 64 (host guarantees block-aligned tiles).
        uint32_t numBlocks = count / HIF4_BLOCK_SIZE;
        for (uint32_t b = 0; b < numBlocks; b++) {
            QuantizeBlock(f32, b * HIF4_BLOCK_SIZE);
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yLocal, f32, AscendC::RoundMode::CAST_RINT, count);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    // One 64-element block: reproduce quant_hifx exactly, in place in the fp32 tile buffer.
    __aicore__ inline void QuantizeBlock(AscendC::LocalTensor<float> &f32, uint32_t base)
    {
        float blk[HIF4_BLOCK_SIZE];
        float maxLv3[HIF4_LV3_PER_BLOCK];
        float maxLv2[HIF4_LV2_PER_BLOCK];
        float lv2[HIF4_LV2_PER_BLOCK];
        float lv3[HIF4_LV3_PER_BLOCK];

        for (uint32_t g = 0; g < HIF4_LV3_PER_BLOCK; g++) {
            maxLv3[g] = 0.0f;
        }
        for (uint32_t g = 0; g < HIF4_LV2_PER_BLOCK; g++) {
            maxLv2[g] = 0.0f;
        }

        // Pass 1: load + 3-level max reduction.
        float maxLv1 = 0.0f;
        for (uint32_t e = 0; e < HIF4_BLOCK_SIZE; e++) {
            float v = f32.GetValue(base + e);
            blk[e] = v;
            float a = v < 0.0f ? -v : v;
            uint32_t g3 = e / HIF4_LV3_GROUP;
            uint32_t g2 = e / HIF4_LV2_GROUP;
            if (a > maxLv3[g3]) {
                maxLv3[g3] = a;
            }
            if (a > maxLv2[g2]) {
                maxLv2[g2] = a;
            }
            if (a > maxLv1) {
                maxLv1 = a;
            }
        }

        // Shared scale (E6M2) + reciprocal.
        float sf = ComputeScaleFactor(maxLv1);
        float rec = Bf16Round(1.0f / sf);

        // L2 micro-exponents: exp2(floor(clip(maxLv2*rec, 0, 4) / 4)) -> {1, 2}
        for (uint32_t g = 0; g < HIF4_LV2_PER_BLOCK; g++) {
            float t = maxLv2[g] * rec;
            if (t < 0.0f) {
                t = 0.0f;
            }
            if (t > 4.0f) {
                t = 4.0f;
            }
            lv2[g] = Pow2i(static_cast<int32_t>(FloorNonNeg(t / 4.0f)));
        }
        // L3 micro-exponents: exp2(floor(clip(maxLv3*rec/lv2, 0, 2) / 2)) -> {1, 2}
        for (uint32_t g = 0; g < HIF4_LV3_PER_BLOCK; g++) {
            float t = maxLv3[g] * rec / lv2[g / 2];
            if (t < 0.0f) {
                t = 0.0f;
            }
            if (t > 2.0f) {
                t = 2.0f;
            }
            lv3[g] = Pow2i(static_cast<int32_t>(FloorNonNeg(t / 2.0f)));
        }

        // Pass 2: quantize mantissa to S1P2 (1/4 steps, saturate to 1.75) and rebuild.
        for (uint32_t e = 0; e < HIF4_BLOCK_SIZE; e++) {
            float v = blk[e];
            float s = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
            float a = v < 0.0f ? -v : v;
            uint32_t g3 = e / HIF4_LV3_GROUP;
            uint32_t g2 = e / HIF4_LV2_GROUP;
            float m = a / lv2[g2] / lv3[g3] * rec;
            m = FloorNonNeg(m * 4.0f + 0.5f) / 4.0f;  // round-half-up to nearest 1/4 (man_bits=3)
            if (m >= 2.0f) {
                m = 1.75f;                            // S1P2 saturation
            }
            f32.SetValue(base + e, s * m * lv2[g2] * lv3[g3] * sf);
        }
    }

private:
    AscendC::TPipe *pipe_;

    uint32_t tileLength_ = 0;
    uint64_t total_ = 0;
    uint64_t tileNum_ = 0;
    uint32_t tailTile_ = 0;

    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> f32Buf_;
};

extern "C" __global__ __vector__ void hifloat4_cast_kernel(GM_ADDR input, GM_ADDR output, GM_ADDR tiling)
{
    auto *t = (__gm__ HiFloat4CastTilingData *)tiling;
    AscendC::TPipe pipe;
    if (t->dtypeFlag == HIF4_BF16) {
        KernelHiFloat4Cast<bfloat16_t> op(&pipe);
        op.Init(input, output, t);
        op.Process();
    } else {
        KernelHiFloat4Cast<half> op(&pipe);
        op.Init(input, output, t);
        op.Process();
    }
}
