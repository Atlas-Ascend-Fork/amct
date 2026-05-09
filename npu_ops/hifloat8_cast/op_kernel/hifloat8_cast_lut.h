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
 * KernelHiFloat8CastLut — LUT 查表实现
 *
 * 与 scalar 实现 (hifloat8_cast_scalar.h) 的差异：
 *
 *  castMode | scalar 每元素指令数 | LUT 每元素指令数 | 优化手段
 *  ---------|----------------|----------------|--------------------------------
 *  0 FP32→  | ~20 条         | ~20 条         | 无变化（A2 无向量位运算）
 *  1 FP16→  | ~20 条         | ~5 条          | 32768-entry 半空间 UB LUT + 符号分离
 *  2 BF16→  | ~20 条         | ~5 条          | 32768-entry 半空间 UB LUT + 符号分离
 *  3 →FP32  | ~15 条         | 1 条           | 256-entry UB LUT 消除分支
 *  4 →FP16  | ~15 条         | 1 条           | 256-entry UB LUT 消除分支
 *  5 →BF16  | ~15 条         | 1 条           | 256-entry UB LUT 消除分支
 *
 * UB 占用（TILE_LENGTH=8192，队列按实际元素宽度分配）：
 *   BF16/FP16 encode:  32KB(LUT16) + 32KB(inQ×2 @2B) + 16KB(outQ×2 @1B) = 80KB  < 256KB
 *   decode FP32:        1KB(LUT8)  + 16KB(inQ×2 @1B) + 64KB(outQ×2 @4B) = 81KB  < 256KB
 *   decode FP16/BF16:   1KB(LUT8)  + 16KB(inQ×2 @1B) + 32KB(outQ×2 @2B) = 49KB  < 256KB
 *
 * 半空间 LUT 原理：
 *   BF16/FP16 编码关于符号位对称：encode(-x) = encode(x) | 0x80（当 encode(x)≠0 时）
 *   只存正半空间（32768 条量级值），compute 时先剥离符号，查表后再叠加符号位。
 *   节省 32KB UB，且 DataCopyPad(32KB) 单次完成（不再需要分两次）。
 *
 * A2 平台约束说明：
 *   Cast<uint32_t, uint16_t> ❌（编译器不支持）→ 无法用 Gather 向量化
 *   ShiftLeft/Right uint16_t/uint32_t ❌
 *   当前 BF16/FP16 encode 仍为标量循环，但每步只需 ~5 条 UB 查表+位运算指令
 */

#pragma once

#include "kernel_operator.h"
#include "hifloat8_cast_tiling.h"

constexpr uint32_t FP32_EXP_BITS_V2 = 8;
constexpr uint32_t FP32_FRAC_BITS_V2 = 23;
constexpr uint32_t FP32_EXP_BIAS_V2 = 127;

constexpr uint32_t FP16_EXP_BITS_V2 = 5;
constexpr uint32_t FP16_FRAC_BITS_V2 = 10;
constexpr uint32_t FP16_EXP_BIAS_V2 = 15;

constexpr uint32_t BF16_EXP_BITS_V2 = 8;
constexpr uint32_t BF16_FRAC_BITS_V2 = 7;
constexpr uint32_t BF16_EXP_BIAS_V2 = 127;

constexpr uint32_t HIF8_SIGN_SHIFT_V2 = 7;
constexpr uint32_t HIF8_DENORMAL_EXP_BIAS_V2 = 23;

// 32768-entry LUT for BF16/FP16 encode (positive half: indexed by absBits = val & 0x7FFF)
// Sign bit is stripped before lookup and re-applied after, halving UB from 64KB→32KB.
constexpr uint32_t LUT16_SIZE = 32768;
// 256-entry LUT for decode (indexed by raw HiF8 uint8_t value)
constexpr uint32_t LUT8_SIZE = 256;

// ── scalar 核心算法（Init 阶段构建 LUT 使用）──────────────────────────────────

constexpr uint32_t HIF8_DOT_TABLE_V2[16] = {
    (0x00 << 24) | (3 << 16) | (0 << 8) | (0 << 4) | 7,
    (0x00 << 24) | (3 << 16) | (1 << 8) | (0 << 4) | 7,
    (0x08 << 24) | (3 << 16) | (3 << 8) | (1 << 4) | 7,
    (0x08 << 24) | (3 << 16) | (3 << 8) | (1 << 4) | 7,
    (0x18 << 24) | (3 << 16) | (4 << 8) | (2 << 4) | 7,
    (0x18 << 24) | (3 << 16) | (4 << 8) | (2 << 4) | 7,
    (0x18 << 24) | (3 << 16) | (4 << 8) | (2 << 4) | 7,
    (0x18 << 24) | (3 << 16) | (4 << 8) | (2 << 4) | 7,
    (0x1c << 24) | (2 << 16) | (4 << 8) | (3 << 4) | 3,
    (0x1c << 24) | (2 << 16) | (4 << 8) | (3 << 4) | 3,
    (0x1c << 24) | (2 << 16) | (4 << 8) | (3 << 4) | 3,
    (0x1c << 24) | (2 << 16) | (4 << 8) | (3 << 4) | 3,
    (0x1e << 24) | (1 << 16) | (4 << 8) | (4 << 4) | 1,
    (0x1e << 24) | (1 << 16) | (4 << 8) | (4 << 4) | 1,
    (0x1e << 24) | (1 << 16) | (4 << 8) | (4 << 4) | 1,
    (0x1e << 24) | (1 << 16) | (4 << 8) | (4 << 4) | 1,
};

__aicore__ inline void GetHif8BitsNumV2(int32_t expNoBias,
                                         uint32_t& dotValue,
                                         uint32_t& expBits,
                                         uint32_t& fracBits)
{
    int32_t absExp = expNoBias < 0 ? -expNoBias : expNoBias;
    if (expNoBias == 0)                  { dotValue = 0x01; expBits = 0; fracBits = 3; }
    else if (absExp == 1)                { dotValue = 0x02; expBits = 1; fracBits = 3; }
    else if (absExp >= 2 && absExp <= 3) { dotValue = 0x04; expBits = 2; fracBits = 3; }
    else if (absExp >= 4 && absExp <= 7) { dotValue = 0x08; expBits = 3; fracBits = 2; }
    else                                 { dotValue = 0x0C; expBits = 4; fracBits = 1; }
}


// HiFloat8 → FP bits（scalar，仅 LUT Init 使用）
__aicore__ inline uint32_t Hif8ToFpBitsV2(uint8_t inData,
                                            uint32_t expBits,
                                            uint32_t fracBits,
                                            uint32_t expBias)
{
    if (inData == 0) return 0;

    uint32_t totalBits = expBits + fracBits + 1;
    uint32_t expMax    = (1u << expBits) - 1;

    if (inData == 0x80) return (expMax << fracBits) | ((1u << fracBits) - 1);
    if (inData == 0x6F) return expMax << fracBits;
    if (inData == 0xEF) return (1u << (totalBits - 1)) | (expMax << fracBits);

    uint8_t dot     = (inData & 0x78) >> 3;
    uint32_t info   = HIF8_DOT_TABLE_V2[dot];

    uint8_t expMask      = (info >> 24) & 0xFF;
    uint8_t fracHiF8Bits = (info >> 16) & 0xFF;
    uint8_t expSignShift = (info >> 8)  & 0xFF;
    uint8_t expHiF8Bits  = (info >> 4)  & 0x0F;
    uint8_t fracMask     = info & 0x0F;

    uint32_t expHiF8 = (inData & expMask) >> fracHiF8Bits;
    uint32_t signExp = (inData >> expSignShift) & 0x01;
    uint32_t fracFpN = inData & fracMask;
    uint32_t expFpN;

    if (expSignShift == 0) {
        expFpN = expBias - (HIF8_DENORMAL_EXP_BIAS_V2 - fracFpN);
        fracFpN = 0;
    } else if (expSignShift == 1) {
        expFpN = expBias;
    } else {
        int32_t expNoBias = (int32_t)((expHiF8 | (1u << (expHiF8Bits - 1))) * (signExp == 1 ? -1 : 1));
        expFpN = (uint32_t)(expNoBias + (int32_t)expBias);
    }

    uint32_t sign = inData >> HIF8_SIGN_SHIFT_V2;
    return (sign << (totalBits - 1)) | (expFpN << fracBits) | (fracFpN << (fracBits - fracHiF8Bits));
}

// ── Kernel 类 ──────────────────────────────────────────────────────────────

class KernelHiFloat8CastLut {
public:
    __aicore__ inline KernelHiFloat8CastLut(AscendC::TPipe* pipe) : pipe_(pipe) {}

    // lut_gm: 预计算好的 LUT，由调用方传入（Python 端一次性生成后复用）
    //   BF16/FP16 encode: 65536-entry uint8_t LUT（64KB），indexed by raw uint16_t bit pattern
    //   decode: NULL（LUT 在 kernel 内部构建，只有 256 条，开销可忽略）
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                 const __gm__ HiFloat8CastTilingData* tiling,
                                 GM_ADDR lut_gm)
    {
        tiling_   = tiling;
        castMode_ = tiling->castMode;
        uint32_t blockIdx = AscendC::GetBlockIdx();

        total_ = (blockIdx < tiling->blockNum - 1)
                 ? tiling->numPerCore : tiling->tailNumLastCore;
        tileNum_ = (total_ + TILE_LENGTH - 1) / TILE_LENGTH;
        tailTile_ = total_ - TILE_LENGTH * (tileNum_ - 1);
        if (tailTile_ == 0 && tileNum_ > 0) { tileNum_--; tailTile_ = TILE_LENGTH; }

        uint64_t offset = (uint64_t)blockIdx * tiling->numPerCore;

        SetTypeParams();
        inputGm_.SetGlobalBuffer((__gm__ uint8_t*)input   + offset * inputBytes_,
                                  total_ * inputBytes_);
        outputGm_.SetGlobalBuffer((__gm__ uint8_t*)output + offset * outputBytes_,
                                   total_ * outputBytes_);

        // 队列按实际元素宽度分配，避免固定 ×4 造成的 UB 浪费
        pipe_->InitBuffer(inQueue_,  DOUBLE_BUFFER, TILE_LENGTH * inputBytes_);
        pipe_->InitBuffer(outQueue_, DOUBLE_BUFFER, TILE_LENGTH * outputBytes_);

        if (castMode_ == 1 || castMode_ == 2) {
            // BF16/FP16 encode: 从 GM 复制预计算半空间 LUT 到 UB
            // LUT16_SIZE=32768 < 65535，单次 DataCopyPad 即可完成（无需分两次）
            pipe_->InitBuffer(lut16Buf_, LUT16_SIZE);
            AscendC::GlobalTensor<uint8_t> lutGm;
            lutGm.SetGlobalBuffer((__gm__ uint8_t*)lut_gm, LUT16_SIZE);
            auto lutUb = lut16Buf_.Get<uint8_t>();
            AscendC::DataCopyParams    cp{1, (uint16_t)LUT16_SIZE, 0, 0};
            AscendC::DataCopyPadParams pp{false, 0, 0, 0};
            AscendC::DataCopyPad(lutUb, lutGm[0], cp, pp);
        }
        if (castMode_ >= 3) {
            // decode: LUT 只有 256 条，在 UB 内直接构建，开销可忽略
            pipe_->InitBuffer(lut8Buf_, LUT8_SIZE * outputBytes_);
            InitLut8();
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum_; i++) {
            uint32_t count = (i == tileNum_ - 1) ? tailTile_ : TILE_LENGTH;
            CopyIn(i, count);
            Compute(count);
            CopyOut(i, count);
        }
    }

private:
    // ── 类型参数设置 ────────────────────────────────────────────────────────

    __aicore__ inline void SetTypeParams()
    {
        switch (castMode_) {
            case 0: expBits_=FP32_EXP_BITS_V2; fracBits_=FP32_FRAC_BITS_V2; expBias_=FP32_EXP_BIAS_V2; inputBytes_=4; outputBytes_=1; break;
            case 1: expBits_=FP16_EXP_BITS_V2; fracBits_=FP16_FRAC_BITS_V2; expBias_=FP16_EXP_BIAS_V2; inputBytes_=2; outputBytes_=1; break;
            case 2: expBits_=BF16_EXP_BITS_V2; fracBits_=BF16_FRAC_BITS_V2; expBias_=BF16_EXP_BIAS_V2; inputBytes_=2; outputBytes_=1; break;
            case 3: expBits_=FP32_EXP_BITS_V2; fracBits_=FP32_FRAC_BITS_V2; expBias_=FP32_EXP_BIAS_V2; inputBytes_=1; outputBytes_=4; break;
            case 4: expBits_=FP16_EXP_BITS_V2; fracBits_=FP16_FRAC_BITS_V2; expBias_=FP16_EXP_BIAS_V2; inputBytes_=1; outputBytes_=2; break;
            case 5: expBits_=BF16_EXP_BITS_V2; fracBits_=BF16_FRAC_BITS_V2; expBias_=BF16_EXP_BIAS_V2; inputBytes_=1; outputBytes_=2; break;
            default: break;
        }
    }

    // 256-entry LUT for decode (HiF8 → FP32/FP16/BF16)
    // key  = HiF8 uint8_t value (0-255)
    // value = target type bits（outputBytes_ 字节，小端存储）
    __aicore__ inline void InitLut8()
    {
        if (outputBytes_ == 4) {
            auto lut = lut8Buf_.Get<uint32_t>();
            for (uint32_t i = 0; i < LUT8_SIZE; i++) {
                lut.SetValue(i, Hif8ToFpBitsV2((uint8_t)i, expBits_, fracBits_, expBias_));
            }
        } else {
            auto lut = lut8Buf_.Get<uint16_t>();
            for (uint32_t i = 0; i < LUT8_SIZE; i++) {
                lut.SetValue(i, (uint16_t)Hif8ToFpBitsV2((uint8_t)i, expBits_, fracBits_, expBias_));
            }
        }
    }

    // ── 数据搬运 ────────────────────────────────────────────────────────────

    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t count)
    {
        uint32_t byteOffset = tileIdx * TILE_LENGTH * inputBytes_;
        uint32_t byteCount  = count * inputBytes_;
        auto xLocal = inQueue_.AllocTensor<uint8_t>();
        AscendC::DataCopyParams    copyParams{1, (uint16_t)byteCount, 0, 0};
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, inputGm_[byteOffset], copyParams, padParams);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t count)
    {
        uint32_t byteOffset = tileIdx * TILE_LENGTH * outputBytes_;
        uint32_t byteCount  = count * outputBytes_;
        auto yLocal = outQueue_.DeQue<uint8_t>();
        AscendC::DataCopyExtParams copyParams{1, byteCount, 0, 0, 0};
        AscendC::DataCopyPad(outputGm_[byteOffset], yLocal, copyParams);
        outQueue_.FreeTensor(yLocal);
    }

    // ── 计算分派 ────────────────────────────────────────────────────────────

    __aicore__ inline void Compute(uint32_t count)
    {
        if (castMode_ == 1 || castMode_ == 2) {
            ComputeEncode16(count);   // BF16/FP16 → HiF8：向量化 LUT Gather
        } else if (castMode_ == 0) {
            ComputeEncodeFp32(count); // FP32 → HiF8：scalar（A2 无 uint32_t 向量位运算）
        } else {
            ComputeDecode(count);     // HiF8 → FP32/FP16/BF16：UB LUT scalar
        }
    }

    // ── BF16/FP16 → HiF8（半空间 UB LUT + 符号分离）────────────────────────
    //
    // LUT 只存正半空间量级（32768 条），利用对称性：
    //   encode(-x) = encode(x) | 0x80  （当 encode(x) ≠ 0 时）
    //   encode(-0) = 0x00              （下溢/零不加符号位）
    // 每元素约 5 条指令（AND + GetValue + SHR + compare + OR），
    // 节省 32KB UB，且 DataCopyPad 从 2 次合并为 1 次。
    __aicore__ inline void ComputeEncode16(uint32_t count)
    {
        auto xLocal = inQueue_.DeQue<uint8_t>();
        auto xU16   = xLocal.template ReinterpretCast<uint16_t>();
        auto yLocal = outQueue_.AllocTensor<uint8_t>();
        auto lut    = lut16Buf_.Get<uint8_t>();

        for (uint32_t i = 0; i < count; i++) {
            uint16_t v       = xU16.GetValue(i);
            uint8_t  sign    = (uint8_t)(v >> 15);
            uint8_t  mag     = lut.GetValue((uint32_t)(v & 0x7FFFu));
            // 下溢/零时 mag==0，不叠加符号位（保持 0x00）
            yLocal.SetValue(i, (mag == 0x00u) ? 0x00u : (mag | (sign << HIF8_SIGN_SHIFT_V2)));
        }

        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    // ── FP32 → HiF8（scalar，A2 平台无法用向量位运算）────────────────────────
    __aicore__ inline void ComputeEncodeFp32(uint32_t count)
    {
        auto xLocal  = inQueue_.DeQue<uint8_t>();
        auto xU32    = xLocal.template ReinterpretCast<uint32_t>();
        auto yLocal  = outQueue_.AllocTensor<uint8_t>();

        for (uint32_t i = 0; i < count; i++) {
            uint32_t val  = xU32.GetValue(i);
            uint32_t sign = val >> 31;
            uint8_t mag   = Fp32MagnitudeToHif8(val & 0x7FFFFFFF);
            yLocal.SetValue(i, mag | (uint8_t)(sign << HIF8_SIGN_SHIFT_V2));
        }

        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    // ── HiF8 → FP32/FP16/BF16（scalar + UB LUT，消除分支）──────────────────
    //
    // InitLut8 已将 256 个 HiF8 值的解码结果预存在 UB（lut8Buf_）中。
    // Compute 阶段只需按 index 读取，无分支、无位运算。
    __aicore__ inline void ComputeDecode(uint32_t count)
    {
        auto xLocal = inQueue_.DeQue<uint8_t>();
        auto yLocal = outQueue_.AllocTensor<uint8_t>();

        if (outputBytes_ == 4) {
            auto lut   = lut8Buf_.Get<uint32_t>();
            auto yU32  = yLocal.template ReinterpretCast<uint32_t>();
            for (uint32_t i = 0; i < count; i++) {
                yU32.SetValue(i, lut.GetValue(xLocal.GetValue(i)));
            }
        } else {
            auto lut   = lut8Buf_.Get<uint16_t>();
            auto yU16  = yLocal.template ReinterpretCast<uint16_t>();
            for (uint32_t i = 0; i < count; i++) {
                yU16.SetValue(i, lut.GetValue(xLocal.GetValue(i)));
            }
        }

        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe*  pipe_;
    const __gm__ HiFloat8CastTilingData* tiling_;

    uint32_t castMode_;
    uint32_t expBits_, fracBits_, expBias_;
    uint32_t inputBytes_, outputBytes_;
    uint32_t total_, tileNum_, tailTile_;

    AscendC::GlobalTensor<uint8_t> inputGm_;
    AscendC::GlobalTensor<uint8_t> outputGm_;

    AscendC::TQue<AscendC::TPosition::VECIN,  DOUBLE_BUFFER> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER> outQueue_;

    // BF16/FP16 encode UB LUT（65536 × 1B = 64KB）
    AscendC::TBuf<AscendC::TPosition::VECCALC> lut16Buf_;
    // decode UB LUT（256 × outputBytes_，最大 1KB）
    AscendC::TBuf<AscendC::TPosition::VECCALC> lut8Buf_;
};
