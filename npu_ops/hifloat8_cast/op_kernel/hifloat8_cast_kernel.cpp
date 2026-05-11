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
 *  castMode | LUT 每元素指令数 | 优化手段
 *  ---------|----------------|--------------------------------
 *  0 FP16→  | ~5 条          | 32768-entry 半空间 UB LUT + 符号分离
 *  1 BF16→  | ~5 条          | 32768-entry 半空间 UB LUT + 符号分离
 *  2 →FP16  | 1 条           | 256-entry UB LUT 消除分支
 *  3 →BF16  | 1 条           | 256-entry UB LUT 消除分支
 *
 * UB 占用（TILE_LENGTH=8192，队列按实际元素宽度分配）：
 *   BF16/FP16 encode:  32KB(LUT16) + 32KB(inQ×2 @2B) + 16KB(outQ×2 @1B) = 80KB  < 256KB
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

#include "kernel_operator.h"
#include "hifloat8_cast_tiling.h"

constexpr uint32_t FP16_EXP_BITS = 5;
constexpr uint32_t FP16_FRAC_BITS = 10;
constexpr uint32_t FP16_EXP_BIAS = 15;

constexpr uint32_t BF16_EXP_BITS = 8;
constexpr uint32_t BF16_FRAC_BITS = 7;
constexpr uint32_t BF16_EXP_BIAS = 127;

constexpr uint32_t HIF8_SIGN_SHIFT = 7;
constexpr uint32_t HIF8_DENORMAL_EXP_BIAS = 23;

// 32768-entry LUT for BF16/FP16 encode (positive half: indexed by absBits = val & 0x7FFF)
// Sign bit is stripped before lookup and re-applied after, halving UB from 64KB→32KB.
constexpr uint32_t LUT16_SIZE = 32768;
// 256-entry LUT for decode (indexed by raw HiF8 uint8_t value)
constexpr uint32_t LUT8_SIZE = 256;

constexpr uint32_t HIF8_DOT_TABLE[16] = {
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

// HiFloat8 → FP16/BF16 bits（scalar，仅 InitLut8 使用；输出恒为 16-bit）
__aicore__ inline uint32_t Hif8ToFpBits(uint8_t inData,
                                         uint32_t expBits,
                                         uint32_t fracBits,
                                         uint32_t expBias)
{
    if (inData == 0) return 0;

    uint32_t expMax = (1u << expBits) - 1;

    if (inData == 0x80) return (expMax << fracBits) | ((1u << fracBits) - 1);
    if (inData == 0x6F) return expMax << fracBits;
    if (inData == 0xEF) return (1u << 15) | (expMax << fracBits);  // sign bit at bit 15

    uint8_t dot           = (inData & 0x78) >> 3;
    uint32_t info         = HIF8_DOT_TABLE[dot];

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
        expFpN = expBias - (HIF8_DENORMAL_EXP_BIAS - fracFpN);
        fracFpN = 0;
    } else if (expSignShift == 1) {
        expFpN = expBias;
    } else {
        int32_t expNoBias = (int32_t)((expHiF8 | (1u << (expHiF8Bits - 1))) * (signExp == 1 ? -1 : 1));
        expFpN = (uint32_t)(expNoBias + (int32_t)expBias);
    }

    uint32_t sign = inData >> HIF8_SIGN_SHIFT;
    return (sign << 15) | (expFpN << fracBits) | (fracFpN << (fracBits - fracHiF8Bits));
}

class KernelHiFloat8CastLut {
public:
    __aicore__ inline KernelHiFloat8CastLut(AscendC::TPipe* pipe) : pipe_(pipe) {}

    // lut_gm: FP16/BF16 encode 时传预计算 32KB GM 指针；decode 时传 0
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                 const __gm__ HiFloat8CastTilingData* tiling,
                                 GM_ADDR lut_gm)
    {
        uint32_t castMode = tiling->castMode;
        uint32_t blockIdx = AscendC::GetBlockIdx();

        isEncode_    = (castMode <= 1);
        inputBytes_  = isEncode_ ? 2u : 1u;
        outputBytes_ = isEncode_ ? 1u : 2u;

        total_    = (blockIdx < tiling->blockNum - 1)
                    ? tiling->numPerCore : tiling->tailNumLastCore;
        tileNum_  = (total_ + TILE_LENGTH - 1) / TILE_LENGTH;
        tailTile_ = total_ - TILE_LENGTH * (tileNum_ - 1);
        if (tailTile_ == 0 && tileNum_ > 0) { tileNum_--; tailTile_ = TILE_LENGTH; }

        uint64_t offset = (uint64_t)blockIdx * tiling->numPerCore;
        inputGm_.SetGlobalBuffer((__gm__ uint8_t*)input  + offset * inputBytes_,
                                  total_ * inputBytes_);
        outputGm_.SetGlobalBuffer((__gm__ uint8_t*)output + offset * outputBytes_,
                                   total_ * outputBytes_);
        pipe_->InitBuffer(inQueue_,  DOUBLE_BUFFER, TILE_LENGTH * inputBytes_);
        pipe_->InitBuffer(outQueue_, DOUBLE_BUFFER, TILE_LENGTH * outputBytes_);

        if (isEncode_) {
            // BF16/FP16 encode: 从 GM 复制预计算半空间 LUT 到 UB
            // LUT16_SIZE=32768 < 65535，单次 DataCopyPad 即可完成
            pipe_->InitBuffer(lut16Buf_, LUT16_SIZE);
            AscendC::GlobalTensor<uint8_t> lutGm;
            lutGm.SetGlobalBuffer((__gm__ uint8_t*)lut_gm, LUT16_SIZE);
            auto lutUb = lut16Buf_.Get<uint8_t>();
            AscendC::DataCopyParams    cp{1, (uint16_t)LUT16_SIZE, 0, 0};
            AscendC::DataCopyPadParams pp{false, 0, 0, 0};
            AscendC::DataCopyPad(lutUb, lutGm[0], cp, pp);
        } else {
            // decode: LUT 只有 256 条，在 UB 内直接构建，开销可忽略
            bool isFP16       = (castMode == 2);
            uint32_t expBits  = isFP16 ? FP16_EXP_BITS  : BF16_EXP_BITS;
            uint32_t fracBits = isFP16 ? FP16_FRAC_BITS : BF16_FRAC_BITS;
            uint32_t expBias  = isFP16 ? FP16_EXP_BIAS  : BF16_EXP_BIAS;
            pipe_->InitBuffer(lut8Buf_, LUT8_SIZE * 2u);
            auto lut = lut8Buf_.Get<uint16_t>();
            for (uint32_t i = 0; i < LUT8_SIZE; i++) {
                lut.SetValue(i, (uint16_t)Hif8ToFpBits((uint8_t)i, expBits, fracBits, expBias));
            }
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
        if (isEncode_) {
            ComputeEncode16(count);
        } else {
            ComputeDecode(count);
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
            uint16_t v    = xU16.GetValue(i);
            uint8_t  sign = (uint8_t)(v >> 15);
            uint8_t  mag  = lut.GetValue((uint32_t)(v & 0x7FFFu));
            // 下溢/零时 mag==0，不叠加符号位（保持 0x00）
            yLocal.SetValue(i, (mag == 0x00u) ? 0x00u : (mag | (sign << HIF8_SIGN_SHIFT)));
        }

        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    // ── HiF8 → FP16/BF16（UB LUT 直查，无分支）────────────────────────────
    __aicore__ inline void ComputeDecode(uint32_t count)
    {
        auto xLocal = inQueue_.DeQue<uint8_t>();
        auto yLocal = outQueue_.AllocTensor<uint8_t>();
        auto lut    = lut8Buf_.Get<uint16_t>();
        auto yU16   = yLocal.template ReinterpretCast<uint16_t>();
        for (uint32_t i = 0; i < count; i++) {
            yU16.SetValue(i, lut.GetValue(xLocal.GetValue(i)));
        }
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe* pipe_;

    bool     isEncode_;
    uint32_t inputBytes_, outputBytes_;
    uint32_t total_, tileNum_, tailTile_;

    AscendC::GlobalTensor<uint8_t> inputGm_;
    AscendC::GlobalTensor<uint8_t> outputGm_;

    AscendC::TQue<AscendC::TPosition::VECIN,  DOUBLE_BUFFER> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER> outQueue_;

    // BF16/FP16 encode UB LUT（32768 × 1B = 32KB，半空间）
    AscendC::TBuf<AscendC::TPosition::VECCALC> lut16Buf_;
    // decode UB LUT（256 × 2B = 512B，FP16/BF16 输出）
    AscendC::TBuf<AscendC::TPosition::VECCALC> lut8Buf_;
};

extern "C" __global__ __vector__ void hifloat8_cast_kernel_lut(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling, GM_ADDR lut_gm)
{
    AscendC::TPipe pipe;
    KernelHiFloat8CastLut op(&pipe);
    op.Init(input, output, (__gm__ HiFloat8CastTilingData*)tiling, lut_gm);
    op.Process();
}
