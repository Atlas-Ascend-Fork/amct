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
 *  castMode          | LUT 每元素指令数 | 优化手段
 *  ------------------|----------------|--------------------------------
 *  FP16_TO_HIF8      | ~5 条          | 32768-entry 半空间 UB LUT + 符号分离
 *  BF16_TO_HIF8      | ~5 条          | 32768-entry 半空间 UB LUT + 符号分离
 *  HIF8_TO_FP16      | 1 条           | 256-entry UB LUT 消除分支
 *  HIF8_TO_BF16      | 1 条           | 256-entry UB LUT 消除分支
 *
 * UB 占用（tileLength 由 host 运行时根据平台实际可用 UB 大小计算；
 *          maxTile ≥ 32768 时对齐到 32768，否则对齐到 32，上限 65536）：
 *   BF16/FP16 encode:  32KB(LUT16) + 1T×2B(inQ) + 1T×1B(outQ)  = 32768 + 3T bytes
 *   decode FP16/BF16:   1KB(LUT8)  + 1T×1B(inQ) + 1T×2B(outQ)  =   512 + 3T bytes
 *
 *   单缓冲：compute 为标量循环，与 MTE 流水重叠收益为 0，
 *   故 TQue depth=1 + InitBuffer num=1，释放队列另一半 UB 给 tile。
 *
 * CopyIn DataCopy 拆分策略（DataCopyParams.blockLen 为 uint16_t，max 65535）：
 *   byteCount ≤ 65535           → {1, byteCount}
 *   byteCount % 32768 == 0      → {byteCount/32768, 32768}   (全 tile，tileLength 对齐保证整除)
 *   byteCount 为偶数（encode tail）→ {2, byteCount/2}         (count ≤ 65535，blockLen ≤ 65535)
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

constexpr uint32_t HIF8_KERNEL_SIGN_SHIFT = 7;

class KernelHiFloat8CastLut {
public:
    __aicore__ inline KernelHiFloat8CastLut(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, const __gm__ HiFloat8CastTilingData *tiling, GM_ADDR lutGm) {
        uint32_t castMode = tiling->castMode;
        uint32_t blockIdx = AscendC::GetBlockIdx();

        isEncode_ = (castMode <= BF16_TO_HIF8);
        inputBytes_ = isEncode_ ? 2u : 1u;
        outputBytes_ = isEncode_ ? 1u : 2u;
        tileLength_ = tiling->tileLength;

        total_ = (blockIdx < tiling->blockNum - 1) ? tiling->numPerCore : tiling->tailNumLastCore;
        tileNum_ = (total_ + tileLength_ - 1) / tileLength_;
        tailTile_ = static_cast<uint32_t>(total_ - static_cast<uint64_t>(tileLength_) * (tileNum_ - 1));
        // tailTile_ > 0 guaranteed: total_ > 0, ceiling division ensures last tile is non-empty

        uint64_t offset = static_cast<uint64_t>(blockIdx) * tiling->numPerCore;
        inputGm_.SetGlobalBuffer((__gm__ uint8_t *)input + offset * inputBytes_, total_ * inputBytes_);
        outputGm_.SetGlobalBuffer((__gm__ uint8_t *)output + offset * outputBytes_, total_ * outputBytes_);
        pipe_->InitBuffer(inQueue_, 1, tileLength_ * inputBytes_);
        pipe_->InitBuffer(outQueue_, 1, tileLength_ * outputBytes_);

        if (isEncode_) {
            pipe_->InitBuffer(lut16Buf_, LUT16_SIZE);
            AscendC::GlobalTensor<uint8_t> lutGmTensor;
            lutGmTensor.SetGlobalBuffer((__gm__ uint8_t *)lutGm, LUT16_SIZE);
            auto lutUb = lut16Buf_.Get<uint8_t>();
            AscendC::DataCopyParams cp{1, static_cast<uint16_t>(LUT16_SIZE), 0, 0};
            AscendC::DataCopyPadParams pp{false, 0, 0, 0};
            AscendC::DataCopyPad(lutUb, lutGmTensor[0], cp, pp);
        } else {
            pipe_->InitBuffer(lut8Buf_, LUT8_SIZE * 2u);
            AscendC::GlobalTensor<uint16_t> lutGmTensor;
            lutGmTensor.SetGlobalBuffer((__gm__ uint16_t *)lutGm, LUT8_SIZE);
            auto lutUb = lut8Buf_.Get<uint16_t>();
            AscendC::DataCopyParams cp{1, static_cast<uint16_t>(LUT8_SIZE * 2u), 0, 0};
            AscendC::DataCopyPadParams pp{false, 0, 0, 0};
            AscendC::DataCopyPad(lutUb, lutGmTensor[0], cp, pp);
        }
    }

    __aicore__ inline void Process() {
        for (uint64_t i = 0; i < tileNum_; i++) {
            uint32_t count = (i == tileNum_ - 1) ? tailTile_ : tileLength_;
            CopyIn(i, count);
            Compute(count);
            CopyOut(i, count);
        }
    }

private:
    // ── 数据搬运 ────────────────────────────────────────────────────────────

    __aicore__ inline void CopyIn(uint64_t tileIdx, uint32_t count) {
        uint64_t byteOffset = tileIdx * static_cast<uint64_t>(tileLength_) * inputBytes_;
        uint32_t byteCount = count * inputBytes_;
        auto xLocal = inQueue_.AllocTensor<uint8_t>();
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        // DataCopyParams.blockLen is uint16_t (max 65535); split large byteCount:
        //
        //  Case A — byteCount ≤ 65535 : {1, byteCount}
        //           covers all decode tiles and small encode tails
        //
        //  Case B — byteCount % 32768 == 0 : {byteCount/32768, 32768}
        //           full tiles when tileLength ∈ {32768, 65536} (tileLength aligned to 32768)
        //           e.g. encode tileLength=65536 → byteCount=131072 → {4, 32768}
        //               decode tileLength=65536 → byteCount=65536  → {2, 32768}
        //
        //  Case C — byteCount even, not case B : {2, byteCount/2}
        //           encode tail tiles; byteCount=count*2, count ≤ 65535 → byteCount/2 ≤ 65535
        AscendC::DataCopyParams cp;
        if (byteCount <= 65535u) {
            cp = {1, static_cast<uint16_t>(byteCount), 0, 0};
        } else if (byteCount % 32768u == 0u) {
            cp = {static_cast<uint16_t>(byteCount / 32768u), 32768u, 0, 0};
        } else {
            cp = {2u, static_cast<uint16_t>(byteCount >> 1), 0, 0};
        }
        AscendC::DataCopyPad(xLocal, inputGm_[byteOffset], cp, padParams);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t tileIdx, uint32_t count) {
        uint64_t byteOffset = tileIdx * static_cast<uint64_t>(tileLength_) * outputBytes_;
        uint32_t byteCount = count * outputBytes_;
        auto yLocal = outQueue_.DeQue<uint8_t>();
        AscendC::DataCopyExtParams copyParams{1, byteCount, 0, 0, 0};
        AscendC::DataCopyPad(outputGm_[byteOffset], yLocal, copyParams);
        outQueue_.FreeTensor(yLocal);
    }

    // ── 计算分派 ────────────────────────────────────────────────────────────

    __aicore__ inline void Compute(uint32_t count) {
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
    __aicore__ inline void ComputeEncode16(uint32_t count) {
        auto xLocal = inQueue_.DeQue<uint8_t>();
        auto xU16 = xLocal.template ReinterpretCast<uint16_t>();
        auto yLocal = outQueue_.AllocTensor<uint8_t>();
        auto lut = lut16Buf_.Get<uint8_t>();

        for (uint32_t i = 0; i < count; i++) {
            uint16_t v = xU16.GetValue(i);
            uint8_t sign = static_cast<uint8_t>(v >> 15);
            uint8_t mag = lut.GetValue(static_cast<uint32_t>(v & 0x7FFFu));
            // 下溢/零时 mag==0，不叠加符号位（保持 0x00）
            yLocal.SetValue(i, (mag == 0x00u) ? 0x00u : (mag | (sign << HIF8_KERNEL_SIGN_SHIFT)));
        }

        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    // ── HiF8 → FP16/BF16（UB LUT 直查，无分支）────────────────────────────
    __aicore__ inline void ComputeDecode(uint32_t count) {
        auto xLocal = inQueue_.DeQue<uint8_t>();
        auto yLocal = outQueue_.AllocTensor<uint8_t>();
        auto lut = lut8Buf_.Get<uint16_t>();
        auto yU16 = yLocal.template ReinterpretCast<uint16_t>();
        for (uint32_t i = 0; i < count; i++) {
            yU16.SetValue(i, lut.GetValue(xLocal.GetValue(i)));
        }
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe *pipe_;

    bool isEncode_ = false;
    uint32_t inputBytes_ = 0;
    uint32_t outputBytes_ = 0;
    uint32_t tileLength_ = 0;
    uint64_t total_ = 0;
    uint64_t tileNum_ = 0;
    uint32_t tailTile_ = 0;

    AscendC::GlobalTensor<uint8_t> inputGm_;
    AscendC::GlobalTensor<uint8_t> outputGm_;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;

    // BF16/FP16 encode UB LUT（32768 × 1B = 32KB，半空间）
    AscendC::TBuf<AscendC::TPosition::VECCALC> lut16Buf_;
    // decode UB LUT（256 × 2B = 512B，FP16/BF16 输出）
    AscendC::TBuf<AscendC::TPosition::VECCALC> lut8Buf_;
};

extern "C" __global__ __vector__ void hifloat8_cast_kernel_lut(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling, GM_ADDR lutGm) {
    AscendC::TPipe pipe;
    KernelHiFloat8CastLut op(&pipe);
    op.Init(input, output, (__gm__ HiFloat8CastTilingData *)tiling, lutGm);
    op.Process();
}
