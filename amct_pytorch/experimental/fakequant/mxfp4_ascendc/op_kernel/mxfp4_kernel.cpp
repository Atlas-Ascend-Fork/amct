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

/**
 * MXFP4 Quant-Dequant Ascend-C Kernel — Reciprocal + Cast floor
 *
 * Per-block FP4 E2M1 quantization with E8M0 power-of-two scales.
 * block_size=32, scale_factor=6.0.
 *
 * Key optimisations over previous version:
 *   1. Phase 1c: derive scale*0.5 tile from invScale tile via
 *      Reciprocal + Muls(0.5), eliminating 192 per-block Duplicate calls.
 *   2. Phase 2 uniform bucketing: use Cast(float→int32, TRUNC) +
 *      Cast(int32→float) for floor, reducing from 10 to 5 tile-wide ops.
 */
#include "kernel_operator.h"
#include "mxfp4_tiling.h"
using namespace AscendC;

constexpr int32_t BLOCK_SIZE = MXFP4_BLOCK_SIZE;
constexpr float INV_SCALE_FACTOR = MXFP4_INV_SCALE_FACTOR;
constexpr float MIN_SCALE_RAW = MXFP4_MIN_SCALE_RAW;
constexpr int32_t BLOCKS_PER_TILE = MXFP4_BLOCKS_PER_TILE;
constexpr int32_t TILE_ELEMS = MXFP4_TILE_ELEMS;
constexpr float BIG = 1e20f;
constexpr float STEP_EPS = 1e-6f;

class KernelMxfp4 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLen, uint32_t numCores, float invScaleMul) {
        // Runtime multiplier applied on top of the compile-time
        // INV_SCALE_FACTOR (1/6.0). Guards against an uninitialised / zero
        // tiling slot so an old cache entry cannot silently zero the scale.
        if (!(invScaleMul > 0.0f))
            invScaleMul = 1.0f;
        effInvScale = INV_SCALE_FACTOR * invScaleMul;

        // Defense in depth: host and entry already clamp numCores>=1, but
        // Init must not divide by zero if a bad tiling buffer reaches here.
        if (numCores == 0)
            numCores = 1;

        uint32_t coreIdx = GetBlockIdx();
        uint32_t nBlocks = totalLen / BLOCK_SIZE;
        uint32_t perCore = nBlocks / numCores;
        uint32_t leftover = nBlocks % numCores;

        if (coreIdx < leftover) {
            numBlocks = perCore + 1;
            startBlock = coreIdx * numBlocks;
        } else {
            numBlocks = perCore;
            startBlock = coreIdx * perCore + leftover;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, totalLen);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLen);

        pipe.InitBuffer(inQueue, 2, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(outQueue, 2, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(calcBuf, (3 * TILE_ELEMS + BLOCK_SIZE) * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (numBlocks == 0)
            return;

        uint32_t off = 0;
        uint32_t bpt = numBlocks;
        if (bpt > (uint32_t)BLOCKS_PER_TILE)
            bpt = (uint32_t)BLOCKS_PER_TILE;

        CopyIn(off, bpt);
        uint32_t nextOff = off + bpt;

        while (nextOff < numBlocks) {
            uint32_t nextBpt = numBlocks - nextOff;
            if (nextBpt > (uint32_t)BLOCKS_PER_TILE)
                nextBpt = (uint32_t)BLOCKS_PER_TILE;

            CopyIn(nextOff, nextBpt);
            Compute(bpt);
            CopyOut(off, bpt);

            off = nextOff;
            bpt = nextBpt;
            nextOff = off + bpt;
        }

        Compute(bpt);
        CopyOut(off, bpt);
    }

private:
    __aicore__ inline void CopyIn(uint32_t blockOff, uint32_t nblk) {
        LocalTensor<float> xL = inQueue.AllocTensor<float>();
        DataCopy(xL, xGm[(startBlock + blockOff) * BLOCK_SIZE], nblk * BLOCK_SIZE);
        inQueue.EnQue(xL);
    }

    __aicore__ inline void CopyOut(uint32_t blockOff, uint32_t nblk) {
        LocalTensor<float> yL = outQueue.DeQue<float>();
        DataCopy(yGm[(startBlock + blockOff) * BLOCK_SIZE], yL, nblk * BLOCK_SIZE);
        outQueue.FreeTensor(yL);
    }

    // E8M0 inv-scale from one block maxAbs (keeps Phase 1b under line limits).
    __aicore__ inline float MaxAbsToInvScale(float maxAbs) {
        float raw = maxAbs * effInvScale;
        if (raw < MIN_SCALE_RAW)
            raw = MIN_SCALE_RAW;
        union {
            float f;
            int32_t i;
        } rb;
        rb.f = raw;
        int32_t e = ((rb.i >> 23) & 0xFF) + ((rb.i & 0x7FFFFF) >= 0x3504F3);
        union {
            float f;
            int32_t i;
        } isb;
        isb.i = (254 - e) << 23;
        return isb.f;
    }

    // Phase 0 + 1a: |x| then BlockReduceMax + tile Max → per-block max at stride 32.
    __aicore__ inline void ComputeAbsAndBlockMax(const LocalTensor<float> &xL, LocalTensor<float> &yAbBuf,
        LocalTensor<float> &tmpBuf, uint32_t nblk, uint32_t N) {
        Abs(yAbBuf, xL, N);
        uint32_t totalRpt = nblk * 4;
        uint32_t doneRpt = 0;
        while (doneRpt < totalRpt) {
            int32_t r = (int32_t)(totalRpt - doneRpt);
            if (r > 255)
                r = 255;
            BlockReduceMax<float, false>(tmpBuf[doneRpt * 8], yAbBuf[doneRpt * 8], r, 8, 1, 1, 1);
            doneRpt += (uint32_t)r;
        }
        Max(tmpBuf, tmpBuf, tmpBuf[16], N - 16);
        Max(tmpBuf, tmpBuf, tmpBuf[8], N - 8);
    }

    // Phase 1b: read block maxes and fill invScaleArr (8x unrolled).
    __aicore__ inline void FillInvScaleArr(const LocalTensor<float> &tmpBuf, float *invScaleArr, uint32_t nblk) {
        uint32_t b = 0;
        for (; b + 7 < nblk; b += 8) {
            invScaleArr[b] = MaxAbsToInvScale(tmpBuf.GetValue((b)*BLOCK_SIZE));
            invScaleArr[b + 1] = MaxAbsToInvScale(tmpBuf.GetValue((b + 1) * BLOCK_SIZE));
            invScaleArr[b + 2] = MaxAbsToInvScale(tmpBuf.GetValue((b + 2) * BLOCK_SIZE));
            invScaleArr[b + 3] = MaxAbsToInvScale(tmpBuf.GetValue((b + 3) * BLOCK_SIZE));
            invScaleArr[b + 4] = MaxAbsToInvScale(tmpBuf.GetValue((b + 4) * BLOCK_SIZE));
            invScaleArr[b + 5] = MaxAbsToInvScale(tmpBuf.GetValue((b + 5) * BLOCK_SIZE));
            invScaleArr[b + 6] = MaxAbsToInvScale(tmpBuf.GetValue((b + 6) * BLOCK_SIZE));
            invScaleArr[b + 7] = MaxAbsToInvScale(tmpBuf.GetValue((b + 7) * BLOCK_SIZE));
        }
        for (; b < nblk; b++) {
            invScaleArr[b] = MaxAbsToInvScale(tmpBuf.GetValue(b * BLOCK_SIZE));
        }
    }

    // Phase 1c: Duplicate invScale into tmpBuf; put scale*0.5 into yL; yAbs *= invScale.
    __aicore__ inline void BroadcastInvScaleAndHalf(LocalTensor<float> &yAbBuf, LocalTensor<float> &tmpBuf,
        LocalTensor<float> &yL, const float *invScaleArr, uint32_t nblk, uint32_t N) {
        uint32_t b = 0;
        for (; b + 7 < nblk; b += 8) {
            Duplicate(tmpBuf[(b)*BLOCK_SIZE], invScaleArr[b], BLOCK_SIZE);
            Duplicate(tmpBuf[(b + 1) * BLOCK_SIZE], invScaleArr[b + 1], BLOCK_SIZE);
            Duplicate(tmpBuf[(b + 2) * BLOCK_SIZE], invScaleArr[b + 2], BLOCK_SIZE);
            Duplicate(tmpBuf[(b + 3) * BLOCK_SIZE], invScaleArr[b + 3], BLOCK_SIZE);
            Duplicate(tmpBuf[(b + 4) * BLOCK_SIZE], invScaleArr[b + 4], BLOCK_SIZE);
            Duplicate(tmpBuf[(b + 5) * BLOCK_SIZE], invScaleArr[b + 5], BLOCK_SIZE);
            Duplicate(tmpBuf[(b + 6) * BLOCK_SIZE], invScaleArr[b + 6], BLOCK_SIZE);
            Duplicate(tmpBuf[(b + 7) * BLOCK_SIZE], invScaleArr[b + 7], BLOCK_SIZE);
        }
        for (; b < nblk; b++) {
            Duplicate(tmpBuf[b * BLOCK_SIZE], invScaleArr[b], BLOCK_SIZE);
        }
        LocalTensor<int32_t> intTmp = tmpBuf.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> intYL = yL.ReinterpretCast<int32_t>();
        Duplicate(intYL, (int32_t)0x7E800000, N);
        Sub(intYL, intYL, intTmp, N);
        Mul(yAbBuf, yAbBuf, tmpBuf, N);
    }

    // Phase 2: FP4 e2m1 bucketing in q2 space (uniform + three step functions).
    __aicore__ inline void BucketFp4(
        LocalTensor<float> &yAbBuf, LocalTensor<float> &qAbBuf, LocalTensor<float> &tmpBuf, uint32_t N) {
        Muls(qAbBuf, yAbBuf, 2.0f, N);
        Adds(qAbBuf, qAbBuf, 0.5f, N);
        LocalTensor<int32_t> intBuf = tmpBuf.ReinterpretCast<int32_t>();
        Cast(intBuf, qAbBuf, RoundMode::CAST_TRUNC, N);
        Cast(qAbBuf, intBuf, RoundMode::CAST_NONE, N);
        Mins(qAbBuf, qAbBuf, 4.0f, N);

        Adds(tmpBuf, yAbBuf, -(2.5f - STEP_EPS), N);
        Maxs(tmpBuf, tmpBuf, 0.0f, N);
        Muls(tmpBuf, tmpBuf, BIG, N);
        Mins(tmpBuf, tmpBuf, 2.0f, N);
        Add(qAbBuf, qAbBuf, tmpBuf, N);

        Adds(tmpBuf, yAbBuf, -(3.5f - STEP_EPS), N);
        Maxs(tmpBuf, tmpBuf, 0.0f, N);
        Muls(tmpBuf, tmpBuf, BIG, N);
        Mins(tmpBuf, tmpBuf, 2.0f, N);
        Add(qAbBuf, qAbBuf, tmpBuf, N);

        Adds(tmpBuf, yAbBuf, -(5.0f - STEP_EPS), N);
        Maxs(tmpBuf, tmpBuf, 0.0f, N);
        Muls(tmpBuf, tmpBuf, BIG, N);
        Mins(tmpBuf, tmpBuf, 4.0f, N);
        Add(qAbBuf, qAbBuf, tmpBuf, N);
    }

    // Phase 3 + 4: apply scale*0.5, restore sign, enqueue output.
    __aicore__ inline void ApplyScaleSignAndFinish(LocalTensor<float> &xL, LocalTensor<float> &yL,
        LocalTensor<float> &yAbBuf, LocalTensor<float> &qAbBuf, uint32_t N) {
        Mul(qAbBuf, qAbBuf, yL, N);
        Muls(yAbBuf, xL, BIG, N);
        Mins(yAbBuf, yAbBuf, 1.0f, N);
        Maxs(yAbBuf, yAbBuf, -1.0f, N);
        Mul(yL, qAbBuf, yAbBuf, N);
        outQueue.EnQue(yL);
        inQueue.FreeTensor(xL);
    }

    __aicore__ inline void Compute(uint32_t nblk) {
        LocalTensor<float> xL = inQueue.DeQue<float>();
        LocalTensor<float> yL = outQueue.AllocTensor<float>();

        LocalTensor<float> buf = calcBuf.Get<float>();
        LocalTensor<float> yAbBuf = buf[0];
        LocalTensor<float> qAbBuf = buf[TILE_ELEMS];
        LocalTensor<float> tmpBuf = buf[2 * TILE_ELEMS];

        uint32_t N = nblk * BLOCK_SIZE;
        float invScaleArr[BLOCKS_PER_TILE];

        ComputeAbsAndBlockMax(xL, yAbBuf, tmpBuf, nblk, N);
        FillInvScaleArr(tmpBuf, invScaleArr, nblk);
        BroadcastInvScaleAndHalf(yAbBuf, tmpBuf, yL, invScaleArr, nblk, N);
        BucketFp4(yAbBuf, qAbBuf, tmpBuf, N);
        ApplyScaleSignAndFinish(xL, yL, yAbBuf, qAbBuf, N);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<QuePosition::VECCALC> calcBuf;
    GlobalTensor<float> xGm, yGm;
    uint32_t startBlock, numBlocks;
    float effInvScale;
};

extern "C" __global__ __aicore__ void mxfp4_quant_dequant(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GlobalTensor<int32_t> tilingGM;
    tilingGM.SetGlobalBuffer((__gm__ int32_t *)tiling, MXFP4_TILING_INTS);
    uint32_t totalLen = (uint32_t)tilingGM.GetValue(0);
    uint32_t numCores = (uint32_t)tilingGM.GetValue(1);
    if (numCores == 0)
        numCores = 1;

    // Mxfp4TilingData.invScaleMulBits carries the float bit-pattern of the
    // runtime scale multiplier (see op_kernel/mxfp4_tiling.h).
    union {
        int32_t i;
        float f;
    } scaleU;
    scaleU.i = tilingGM.GetValue(2);
    float invScaleMul = scaleU.f;

    KernelMxfp4 op;
    op.Init(x, y, totalLen, numCores, invScaleMul);
    op.Process();
}
