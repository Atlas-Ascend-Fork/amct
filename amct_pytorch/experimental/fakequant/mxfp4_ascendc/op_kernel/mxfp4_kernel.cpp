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

    __aicore__ inline void Compute(uint32_t nblk) {
        LocalTensor<float> xL = inQueue.DeQue<float>();
        LocalTensor<float> yL = outQueue.AllocTensor<float>();

        LocalTensor<float> buf = calcBuf.Get<float>();
        LocalTensor<float> yAbBuf = buf[0];
        LocalTensor<float> qAbBuf = buf[TILE_ELEMS];
        LocalTensor<float> tmpBuf = buf[2 * TILE_ELEMS];
        LocalTensor<float> wrkL = buf[3 * TILE_ELEMS];

        uint32_t N = nblk * BLOCK_SIZE;
        float invScaleArr[BLOCKS_PER_TILE];

        /* ==== Phase 0: tile-wide |x| ==== */
        Abs(yAbBuf, xL, N);

        /* ==== Phase 1a: BlockReduceMax + tile-wide Max ====
         * BlockReduceMax reduces each 8-float (32-byte) data block to
         * its max in a single API call with repeat/stride.  Two tile-wide
         * Max ops at 8-aligned offsets (16, 8) combine the 4 partial
         * maxes per 32-element block into one final max at position 0.
         * Replaces 192 × ReduceMax(32) with ~5 API calls total. */
        {
            uint32_t totalRpt = nblk * 4;
            uint32_t doneRpt = 0;
            while (doneRpt < totalRpt) {
                int32_t r = (int32_t)(totalRpt - doneRpt);
                if (r > 255)
                    r = 255;
                BlockReduceMax<float, false>(tmpBuf[doneRpt * 8], yAbBuf[doneRpt * 8], r, 8, 1, 1, 1);
                doneRpt += (uint32_t)r;
            }
        }
        Max(tmpBuf, tmpBuf, tmpBuf[16], N - 16);
        Max(tmpBuf, tmpBuf, tmpBuf[8], N - 8);

        /* ==== Phase 1b: read results + compute E8M0 scales ====
         * 8x unrolled for maximum scalar ILP. */
        {
            uint32_t b = 0;
            for (; b + 7 < nblk; b += 8) {
                float m0 = tmpBuf.GetValue((b)*BLOCK_SIZE);
                float m1 = tmpBuf.GetValue((b + 1) * BLOCK_SIZE);
                float m2 = tmpBuf.GetValue((b + 2) * BLOCK_SIZE);
                float m3 = tmpBuf.GetValue((b + 3) * BLOCK_SIZE);
                float m4 = tmpBuf.GetValue((b + 4) * BLOCK_SIZE);
                float m5 = tmpBuf.GetValue((b + 5) * BLOCK_SIZE);
                float m6 = tmpBuf.GetValue((b + 6) * BLOCK_SIZE);
                float m7 = tmpBuf.GetValue((b + 7) * BLOCK_SIZE);

                float r0 = m0 * effInvScale;
                if (r0 < MIN_SCALE_RAW)
                    r0 = MIN_SCALE_RAW;
                float r1 = m1 * effInvScale;
                if (r1 < MIN_SCALE_RAW)
                    r1 = MIN_SCALE_RAW;
                float r2 = m2 * effInvScale;
                if (r2 < MIN_SCALE_RAW)
                    r2 = MIN_SCALE_RAW;
                float r3 = m3 * effInvScale;
                if (r3 < MIN_SCALE_RAW)
                    r3 = MIN_SCALE_RAW;
                float r4 = m4 * effInvScale;
                if (r4 < MIN_SCALE_RAW)
                    r4 = MIN_SCALE_RAW;
                float r5 = m5 * effInvScale;
                if (r5 < MIN_SCALE_RAW)
                    r5 = MIN_SCALE_RAW;
                float r6 = m6 * effInvScale;
                if (r6 < MIN_SCALE_RAW)
                    r6 = MIN_SCALE_RAW;
                float r7 = m7 * effInvScale;
                if (r7 < MIN_SCALE_RAW)
                    r7 = MIN_SCALE_RAW;

                union {
                    float f;
                    int32_t i;
                } rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;
                rb0.f = r0;
                rb1.f = r1;
                rb2.f = r2;
                rb3.f = r3;
                rb4.f = r4;
                rb5.f = r5;
                rb6.f = r6;
                rb7.f = r7;

                int32_t e0 = ((rb0.i >> 23) & 0xFF) + ((rb0.i & 0x7FFFFF) >= 0x3504F3);
                int32_t e1 = ((rb1.i >> 23) & 0xFF) + ((rb1.i & 0x7FFFFF) >= 0x3504F3);
                int32_t e2 = ((rb2.i >> 23) & 0xFF) + ((rb2.i & 0x7FFFFF) >= 0x3504F3);
                int32_t e3 = ((rb3.i >> 23) & 0xFF) + ((rb3.i & 0x7FFFFF) >= 0x3504F3);
                int32_t e4 = ((rb4.i >> 23) & 0xFF) + ((rb4.i & 0x7FFFFF) >= 0x3504F3);
                int32_t e5 = ((rb5.i >> 23) & 0xFF) + ((rb5.i & 0x7FFFFF) >= 0x3504F3);
                int32_t e6 = ((rb6.i >> 23) & 0xFF) + ((rb6.i & 0x7FFFFF) >= 0x3504F3);
                int32_t e7 = ((rb7.i >> 23) & 0xFF) + ((rb7.i & 0x7FFFFF) >= 0x3504F3);

                union {
                    float f;
                    int32_t i;
                } is0, is1, is2, is3, is4, is5, is6, is7;
                is0.i = (254 - e0) << 23;
                is1.i = (254 - e1) << 23;
                is2.i = (254 - e2) << 23;
                is3.i = (254 - e3) << 23;
                is4.i = (254 - e4) << 23;
                is5.i = (254 - e5) << 23;
                is6.i = (254 - e6) << 23;
                is7.i = (254 - e7) << 23;

                invScaleArr[b] = is0.f;
                invScaleArr[b + 1] = is1.f;
                invScaleArr[b + 2] = is2.f;
                invScaleArr[b + 3] = is3.f;
                invScaleArr[b + 4] = is4.f;
                invScaleArr[b + 5] = is5.f;
                invScaleArr[b + 6] = is6.f;
                invScaleArr[b + 7] = is7.f;
            }
            for (; b < nblk; b++) {
                float maxAbs = tmpBuf.GetValue(b * BLOCK_SIZE);
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
                invScaleArr[b] = isb.f;
            }
        }

        /* ==== Phase 1c: build invScale tile (tmpBuf) + derive scale*0.5 tile (yL) ====
         * 8x unrolled Duplicate loop for maximum dispatch throughput.
         * Derive scale*0.5 from invScale via int32 subtraction. */
        {
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
        }
        {
            LocalTensor<int32_t> intTmp = tmpBuf.ReinterpretCast<int32_t>();
            LocalTensor<int32_t> intYL = yL.ReinterpretCast<int32_t>();
            Duplicate(intYL, (int32_t)0x7E800000, N);
            Sub(intYL, intYL, intTmp, N);
        }
        Mul(yAbBuf, yAbBuf, tmpBuf, N);

        /* ==== Phase 2: tile-wide FP4 bucketing in q2 space ====
         *
         * Uniform [0,2]: q2 = min(floor(2*y+0.5), 4)
         * floor via Cast(float→int32, TRUNC) for non-negative values.
         *
         * Non-uniform: 3 step functions with q2 deltas {2,2,4}.
         * Total: 5 + 15 = 20 tile-wide ops (was 25). */

        /* --- uniform [0,2] in q2 space: 5 ops --- */
        Muls(qAbBuf, yAbBuf, 2.0f, N);
        Adds(qAbBuf, qAbBuf, 0.5f, N);
        LocalTensor<int32_t> intBuf = tmpBuf.ReinterpretCast<int32_t>();
        Cast(intBuf, qAbBuf, RoundMode::CAST_TRUNC, N);
        Cast(qAbBuf, intBuf, RoundMode::CAST_NONE, N);
        Mins(qAbBuf, qAbBuf, 4.0f, N);

        /* --- step functions for {3,4,6} in q2 space: 15 ops --- */
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

        /* ==== Phase 3: apply scale*0.5 from yL ==== */
        Mul(qAbBuf, qAbBuf, yL, N);

        /* ==== Phase 4: sign + final output ==== */
        Muls(yAbBuf, xL, BIG, N);
        Mins(yAbBuf, yAbBuf, 1.0f, N);
        Maxs(yAbBuf, yAbBuf, -1.0f, N);
        Mul(yL, qAbBuf, yAbBuf, N);

        outQueue.EnQue(yL);
        inQueue.FreeTensor(xL);
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
