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
 * HiFloat4 (hifx4) FP -> HiF4 -> FP fake-quant kernel, hand-written AscendC.
 *
 * Per 64-element block: scale_factor = round_E6M2(bf16(max/7)) clamped to [2^-48, 49152];
 * per-8 / per-4 micro exponents in {1, 2}; S1P2 in-group codes with round-half-up.
 * BATCH = 512 per iteration; the host pads N to a multiple of 512, so the in-kernel tail
 * path is dead code kept only as a safety net. Non-finite handling is bit-identical to
 * the HiFloat4 numpy golden: NaN/+-Inf elements are masked to 0 for the block max, and a
 * block containing any non-finite element dequantizes to NaN as a whole.
 *
 * The compute is split into small stage methods (LoadInput / PrepareSignAbs /
 * HandleNonFinite / ReduceBlockMax / PoisonBlock / RoundScaleE6M2 / ComputeExp1 /
 * Exp2PartMax / Exp2Select / ComputeMantissa / Reconstruct / StoreOutput) driven by
 * Compute(). The host launchers (run_hifx_kernel / run_hifx_kernel_bf16) are `extern "C"`.
 * Kernel body guarded by __DAV_C220_VEC__ (Ascend A2/A3, dav-2201); validate on the NPU.
 */
#include "tensorutils.h"

struct Hifxv14ShapeInfo {
    int M, N, N_ceil;
    int mn0, mn1;
    int MB;
};

template <typename T, int E6MX>
class Hifv14Kernel {
public:
    aifunc Hifv14Kernel() {}
    aifunc void Init(GM_ADDR xmtx_, GM_ADDR out_, int M_, int N_, int MB_) {
        shape.M = M_;
        shape.N = N_;
        shape.MB = MB_;
        tiling();

        // assign global tensors
        xmtx = Tensor<T, PGM>(xmtx_);
        out = Tensor<T, PGM>(out_);

        // assign in/out buffer
        xbuf = DBuff<T, PUB>(BATCH);
        outbuf = DBuff<T, PUB>(BATCH);

        //
        absbuf = Tensor<float, PUB>(BATCH);
        signbuf = Tensor<float, PUB>(BATCH);
        exp0buf = Tensor<float, PUB>(BATCH);
        exp0recbuf = Tensor<float, PUB>(BATCH);
        brcbbuf = Tensor<float, PUB>(BATCH);
        sfexpbuf = Tensor<float, PUB>(BATCH);

        // for exp1
        exp1maxbuf = Tensor<float, PUB>(BATCH);
        exp1buf = Tensor<float, PUB>(BATCH);
        exp1maskbuf = Tensor<float, PUB>(BATCH); // no need batch but it's ok to allocate more

        // for exp2
        exp2parta = Tensor<float, PUB>(BATCH);
        exp2partb = Tensor<float, PUB>(BATCH);
        exp2maxbufa = Tensor<float, PUB>(BATCH);
        exp2maxbufb = Tensor<float, PUB>(BATCH);
        exp2buf = Tensor<float, PUB>(BATCH);
        exp2maskbuf = Tensor<float, PUB>(BATCH); // no need batch but it's ok to allocate more

        // for mantissa
        mantbuf = Tensor<float, PUB>(BATCH);

        fp32inbuf = Tensor<float, PUB>(BATCH);
        fp32outbuf = Tensor<float, PUB>(BATCH);

        expmask = Tensor<uint32_t, PUB>(SEG_ELEMS);
        onesbuf = Tensor<float, PUB>(BATCH);
        twosbuf = Tensor<float, PUB>(SEG_ELEMS);
        bf16_seven_buf_bf16 = Tensor<bfloat16_t, PUB>(BATCH);
        bf16_seven_buf = Tensor<float, PUB>(BATCH);
        bf16_general_buf = Tensor<bfloat16_t, PUB>(BATCH);
        signmaskbuf1 = Tensor<uint16_t, PUB>(SIGN_MASK_ELEMS);
        signmaskbuf2 = Tensor<uint16_t, PUB>(SIGN_MASK_ELEMS);
        // non-finite handling buffers (nfmaxbuf must be BATCH-sized: the block-flag
        // reduction writes as many elements as the scale path's exp0buf)
        nfbuf = Tensor<float, PUB>(BATCH);
        nfmaxbuf = Tensor<float, PUB>(BATCH);
        zerobuf = Tensor<float, PUB>(SEG_ELEMS);
        nanbuf = Tensor<float, PUB>(SEG_ELEMS);
        InitConstants();
    }

    aifunc void InitConstants() {
        for (int i = 0; i < SEG_ELEMS; ++i) {
            *signmaskbuf1[i * 2 + 1].ptr() = 0x3f80;
            *signmaskbuf1[i * 2].ptr() = 0;
            *signmaskbuf2[i * 2 + 1].ptr() = 0x8000;
            *signmaskbuf2[i * 2].ptr() = 0;
        }

        vector_dup(expmask.ptr(), 0x7F800000, 1, 1, 1, VEC_REP_ELEMS, VEC_REP_ELEMS);
        vector_dup(onesbuf.ptr(), 1.0f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        vector_dup(twosbuf.ptr(), 2.0f, 1, 1, 1, VEC_REP_ELEMS, VEC_REP_ELEMS);
        vector_dup(zerobuf.ptr(), 0.0f, 1, 1, 1, VEC_REP_ELEMS, VEC_REP_ELEMS);
        vector_dup(((__ubuf__ uint32_t *)nanbuf.ptr()), 0x7FC00000, 1, 1, 1, VEC_REP_ELEMS, VEC_REP_ELEMS);
        vector_dup(nfbuf.ptr(), 1.0f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        vector_dup(bf16_seven_buf.ptr(), (1.0f / 7.0f), BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322bf16r(bf16_seven_buf_bf16.ptr(), bf16_seven_buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_bf162f32(bf16_seven_buf.ptr(), bf16_seven_buf_bf16.ptr(), BATCH / SEG_ELEMS, 1, 1, 8, 4);
    }

    aifunc void tiling() {
        shape.N_ceil = (shape.N + BATCH - 1) / BATCH * BATCH;
        int total_split = shape.M * shape.N_ceil / BATCH;
        int split_per_core = (total_split + GetBlockNum() - 1) / GetBlockNum();
        shape.mn0 = split_per_core * GetBlockIdx();
        shape.mn1 = split_per_core + shape.mn0;
        if (shape.mn1 > total_split) {
            shape.mn1 = total_split;
        }
    }

    aifunc void Process() {
        input_empty.setall();
        output_empty.setall();
        for (int mn = shape.mn0; mn < shape.mn1; ++mn) {
            Compute(mn);
        }
        input_empty.release();
        output_empty.release();
    }

    aifunc void Compute(int mn) {
        mn_ = mn;
        m_ = mn / (shape.N_ceil / BATCH);
        n_ = mn % (shape.N_ceil / BATCH);
        n_tail_ = (n_ + 1) * BATCH - shape.N;
        LoadInput();
        ZeroTail();
        PrepareSignAbs();
        HandleNonFinite();
        ReduceBlockMax();
        PoisonBlock();
        RoundScaleE6M2();
        ComputeExp1();
        Exp2PartMax();
        Exp2Select();
        ComputeMantissa();
        Reconstruct();
        StoreOutput();
    }

    aifunc void LoadInput() {
        input_empty.wait();
        if (n_ * BATCH + BATCH <= shape.N) {
            copy_gm_to_ubuf(xbuf.get(mn_).vptr(), xmtx[m_ * shape.N + n_ * BATCH].vptr(), 0, 1,
                BATCH * sizeof(T) / COPY_UNIT_BYTES, 0, 0);
        } else {
            if constexpr (sizeof(T) == 2) {
                copy_gm_to_ubuf_align_b16(xbuf.get(mn_).vptr(), xmtx[m_ * shape.N + n_ * BATCH].vptr(), 0, 1,
                    (shape.N - n_ * BATCH) * sizeof(T), 0, 0, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b32(xbuf.get(mn_).vptr(), xmtx[m_ * shape.N + n_ * BATCH].vptr(), 0, 1,
                    (shape.N - n_ * BATCH) * sizeof(T), 0, 0, 0, 0);
            }
        }
        input_ready.set();

        output_empty.wait();
        input_ready.wait();
        if constexpr (std::is_same<T, float>::value) {
            inp_tsr_ = xbuf.get(mn_);
            out_tsr_ = outbuf.get(mn_);
        } else {
            inp_tsr_ = fp32inbuf;
            out_tsr_ = fp32outbuf;
            vconv_bf162f32(inp_tsr_.ptr(), xbuf.get(mn_).ptr(), BATCH / SEG_ELEMS, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }
    }

    aifunc void ZeroTail() {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        if (n_tail_ > 0) {
            // Tail masking: the host guarantees N is a multiple of BATCH (512) by
            // zero-padding, so this path is dead code in normal operation.  Kept as a
            // safety net in case the kernel is called directly with non-aligned N.
            for (int tail = n_tail_; tail > 0; tail -= SEG_ELEMS) {
                if (tail < SEG_ELEMS) {
                    uint64_t mask = 1;
                    for (int i = 1; i < tail; ++i) {
                        mask |= (mask << 1);
                    }
                    set_vector_mask(0, mask);
                }
                pipe_barrier(PIPE_V);
                vector_dup(inp_tsr_[BATCH - SEG_ELEMS].ptr(), 0.0f, 1, 1, 1, VEC_REP_ELEMS, VEC_REP_ELEMS);
                pipe_barrier(PIPE_V);
            }
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);
    }

    aifunc void PrepareSignAbs() {
        // element-wise abs + sign (sign = +/-1.0 via the 16-bit fp16 pattern trick)
        pipe_barrier(PIPE_V);
        vabs(absbuf.ptr(), inp_tsr_.ptr(), BATCH / SEG_ELEMS, 1, 1, 8, 8);
        vand((__ubuf__ uint16_t *)signbuf.ptr(), (__ubuf__ uint16_t *)inp_tsr_.ptr(), signmaskbuf2.ptr(),
            BATCH / SEG_ELEMS, 1, 1, 0, 8, 8, 0);
        pipe_barrier(PIPE_V);
        vor((__ubuf__ uint16_t *)signbuf.ptr(), (__ubuf__ uint16_t *)signbuf.ptr(), signmaskbuf1.ptr(),
            BATCH / SEG_ELEMS, 1, 1, 0, 8, 8, 0);
    }

    // ---- non-finite handling (aligned with the HiFloat4 golden) ----
    // vsel(dst, src0, mask) = dst = mask ? src0 : <value in the compare-mask register>
    // (the register is loaded with set_cmpmask before each select).
    aifunc void HandleNonFinite() {
        // nf marker: 1 where |x| is non-finite (NaN/Inf), 0 elsewhere.
        vcmpvs_le((__ubuf__ uint8_t *)exp1maskbuf.ptr(), absbuf.ptr(), 3.4028234663852886e+38f, BATCH / SEG_ELEMS, 1, 1,
            8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(onesbuf.vptr()); // fallback = 1.0 (non-finite side)
        pipe_barrier(PIPE_V);
        vsel(nfbuf.ptr(), zerobuf.ptr(), exp1maskbuf.vptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 0, 1, 1);
        pipe_barrier(PIPE_V);
        // zero the non-finite magnitudes (re-compare: the first select may consume the mask)
        vcmpvs_le((__ubuf__ uint8_t *)exp2maskbuf.ptr(), absbuf.ptr(), 3.4028234663852886e+38f, BATCH / SEG_ELEMS, 1, 1,
            8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(zerobuf.vptr()); // fallback = 0.0 (non-finite side)
        pipe_barrier(PIPE_V);
        vsel(absbuf.ptr(), absbuf.ptr(), exp2maskbuf.vptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 1, 1);
        pipe_barrier(PIPE_V);
        // per-64-block poison flag: max of the markers (same reduction as the scale path)
        vcmax(nfmaxbuf.ptr(), nfbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 8, ONLY_VALUE);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t *)brcbbuf.ptr(), (__ubuf__ uint32_t *)nfmaxbuf.ptr(), 1, 8, (BATCH + 511) / 512);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t *)nfmaxbuf.ptr(), (__ubuf__ uint32_t *)brcbbuf.ptr(), 1, 8, BATCH / SEG_ELEMS);
        pipe_barrier(PIPE_V);
    }

    aifunc void ReduceBlockMax() {
        // block max and the bf16 scale factor, clamped to [2^-48, 49152]
        vcmax(exp0buf.ptr(), absbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 8, ONLY_VALUE);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t *)brcbbuf.ptr(), (__ubuf__ uint32_t *)exp0buf.ptr(), 1, 8, (BATCH + 511) / 512);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t *)exp0buf.ptr(), (__ubuf__ uint32_t *)brcbbuf.ptr(), 1, 8, BATCH / SEG_ELEMS);
        pipe_barrier(PIPE_V);
        vmul(exp0buf.ptr(), exp0buf.ptr(), bf16_seven_buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322bf16r(bf16_general_buf.ptr(), exp0buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_bf162f32(exp0buf.ptr(), bf16_general_buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmins(exp0buf.ptr(), exp0buf.ptr(), 49152.0f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        uint32_t twopowneg48_int = 0x27800000;
        float twopowneg48 = *(float *)&twopowneg48_int;
        vmaxs(exp0buf.ptr(), exp0buf.ptr(), twopowneg48, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    aifunc void PoisonBlock() {
        // NaN block poison: blocks with any non-finite element dequantize to NaN as
        // a whole (aligned with the HiFloat4 golden). Only the clean mask is computed
        // here (1 = clean block); NaN replacement is deferred to the Reconstruct output
        // stage so NaN scales never enter intermediate vector arithmetic (vdiv/vconv
        // raise AI Core exceptions on NaN operands).
        vcmpvs_lt((__ubuf__ uint8_t *)exp1maskbuf.ptr(), nfmaxbuf.ptr(), 0.5f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    aifunc void RoundScaleE6M2() {
        // convert scale factor to e6m2
        vand((__ubuf__ uint16_t *)sfexpbuf.ptr(), (__ubuf__ uint16_t *)expmask.ptr(),
            (__ubuf__ uint16_t *)exp0buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 0, 8);
        pipe_barrier(PIPE_V);
        if constexpr (E6MX == 2) {
            vmuls(sfexpbuf.ptr(), sfexpbuf.ptr(), 0.25f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        } else if constexpr (E6MX == 1) {
            vmuls(sfexpbuf.ptr(), sfexpbuf.ptr(), 0.5f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        } else {
        }
        pipe_barrier(PIPE_V);
        vdiv(exp0buf.ptr(), exp0buf.ptr(), sfexpbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f32r(exp0buf.ptr(), exp0buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(exp0buf.ptr(), exp0buf.ptr(), sfexpbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

    aifunc void ComputeExp1() {
        // per-8-group micro exponent: exp1 = (max8 * rec >= 4) ? 2 : 1
        vcgmax(exp1maxbuf.ptr(), absbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 8);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t *)exp1buf.ptr(), (__ubuf__ uint32_t *)exp1maxbuf.ptr(), 1, 8, BATCH / SEG_ELEMS);
        pipe_barrier(PIPE_V);
        vdiv(exp0recbuf.ptr(), onesbuf.ptr(), exp0buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322bf16r(bf16_general_buf.ptr(), exp0recbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_bf162f32(exp0recbuf.ptr(), bf16_general_buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmul(exp1buf.ptr(), exp1buf.ptr(), exp0recbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vcmpvs_ge((__ubuf__ uint8_t *)exp1maskbuf.ptr(), exp1buf.ptr(), 4.0f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(onesbuf.vptr());
        pipe_barrier(PIPE_V);
        vsel(exp1buf.ptr(), twosbuf.ptr(), exp1maskbuf.vptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 0, 1, 1);
        pipe_barrier(PIPE_V);
    }

    aifunc void Exp2PartMax() {
        // per-4-group maxima: split even/odd halves, zero the other half, reduce
        copy_ubuf_to_ubuf(exp2parta.vptr(), absbuf.vptr(), 0, 1, BATCH / 8, 0, 0);
        copy_ubuf_to_ubuf(exp2partb.vptr(), absbuf.vptr(), 0, 1, BATCH / 8, 0, 0);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, 0x0f0f0f0f0f0f0f0f);
        pipe_barrier(PIPE_V);
        vector_dup(exp2parta.ptr(), -999999.0f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, 0xf0f0f0f0f0f0f0f0);
        pipe_barrier(PIPE_V);
        vector_dup(exp2partb.ptr(), -999999.0f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);
        vcgmax(exp2maxbufa.ptr(), exp2parta.ptr(), BATCH / SEG_ELEMS, 1, 1, 8);
        vcgmax(exp2maxbufb.ptr(), exp2partb.ptr(), BATCH / SEG_ELEMS, 1, 1, 8);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t *)exp2parta.ptr(), (__ubuf__ uint32_t *)exp2maxbufa.ptr(), 1, 8, BATCH / SEG_ELEMS);
        vbrcb((__ubuf__ uint32_t *)exp2partb.ptr(), (__ubuf__ uint32_t *)exp2maxbufb.ptr(), 1, 8, BATCH / SEG_ELEMS);
        pipe_barrier(PIPE_V);
    }

    aifunc void Exp2Select() {
        // combine the two halves, then exp2 = (max4 / exp1 * rec >= 2) ? 2 : 1
        set_vector_mask(0, 0x0f0f0f0f0f0f0f0f);
        pipe_barrier(PIPE_V);
        vector_dup(exp2parta.ptr(), 0.0f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, 0xf0f0f0f0f0f0f0f0);
        pipe_barrier(PIPE_V);
        vector_dup(exp2partb.ptr(), 0.0f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);
        vadd(exp2buf.ptr(), exp2parta.ptr(), exp2partb.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(exp2buf.ptr(), exp2buf.ptr(), exp1buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(exp2buf.ptr(), exp2buf.ptr(), exp0recbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vcmpvs_ge((__ubuf__ uint8_t *)exp2maskbuf.ptr(), exp2buf.ptr(), 2.0f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(onesbuf.vptr());
        pipe_barrier(PIPE_V);
        vsel(exp2buf.ptr(), twosbuf.ptr(), exp2maskbuf.vptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 0, 1, 1);
        pipe_barrier(PIPE_V);
    }

    aifunc void ComputeMantissa() {
        // in-group value: bf16-round(abs / exp2 / exp1 * rec), then S1P2 (2 frac bits,
        // round-half-up via *4+0.5 then truncate), clamped to +/-1.75
        vdiv(mantbuf.ptr(), absbuf.ptr(), exp2buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(mantbuf.ptr(), mantbuf.ptr(), exp1buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(mantbuf.ptr(), mantbuf.ptr(), exp0recbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // round the in-group mantissa quotient to bf16 before the S1P2 round
        vconv_f322bf16r(bf16_general_buf.ptr(), mantbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_bf162f32(mantbuf.ptr(), bf16_general_buf.ptr(), BATCH / SEG_ELEMS, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);

        vmuls(mantbuf.ptr(), mantbuf.ptr(), (float)(1 << (shape.MB - 1)), BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(mantbuf.ptr(), mantbuf.ptr(), 0.5f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f32z(mantbuf.ptr(), mantbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(mantbuf.ptr(), mantbuf.ptr(), 1.0f / (float)(1 << (shape.MB - 1)), BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        float maxmant = 2.0f - 1.0f / (float)(1 << (shape.MB - 1));
        vmins(mantbuf.ptr(), mantbuf.ptr(), maxmant, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmaxs(mantbuf.ptr(), mantbuf.ptr(), -1.0f * maxmant, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    aifunc void Reconstruct() {
        // out = sign * mant * exp2 * exp1 * scale
        vmul(out_tsr_.ptr(), exp2buf.ptr(), mantbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(out_tsr_.ptr(), exp1buf.ptr(), out_tsr_.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(out_tsr_.ptr(), exp0buf.ptr(), out_tsr_.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(out_tsr_.ptr(), out_tsr_.ptr(), signbuf.ptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // Replace poisoned blocks with NaN as a whole (clean mask=1 keeps the computed
        // result; nfmaxbuf still holds the per-block flags).
        vcmpvs_lt((__ubuf__ uint8_t *)exp2maskbuf.ptr(), nfmaxbuf.ptr(), 0.5f, BATCH / SEG_ELEMS, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(nanbuf.vptr()); // fallback = quiet NaN (poisoned side)
        pipe_barrier(PIPE_V);
        vsel(out_tsr_.ptr(), out_tsr_.ptr(), exp2maskbuf.vptr(), BATCH / SEG_ELEMS, 1, 1, 1, 8, 8, 1, 1);
        pipe_barrier(PIPE_V);
    }

    aifunc void StoreOutput() {
        if constexpr (std::is_same<T, float>::value) {
        } else {
            vconv_f322bf16a(outbuf.get(mn_).ptr(), out_tsr_.ptr(), BATCH / SEG_ELEMS, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
        }

        output_ready.set();
        input_empty.set();

        output_ready.wait();
        // mte3
        if (n_tail_ > 0) {
            copy_ubuf_to_gm(out[m_ * shape.N + n_ * BATCH].vptr(), outbuf.get(mn_).vptr(), 0, 1,
                sizeof(T) * (BATCH - n_tail_), 0, 0, BM_ENABLE);
        } else {
            copy_ubuf_to_gm(out[m_ * shape.N + n_ * BATCH].vptr(), outbuf.get(mn_).vptr(), 0, 1,
                BATCH * sizeof(T) / COPY_UNIT_BYTES, 0, 0);
        }
        output_empty.set();
    }

private:
    TPipe pipe;
    Hifxv14ShapeInfo shape;
    Tensor<T, PGM> xmtx, out;
    int mn_, m_, n_, n_tail_;
    Tensor<float, PUB> inp_tsr_, out_tsr_;
    DBuff<T, PUB> xbuf, outbuf;
    Tensor<float, PUB> absbuf, signbuf, exp0buf, brcbbuf, sfexpbuf, exp0recbuf;              // for exp0
    Tensor<float, PUB> exp1maxbuf, exp1buf, exp1maskbuf;                                     // for exp1
    Tensor<float, PUB> exp2parta, exp2partb, exp2maxbufa, exp2maxbufb, exp2buf, exp2maskbuf; // for exp2
    Tensor<float, PUB> mantbuf;
    Tensor<float, PUB> fp32inbuf, fp32outbuf;
    Tensor<uint32_t, PUB> expmask;
    Tensor<float, PUB> onesbuf, twosbuf, bf16_seven_buf;
    Tensor<float, PUB> nfbuf, nfmaxbuf, zerobuf, nanbuf;
    Tensor<bfloat16_t, PUB> bf16_seven_buf_bf16, bf16_general_buf;
    Tensor<uint16_t, PUB> signmaskbuf1, signmaskbuf2;
    DEvent<PIPE_MTE2, PIPE_V> input_ready{3, 4};
    DEvent<PIPE_V, PIPE_MTE2> input_empty{3, 4};
    DEvent<PIPE_V, PIPE_MTE3> output_ready{3, 4};
    DEvent<PIPE_MTE3, PIPE_V> output_empty{3, 4};
    static constexpr int BATCH = 512;
    // Vector instructions operate on 64-element segments (8x8 repeat units);
    // BATCH / SEG_ELEMS is the number of segments per batch.
    static constexpr int SEG_ELEMS = 64;
    // Two-segment uint16 buffer for sign masks (one mask for the high and low
    // 16 bits of each element).
    static constexpr int SIGN_MASK_ELEMS = 2 * SEG_ELEMS;
    // Length unit of copy_gm_to_ubuf (bytes).
    static constexpr int COPY_UNIT_BYTES = 32;
    // Repeat row/col count of a single vector_dup segment.
    static constexpr int VEC_REP_ELEMS = 8;
    static constexpr float eps = 5.421011e-20;
};

extern "C" __global__ __aicore__ void hifx_kernel(GM_ADDR xmtx, GM_ADDR out, int M, int N, int mant_bit) {
    if ASCEND_IS_AIV {
#ifdef __DAV_C220_VEC__
        Hifv14Kernel<float, 2> vec;
        vec.Init(xmtx, out, M, N, mant_bit);
        vec.Process();
#endif
    }
}

extern "C" __global__ __aicore__ void hifx_kernel_bf16(GM_ADDR xmtx, GM_ADDR out, int M, int N, int mant_bit) {
    if ASCEND_IS_AIV {
#ifdef __DAV_C220_VEC__
        Hifv14Kernel<bfloat16_t, 2> vec;
        vec.Init(xmtx, out, M, N, mant_bit);
        vec.Process();
#endif
    }
}

extern "C" void run_hifx_kernel(
    uint32_t blockDim, void *stream, uint8_t *xmtx, uint8_t *out, int M, int N, int mant_bit) {
    hifx_kernel<<<40, nullptr, stream>>>(xmtx, out, M, N, mant_bit);
}

extern "C" void run_hifx_kernel_bf16(
    uint32_t blockDim, void *stream, uint8_t *xmtx, uint8_t *out, int M, int N, int mant_bit) {
    hifx_kernel_bf16<<<40, nullptr, stream>>>(xmtx, out, M, N, mant_bit);
}
