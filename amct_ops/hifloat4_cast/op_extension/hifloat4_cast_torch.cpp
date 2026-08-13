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

#include <cstdint>
#include <vector>

#include "ops.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Host launchers emitted by op_kernel/hifloat4_cast_kernel.cpp, which includes the
// generated ccec header hifx_quant_op.h (compiled with the ASC compiler; the launchers
// there use plain C++ linkage, so declare them the same way). They quantize a contiguous
// [M, N] buffer along N, with 64-element blocks.
void run_hifx_kernel(
    uint32_t blockDim, void *stream, uint8_t *xmtx, uint8_t *out, int M, int N, int mant_bit);
void run_hifx_kernel_bf16(
    uint32_t blockDim, void *stream, uint8_t *xmtx, uint8_t *out, int M, int N, int mant_bit);

namespace {
constexpr int HIF4_MANT_BIT = 3;   // hifx4: S1P2 element format, man_bits = 3
constexpr uint32_t HIF4_BLOCK_DIM = 40;
}  // namespace

namespace AscendKernel {

// The kernel quantizes along the last dim (blocks of 64, tail handled in-kernel). For qdim != -1
// we permute that dim to the last axis, run, then permute back. dtype/device are validated at the
// dispatcher entry (register.cpp).
at::Tensor Hifloat4CastTorch(const at::Tensor &input, int64_t qdim)
{
    int64_t nd = input.dim();
    TORCH_CHECK(nd > 0, "hifloat4_fake_quant: input must have at least one dimension");
    int64_t qd = qdim >= 0 ? qdim : qdim + nd;
    TORCH_CHECK(qd >= 0 && qd < nd, "hifloat4_fake_quant: qdim ", qdim, " out of range for ", nd, "-D input");

    bool needPermute = (qd != nd - 1);
    std::vector<int64_t> perm;
    at::Tensor xp;
    if (needPermute) {
        perm.reserve(nd);
        for (int64_t i = 0; i < nd; i++) {
            if (i != qd) {
                perm.push_back(i);
            }
        }
        perm.push_back(qd);
        xp = input.permute(perm).contiguous();
    } else {
        xp = input.contiguous();
    }

    int64_t n = xp.size(-1);
    int64_t m_raw = xp.numel() / n;

    // Pad N to a multiple of BATCH (512) so the kernel never sees a partial tail --
    // the kernel requires aligned input (groups of 64).
    // The kernel's tail-masking path becomes dead code; all tail handling is done here.
    constexpr int64_t BATCH = 512;
    int64_t n_pad = ((n + BATCH - 1) / BATCH) * BATCH;
    bool needSlice = (n_pad != n);

    auto inDtype = input.scalar_type();
    bool cvtFp32 = (inDtype == at::kHalf);
    at::Tensor xin = cvtFp32 ? xp.to(at::kFloat) : xp;
    if (needSlice) {
        xin = at::constant_pad_nd(xin, {0, n_pad - n}, 0.0);
    }
    int64_t n_kernel = n_pad;
    int64_t m = xin.numel() / n_kernel;
    at::Tensor output = at::empty_like(xin);

    auto stream = c10_npu::getCurrentNPUStream(input.device().index());
    void *aclStream = stream.stream();
    auto *xptr = reinterpret_cast<uint8_t *>(xin.data_ptr());
    auto *optr = reinterpret_cast<uint8_t *>(output.data_ptr());
    if (inDtype == at::kBFloat16) {
        run_hifx_kernel_bf16(HIF4_BLOCK_DIM, aclStream, xptr, optr, static_cast<int>(m), static_cast<int>(n_kernel),
            HIF4_MANT_BIT);
    } else {
        run_hifx_kernel(HIF4_BLOCK_DIM, aclStream, xptr, optr, static_cast<int>(m), static_cast<int>(n_kernel),
            HIF4_MANT_BIT);
    }

    // Slice off the pad, restoring the original N.
    if (needSlice) {
        output = output.slice(-1, 0, n);
    }

    if (cvtFp32) {
        output = output.to(inDtype);   // FP32 result -> original FP16
    }

    if (needPermute) {
        std::vector<int64_t> inv(nd);
        for (int64_t newPos = 0; newPos < nd; newPos++) {
            inv[perm[newPos]] = newPos;
        }
        output = output.permute(inv).contiguous();
    }
    return output;
}

}  // namespace AscendKernel
