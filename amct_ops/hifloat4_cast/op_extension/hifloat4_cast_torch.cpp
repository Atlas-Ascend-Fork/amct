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

// Host launchers defined in op_kernel/hifloat4_cast_kernel.cpp (compiled with the ASC
// compiler, extern "C" linkage). They quantize a contiguous [M, N] buffer along N,
// with 64-element blocks.
extern "C" void run_hifx_kernel(
    uint32_t blockDim, void *stream, uint8_t *xmtx, uint8_t *out, int M, int N, int mant_bit);
extern "C" void run_hifx_kernel_bf16(
    uint32_t blockDim, void *stream, uint8_t *xmtx, uint8_t *out, int M, int N, int mant_bit);

namespace {
constexpr int HIF4_MANT_BIT = 3; // hifx4: S1P2 element format, man_bits = 3
constexpr uint32_t HIF4_BLOCK_DIM = 40;
} // namespace

namespace AscendKernel {

namespace {

// Move the quant dim to the last axis so the kernel can always reduce along dim -1.
at::Tensor PermuteToLast(const at::Tensor &input, int64_t qd, std::vector<int64_t> &perm) {
    int64_t nd = input.dim();
    if (qd == nd - 1) {
        return input.contiguous();
    }
    perm.reserve(nd);
    for (int64_t i = 0; i < nd; i++) {
        if (i != qd) {
            perm.push_back(i);
        }
    }
    perm.push_back(qd);
    return input.permute(perm).contiguous();
}

// Pad N to a multiple of BATCH (512) so the kernel never sees a partial tile.
// The caller validates that N is a multiple of 64; the pad here only covers the
// kernel's 512-element tile granularity and is sliced off afterwards.
at::Tensor PadLastDim(const at::Tensor &xp, int64_t &n, bool &need_slice) {
    constexpr int64_t BATCH = 512;
    int64_t n_pad = ((n + BATCH - 1) / BATCH) * BATCH;
    need_slice = (n_pad != n);
    if (need_slice) {
        return at::constant_pad_nd(xp, {0, n_pad - n}, 0.0);
    }
    return xp;
}

at::Tensor Unpermute(const at::Tensor &output, const std::vector<int64_t> &perm) {
    int64_t nd = output.dim();
    std::vector<int64_t> inv(nd);
    for (int64_t new_pos = 0; new_pos < nd; new_pos++) {
        inv[perm[new_pos]] = new_pos;
    }
    return output.permute(inv).contiguous();
}

} // namespace

// The kernel quantizes along the last dim (blocks of 64). dtype/device are validated
// at the dispatcher entry (register.cpp).
at::Tensor Hifloat4CastTorch(const at::Tensor &input, int64_t qdim) {
    int64_t nd = input.dim();
    TORCH_CHECK(nd > 0, "hifloat4_fake_quant: input must have at least one dimension");
    int64_t qd = qdim >= 0 ? qdim : qdim + nd;
    TORCH_CHECK(qd >= 0 && qd < nd, "hifloat4_fake_quant: qdim ", qdim, " out of range for ", nd, "-D input");

    std::vector<int64_t> perm;
    at::Tensor xp = PermuteToLast(input, qd, perm);

    int64_t n = xp.size(-1);
    TORCH_CHECK(n > 0, "hifloat4_fake_quant: input must have a non-empty last dim");
    TORCH_CHECK(n % 64 == 0, "hifloat4_fake_quant: quant dim length must be a multiple of 64, got ", n);
    bool need_slice = false;
    at::Tensor xin = PadLastDim(xp, n, need_slice);
    int64_t m = xin.numel() / xin.size(-1);

    auto in_dtype = input.scalar_type();
    bool cvt_fp32 = (in_dtype == at::kHalf);
    if (cvt_fp32) {
        xin = xin.to(at::kFloat);
    }
    at::Tensor output = at::empty_like(xin);

    auto stream = c10_npu::getCurrentNPUStream(input.device().index());
    void *acl_stream = stream.stream();
    auto *xptr = reinterpret_cast<uint8_t *>(xin.data_ptr());
    auto *optr = reinterpret_cast<uint8_t *>(output.data_ptr());
    if (in_dtype == at::kBFloat16) {
        run_hifx_kernel_bf16(
            HIF4_BLOCK_DIM, acl_stream, xptr, optr, static_cast<int>(m), static_cast<int>(xin.size(-1)), HIF4_MANT_BIT);
    } else {
        run_hifx_kernel(
            HIF4_BLOCK_DIM, acl_stream, xptr, optr, static_cast<int>(m), static_cast<int>(xin.size(-1)), HIF4_MANT_BIT);
    }

    // Slice off the pad, restoring the original N.
    if (need_slice) {
        output = output.slice(-1, 0, n);
    }

    if (cvt_fp32) {
        output = output.to(in_dtype); // FP32 result -> original FP16
    }

    if (!perm.empty()) {
        output = Unpermute(output, perm);
    }
    return output;
}

} // namespace AscendKernel
