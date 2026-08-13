#pragma once

// HiF4 (hifx4) plain 1D fake-quant kernels, auto-migrated from the in-repo
// EasyASC DSL hifx.py. This umbrella header includes the two generated
// per-dtype headers and keeps the host launchers (run_hifx_kernel /
// run_hifx_kernel_bf16) that the CXX torch binding links against.

#include "hifx_quant_op_fp32.h"
#include "hifx_quant_op_bf16.h"

extern "C" __global__ __aicore__ void hifx_kernel(GM_ADDR xmtx, GM_ADDR out, int M, int N, int mant_bit) {
    PipeBarrier<PIPE_ALL>();
    if ASCEND_IS_AIV {
        hifx_detail::hifx_kernel_fp32_vec(xmtx, out, (GM_ADDR) nullptr, M, N, mant_bit);
    }
}

extern "C" __global__ __aicore__ void hifx_kernel_bf16(GM_ADDR xmtx, GM_ADDR out, int M, int N, int mant_bit) {
    PipeBarrier<PIPE_ALL>();
    if ASCEND_IS_AIV {
        hifx_detail::hifx_kernel_bf16_vec(xmtx, out, (GM_ADDR) nullptr, M, N, mant_bit);
    }
}

void run_hifx_kernel(uint32_t blockDim, void *stream, uint8_t *xmtx, uint8_t *out, int M, int N, int mant_bit) {
    hifx_kernel<<<40, nullptr, stream>>>(xmtx, out, M, N, mant_bit);
}

void run_hifx_kernel_bf16(uint32_t blockDim, void *stream, uint8_t *xmtx, uint8_t *out, int M, int N, int mant_bit) {
    hifx_kernel_bf16<<<40, nullptr, stream>>>(xmtx, out, M, N, mant_bit);
}
