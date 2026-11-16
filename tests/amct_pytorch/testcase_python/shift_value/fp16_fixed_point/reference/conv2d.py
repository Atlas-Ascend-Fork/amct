#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch

INT32MIN, INT32MAX = -(2**31), 2**31 - 1


def _f2i32_x86(x_i64):
    oob = (x_i64 > INT32MAX) | (x_i64 < INT32MIN)
    return torch.where(oob, torch.full_like(x_i64, INT32MIN), x_i64)


def _fixpoint_src(exp, man, sig, is_norm, max_exp):
    delta = max_exp.unsqueeze(-1) - exp
    base = torch.where(is_norm, 4, 5)
    e = torch.clamp(base - delta, -11, 5)
    m = torch.where(is_norm, man.to(torch.int64) + 1024, man.to(torch.int64))
    zero = torch.zeros_like(e)
    e_pos = torch.where(e >= 0, e, zero).to(torch.int64)
    e_neg = torch.where(e < 0, -e, zero).to(torch.int64)
    src = torch.where(e >= 0, m << e_pos, m >> e_neg)
    sgn = torch.where(sig == 1, -1, 1).to(torch.int64)
    return src * sgn


def _rint_half_even(mul, shift):
    pow2 = torch.ones_like(mul) << shift
    low = mul & (pow2 - 1)
    half = pow2 >> 1
    lsb = (mul >> shift) & 1
    carry = (low > half) | ((low == half) & (lsb == 1))
    return (mul >> shift) + carry


def torch_mmad_f16f16(a, b, bias, sv=42):
    M, K = a.shape
    N, _ = b.shape
    if K % 16:
        pad = 16 - K % 16
        a = torch.nn.functional.pad(a, (0, pad))
        b = torch.nn.functional.pad(b, (0, pad))
        K = a.shape[-1]
    au = a.contiguous().view(torch.int16).to(torch.int32)
    by = b.contiguous().view(torch.int16).to(torch.int32)
    a_sig, a_exp, a_man = (au >> 15) & 1, (au >> 10) & 0x1F, au & 0x3FF
    b_sig, b_exp, b_man = (by >> 15) & 1, (by >> 10) & 0x1F, by & 0x3FF
    a_norm, b_norm = a_exp > 0, b_exp > 0
    bias_scaled = torch.round(bias.to(torch.float32) * float(2 ** (58 - sv))).to(
        torch.int64
    )
    bias_s32 = _f2i32_x86(bias_scaled)
    C = torch.zeros((M, N), dtype=torch.int64)
    for j in range(K // 16):
        ks, ke = 16 * j, 16 * j + 16
        a_max = a_exp[:, ks:ke].amax(-1).to(torch.int64)
        b_max = b_exp[:, ks:ke].amax(-1).to(torch.int64)
        a_src = _fixpoint_src(
            a_exp[:, ks:ke], a_man[:, ks:ke], a_sig[:, ks:ke], a_norm[:, ks:ke], a_max
        )
        b_src = _fixpoint_src(
            b_exp[:, ks:ke], b_man[:, ks:ke], b_sig[:, ks:ke], b_norm[:, ks:ke], b_max
        )
        mul = (a_src.unsqueeze(1) * b_src.unsqueeze(0)).sum(-1)
        shift = torch.clamp(sv - a_max.unsqueeze(1) - b_max.unsqueeze(0), -26, 36)
        zero = torch.zeros_like(shift)
        s_pos = torch.where(shift >= 0, shift, zero)
        s_neg = torch.where(shift < 0, -shift, zero)
        mul_result = torch.where(shift >= 0, _rint_half_even(mul, s_pos), mul << s_neg)
        C = torch.clamp(C + mul_result, INT32MIN, INT32MAX)
    C = torch.clamp(C + bias_s32.unsqueeze(0), INT32MIN, INT32MAX)
    return C


def torch_mmad_optimize(a, b, bias, sv=42):
    c = torch_mmad_f16f16(a, b, bias, sv)
    v64 = (c.to(torch.float64) * (2.0 ** (-58 + sv))).numpy()
    return torch.from_numpy(v64.astype(np.float16))


def _ceil_div(a, b):
    return (a + b - 1) // b


def _conv_im2col(
    x, kh, kw, stride_h, stride_w, dil_h, dil_w, pads, h_out, w_out, k1, k0=16
):
    cin, h_in, w_in = x.shape
    pad_t, pad_b, pad_l, pad_r = pads
    col_c = k1 * k0
    xp = torch.zeros(
        col_c,
        h_in + pad_t + pad_b,
        w_in + pad_l + pad_r,
        dtype=x.dtype,
        device=x.device,
    )
    xp[:cin, pad_t : pad_t + h_in, pad_l : pad_l + w_in] = x
    xp = xp.reshape(k1, k0, xp.shape[1], xp.shape[2]).permute(0, 2, 3, 1)

    idx_h = (torch.arange(kh, device=x.device) * dil_h).unsqueeze(1) + torch.arange(
        h_out, device=x.device
    ).unsqueeze(0) * stride_h
    idx_w = (torch.arange(kw, device=x.device) * dil_w).unsqueeze(1) + torch.arange(
        w_out, device=x.device
    ).unsqueeze(0) * stride_w

    rows = xp[:, idx_h]
    patch = rows[:, :, :, idx_w]
    patch = patch.permute(2, 4, 0, 1, 3, 5)
    return patch.reshape(h_out * w_out, -1)


def _conv_weight_to_kmat(w, cin, kh, kw, k1, k0=16):
    cout = w.shape[0]
    wp = torch.zeros(cout, k1 * k0, kh, kw, dtype=w.dtype, device=w.device)
    wp[:, :cin] = w
    wp = wp.reshape(cout, k1, k0, kh, kw).permute(0, 1, 3, 4, 2)
    return wp.reshape(cout, -1)


def conv_f16f16(x, w, bias, pads, strides, dilations, groups, scale=42):
    sv = scale
    N, Cin, H_in, W_in = x.shape
    cout, cin_w, kh, kw = w.shape
    pad_t, pad_b, pad_l, pad_r = pads
    stride_h, stride_w = strides
    dil_h, dil_w = dilations

    hk_dil = (kh - 1) * dil_h + 1
    wk_dil = (kw - 1) * dil_w + 1
    h_out = (H_in + pad_t + pad_b - hk_dil) // stride_h + 1
    w_out = (W_in + pad_l + pad_r - wk_dil) // stride_w + 1

    cin_pg = Cin // groups
    cout_pg = cout // groups
    k0 = 16
    k1 = _ceil_div(cin_pg, k0)

    if bias is None:
        bias = torch.zeros(cout, dtype=torch.float16, device=x.device)
    bias = bias.to(torch.float32)

    out = torch.zeros(N, cout, h_out, w_out, dtype=torch.float16, device=x.device)
    for b in range(N):
        for g in range(groups):
            x_g = x[b, g * cin_pg : (g + 1) * cin_pg]
            col = _conv_im2col(
                x_g,
                kh,
                kw,
                stride_h,
                stride_w,
                dil_h,
                dil_w,
                pads,
                h_out,
                w_out,
                k1,
                k0,
            )
            w_g = w[g * cout_pg : (g + 1) * cout_pg]
            wkmat = _conv_weight_to_kmat(w_g, cin_pg, kh, kw, k1, k0)
            bias_g = bias[g * cout_pg : (g + 1) * cout_pg]
            res = torch_mmad_optimize(col, wkmat, bias_g, sv)
            out[b, g * cout_pg : (g + 1) * cout_pg] = res.T.reshape(
                cout_pg, h_out, w_out
            )

    return out
