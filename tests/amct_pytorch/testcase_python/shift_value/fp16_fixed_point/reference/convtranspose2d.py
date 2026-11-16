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

import torch
import torch.nn.functional as F

INT32MIN, INT32MAX = -(2**31), 2**31 - 1


def _f2i32_x86(x):
    oob = (x > INT32MAX) | (x < INT32MIN)
    return torch.where(oob, torch.full_like(x, INT32MIN), x)


def _fixpoint_src_f16_int64(exp, man, sig, is_norm, max_exp):
    delta = max_exp.unsqueeze(-1) - exp
    base = torch.where(is_norm, 4, 5)
    e = torch.clamp(base - delta, -16, 15)
    m = torch.where(is_norm, man.to(torch.int64) + 1024, man.to(torch.int64))
    zero = torch.zeros_like(e)
    e_pos = torch.where(e >= 0, e, zero).to(torch.int64)
    e_neg = torch.where(e < 0, -e, zero).to(torch.int64)
    fixed = torch.where(e >= 0, m << e_pos, m >> e_neg)
    sgn = torch.where(sig == 1, -1, 1).to(torch.int64)
    return fixed * sgn


def _gemm_convt(a, b, sv, acc0):
    M, K = a.shape
    if K % 16:
        pad = 16 - K % 16
        a = F.pad(a, (0, pad))
        b = F.pad(b, (0, pad))
        K = a.shape[-1]
    au = a.contiguous().view(torch.int16).to(torch.int32)
    by = b.contiguous().view(torch.int16).to(torch.int32)
    a_sig, a_exp, a_man = (au >> 15) & 1, (au >> 10) & 0x1F, au & 0x3FF
    b_sig, b_exp, b_man = (by >> 15) & 1, (by >> 10) & 0x1F, by & 0x3FF
    a_norm, b_norm = a_exp > 0, b_exp > 0
    acc = acc0.to(torch.int64)
    for t in range(K // 16):
        ks, ke = 16 * t, 16 * t + 16
        a_max = a_exp[:, ks:ke].amax(-1)
        b_max = b_exp[:, ks:ke].amax(-1)
        a_val = _fixpoint_src_f16_int64(
            a_exp[:, ks:ke], a_man[:, ks:ke], a_sig[:, ks:ke], a_norm[:, ks:ke], a_max
        )
        b_val = _fixpoint_src_f16_int64(
            b_exp[:, ks:ke], b_man[:, ks:ke], b_sig[:, ks:ke], b_norm[:, ks:ke], b_max
        )
        ts = (a_val.unsqueeze(1) * b_val.unsqueeze(0)).sum(-1)
        shift = torch.clamp(sv - a_max.unsqueeze(1) - b_max.unsqueeze(0), -26, 36)

        scaled = ts.to(torch.float64) * torch.exp2((-shift).to(torch.float64))
        acc = torch.round(acc.to(torch.float64) + scaled)
        acc = torch.clamp(acc, INT32MIN, INT32MAX).to(torch.int64)
    return acc


def _upsample_pad(
    x,
    kh,
    kw,
    stride_h,
    stride_w,
    dil_h,
    dil_w,
    pad_t,
    pad_b,
    pad_l,
    pad_r,
    opad_h,
    opad_w,
):
    cin, h_in, w_in = x.shape
    up_h = (h_in - 1) * stride_h + 1
    up_w = (w_in - 1) * stride_w + 1
    hk_dil = (kh - 1) * dil_h + 1
    wk_dil = (kw - 1) * dil_w + 1
    lead_t = hk_dil - 1 - pad_t
    lead_b = hk_dil - 1 - pad_b
    lead_l = wk_dil - 1 - pad_l
    lead_r = wk_dil - 1 - pad_r
    h_pad = up_h + lead_t + lead_b + opad_h
    w_pad = up_w + lead_l + lead_r + opad_w
    up = torch.zeros(cin, h_pad, w_pad, dtype=x.dtype, device=x.device)
    up[:, lead_t : lead_t + up_h : stride_h, lead_l : lead_l + up_w : stride_w] = x
    return up


def _im2col(up, kh, kw, dil_h, dil_w, h_out, w_out):
    idx_h = (torch.arange(kh, device=up.device) * dil_h).unsqueeze(1) + torch.arange(
        h_out, device=up.device
    ).unsqueeze(0)
    idx_w = (torch.arange(kw, device=up.device) * dil_w).unsqueeze(1) + torch.arange(
        w_out, device=up.device
    ).unsqueeze(0)

    rows = up[:, idx_h]
    patch = rows[:, :, :, idx_w]
    patch = patch.permute(2, 4, 1, 3, 0)
    return patch.reshape(h_out * w_out, -1)


def _convt_weight_to_kmat(w_group):
    flipped = torch.flip(w_group, [2, 3])
    return flipped.permute(1, 2, 3, 0).reshape(w_group.shape[1], -1)


def convt_f16f16_int32(
    x, w, bias, pads, strides, dilations, groups, output_shape, scale=42
):
    sv = scale
    N, Cin, H_in, W_in = x.shape
    cin_w, cout_pg, kh, kw = w.shape
    Cout = cout_pg * groups
    _, _, H_out, W_out = output_shape
    pad_t, pad_b, pad_l, pad_r = pads
    stride_h, stride_w = strides
    dil_h, dil_w = dilations

    opad_h = max(
        0, H_out - ((H_in - 1) * stride_h + (kh - 1) * dil_h + 1 - pad_t - pad_b)
    )
    opad_w = max(
        0, W_out - ((W_in - 1) * stride_w + (kw - 1) * dil_w + 1 - pad_l - pad_r)
    )

    cin_pg = Cin // groups
    col_c = ((cin_pg + 15) // 16) * 16

    result = torch.zeros(N, Cout, H_out, W_out, dtype=torch.int32, device=x.device)
    if bias is not None:
        bias_f32 = bias.to(torch.float32)
        bias_int = _f2i32_x86(torch.round(bias_f32 * float(1 << (58 - sv))))
        result = result + bias_int.view(1, Cout, 1, 1)

    for n in range(N):
        for g in range(groups):
            cs = g * cin_pg
            x_g = x[n, cs : cs + cin_pg]
            w_g = w[cs : cs + cin_pg]
            if cin_pg < col_c:
                x_g = F.pad(x_g, (0, 0, 0, 0, 0, col_c - cin_pg))
                w_g = F.pad(w_g, (0, 0, 0, 0, 0, 0, 0, col_c - cin_pg))
            up = _upsample_pad(
                x_g,
                kh,
                kw,
                stride_h,
                stride_w,
                dil_h,
                dil_w,
                pad_t,
                pad_b,
                pad_l,
                pad_r,
                opad_h,
                opad_w,
            )
            col = _im2col(up, kh, kw, dil_h, dil_w, H_out, W_out)
            wt = _convt_weight_to_kmat(w_g)
            acc0 = (
                result[n, g * cout_pg : (g + 1) * cout_pg]
                .permute(1, 2, 0)
                .reshape(H_out * W_out, cout_pg)
            )
            mm = _gemm_convt(col, wt, sv, acc0)
            mm = mm.T.reshape(cout_pg, H_out, W_out)
            result[n, g * cout_pg : (g + 1) * cout_pg] = mm.to(torch.int32)
    result = torch.clamp(result, INT32MIN, INT32MAX)
    return result


def convt_f16f16(
    x,
    w,
    bias,
    pads,
    strides,
    dilations,
    groups,
    output_shape,
    scale=42,
):
    result = convt_f16f16_int32(
        x, w, bias, pads, strides, dilations, groups, output_shape, scale
    )
    out = (result.to(torch.float32) / float(1 << (58 - scale))).to(torch.float16)
    return out
