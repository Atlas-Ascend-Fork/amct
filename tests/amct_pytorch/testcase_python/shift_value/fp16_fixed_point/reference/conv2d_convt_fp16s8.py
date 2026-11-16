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


def _ceil_div(a, b):
    return (a + b - 1) // b


def _fixpoint_src_f16(exp, man, sig, is_norm, max_exp):
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


def _im2col_convt(up, kh, kw, dil_h, dil_w, h_out, w_out):
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


def f16xs8_gemm(a, b, sv, acc0=None):
    M, K = a.shape
    N = b.shape[0]
    if K % 16:
        pad = 16 - K % 16
        a = F.pad(a, (0, pad))
        b = F.pad(b, (0, pad))
        K = a.shape[-1]
    au = a.contiguous().view(torch.int16).to(torch.int32)
    a_sig, a_exp, a_man = (au >> 15) & 1, (au >> 10) & 0x1F, au & 0x3FF
    a_norm = a_exp > 0
    b_i64 = b.to(torch.int64)
    if acc0 is None:
        acc = torch.zeros(M, N, dtype=torch.float64, device=a.device)
    else:
        acc = acc0.to(torch.float64)

    for t in range(K // 16):
        ks, ke = 16 * t, 16 * t + 16
        a_max = a_exp[:, ks:ke].amax(-1)
        a_val = _fixpoint_src_f16(
            a_exp[:, ks:ke], a_man[:, ks:ke], a_sig[:, ks:ke], a_norm[:, ks:ke], a_max
        )
        ts = (a_val.unsqueeze(1) * b_i64[:, ks:ke].unsqueeze(0)).sum(-1)
        shift = (sv - a_max).to(torch.int64)
        scaled = ts.to(torch.float64) * torch.exp2(
            (-shift).to(torch.float64)
        ).unsqueeze(1)
        acc = torch.clamp(torch.round(acc + scaled), INT32MIN, INT32MAX)
    return acc.to(torch.int32)


def s8_bias_to_int32(bias, antiquant_scale):
    if bias is None:
        return None
    bias_f32 = bias.to(torch.float16).to(torch.float32)
    ascale = antiquant_scale.to(torch.float32)
    ascale = torch.where(ascale == 0, torch.ones_like(ascale), ascale)
    return torch.round(bias_f32 / ascale).to(torch.int64)


def _normalize_ascale(antiquant_scale, cout):
    a = antiquant_scale.to(torch.float32).reshape(-1)
    if a.shape[0] == 1:
        a = a.repeat(cout)
    assert a.shape[0] == cout, f"antiquant_scale size {a.shape[0]} != Cout {cout}"
    return a


def int32_dequant_to_fp16_s8(acc, antiquant_scale):
    ascale = antiquant_scale.to(torch.float32)
    if ascale.dim() == 1 and ascale.shape[-1] == acc.shape[-1]:
        ascale = ascale.view(1, -1)
    return (acc.to(torch.float32) * ascale).to(torch.float16)


def conv_f16s8(
    x,
    w,
    bias,
    antiquant_scale,
    pads,
    strides,
    dilations,
    groups,
    scale=13,
):
    sv = scale
    N, Cin, H_in, W_in = x.shape
    cout, cin_w, kh, kw = w.shape
    ascale = _normalize_ascale(antiquant_scale, cout)
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

    result = torch.zeros(N, h_out, w_out, cout, dtype=torch.int32, device=x.device)
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
            acc = f16xs8_gemm(col, wkmat, sv)
            if bias is not None:
                ascale_g = ascale[g * cout_pg : (g + 1) * cout_pg]
                bias_int = s8_bias_to_int32(
                    bias[g * cout_pg : (g + 1) * cout_pg], ascale_g
                )
                acc = torch.clamp(
                    acc.to(torch.int64) + bias_int.unsqueeze(0), INT32MIN, INT32MAX
                ).to(torch.int32)
            result[b, :, :, g * cout_pg : (g + 1) * cout_pg] = acc.reshape(
                h_out, w_out, cout_pg
            )

    out = int32_dequant_to_fp16_s8(result, ascale)
    out = out.permute(0, 3, 1, 2)

    return out


def convt_f16s8(
    x,
    w,
    bias,
    antiquant_scale,
    pads,
    strides,
    dilations,
    groups,
    output_shape,
    scale=13,
):
    sv = scale
    N, Cin, H_in, W_in = x.shape
    cin_w, cout_pg, kh, kw = w.shape
    cout = cout_pg * groups
    ascale = _normalize_ascale(antiquant_scale, cout)
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
    col_c = _ceil_div(cin_pg, 16) * 16

    result = torch.zeros(N, H_out, W_out, cout, dtype=torch.int32, device=x.device)

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
            col = _im2col_convt(up, kh, kw, dil_h, dil_w, H_out, W_out)
            wkmat = _convt_weight_to_kmat(w_g)
            acc0 = None

            if bias is not None:
                ascale_g = ascale[g * cout_pg : (g + 1) * cout_pg]
                bias_int = s8_bias_to_int32(
                    bias[g * cout_pg : (g + 1) * cout_pg], ascale_g
                )
                acc0 = bias_int.unsqueeze(0).expand(H_out * W_out, cout_pg)

            acc = f16xs8_gemm(col, wkmat, sv, acc0=acc0)
            result[n, :, :, g * cout_pg : (g + 1) * cout_pg] = acc.reshape(
                H_out, W_out, cout_pg
            )

    out = int32_dequant_to_fp16_s8(result, ascale)
    out = out.permute(0, 3, 1, 2)
    return out
