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


def F16_matmul_F16(a, b, bias, fixed_shift_val=42):
    """
    f16_matmul_f16 vectorized + multi-batch optimized version
    Simulates hardware F16xF16 Cube Unit computation with per-tile
    """
    BaseK = 16
    scale_6bit = fixed_shift_val
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max

    # step1: broadcast batch dimensions
    *batch_a, M_in, K_a = a.shape
    *batch_b, N_in, K_b = b.shape
    assert K_a == K_b, f"K dim mismatch: a.K={K_a} vs b.K={K_b}"

    broadcast_batch = np.broadcast_shapes(batch_a, batch_b)

    if tuple(batch_a) != broadcast_batch:
        a = np.broadcast_to(a, broadcast_batch + (M_in, K_a))
    if tuple(batch_b) != broadcast_batch:
        b = np.broadcast_to(b, broadcast_batch + (N_in, K_b))

    K = K_a
    if K % BaseK != 0:
        pad_k = BaseK - K % BaseK
        pad_a = [(0, 0)] * len(broadcast_batch) + [(0, 0), (0, pad_k)]
        pad_b = [(0, 0)] * len(broadcast_batch) + [(0, 0), (0, pad_k)]

        a = np.pad(a, pad_a, mode='constant', constant_values=0)
        b = np.pad(b, pad_b, mode='constant', constant_values=0)
        K = a.shape[-1]

    M = a.shape[-2]
    N = b.shape[-2]

    a_u16 = a.view(np.uint16)
    b_u16 = b.view(np.uint16)

    a_sign_mask = (a_u16 >> 15) & 1
    b_sign_mask = (b_u16 >> 15) & 1
    a_exp_raw = (a_u16 >> 10) & 0x1F
    b_exp_raw = (b_u16 >> 10) & 0x1F
    a_man_raw = a_u16 & 0x3FF
    b_man_raw = b_u16 & 0x3FF

    a_hw_exp = a_exp_raw.astype(np.int32)
    b_hw_exp = b_exp_raw.astype(np.int32)

    a_is_normal = a_exp_raw > 0
    b_is_normal = b_exp_raw > 0

    a_exp_base = np.where(a_is_normal, np.int32(4), np.int32(5))
    b_exp_base = np.where(b_is_normal, np.int32(4), np.int32(5))

    a_man_val = np.where(
        a_is_normal, a_man_raw.astype(np.int32) + 1024, a_man_raw.astype(np.int32)
    )
    b_man_val = np.where(
        b_is_normal, b_man_raw.astype(np.int32) + 1024, b_man_raw.astype(np.int32)
    )

    a_sign = np.where(a_sign_mask == 1, np.int32(-1), np.int32(1))
    b_sign = np.where(b_sign_mask == 1, np.int32(-1), np.int32(1))

    # step 4 per-tile computation
    num_tiles = K // BaseK

    C = np.zeros(broadcast_batch + (M, N), dtype=np.int32)

    if bias is not None:
        bias = bias.astype(np.float16).astype(np.float32)
        bias = bias * (1 << (58 - scale_6bit))
        bias_arr = np.round(bias).astype(np.int64)
    else:
        bias_arr = np.zeros((N,), dtype=np.int64)
    bias_f2 = bias_arr.reshape(-1, N) if bias_arr.ndim > 1 else bias_arr.reshape(1, N)

    bc_prod = np.prod(broadcast_batch, dtype=int) if broadcast_batch else 1
    bias_bc2 = np.broadcast_to(bias_f2, (bc_prod, N))
    bias_4d = bias_bc2.reshape(broadcast_batch + (N,) if broadcast_batch else (N,))
    C = C + bias_4d.reshape(broadcast_batch + (1, N) if broadcast_batch else (1, N))

    for t in range(num_tiles):
        ks = t * BaseK
        ke = ks + BaseK

        A_t_exp = a_hw_exp[..., ks:ke]
        B_t_exp = b_hw_exp[..., ks:ke]

        A_max = np.max(A_t_exp, axis=-1)
        B_max = np.max(B_t_exp, axis=-1)

        shift_raw = scale_6bit - np.expand_dims(A_max, -1) - np.expand_dims(B_max, -2)
        shift = np.clip(shift_raw, -26, 36).astype(np.int64)

        A_delta = np.expand_dims(A_max, -1) - A_t_exp
        A_exp = np.clip(a_exp_base[..., ks:ke] - A_delta, -16, 15)

        A_man = a_man_val[..., ks:ke].astype(np.int64)
        A_fix_pos = A_man << A_exp  # exp>=0: left shift
        A_fix_neg = A_man >> (-A_exp)  # exp<0: right shift
        A_fixed = np.where(A_exp >= 0, A_fix_pos, A_fix_neg)

        # ai bug 用高精度浮点，不是用int64
        A_signed = A_fixed * a_sign[..., ks:ke].astype(np.float64)

        B_delta = np.expand_dims(B_max, -1) - B_t_exp
        B_exp = np.clip(b_exp_base[..., ks:ke] - B_delta, -16, 15)

        B_man = b_man_val[..., ks:ke].astype(np.int64)
        B_fix_pos = B_man << B_exp  # exp>=0: left shift
        B_fix_neg = B_man >> (-B_exp)  # exp<0: right shift
        B_fixed = np.where(B_exp >= 0, B_fix_pos, B_fix_neg)

        # ai bug 用高精度浮点，不是用int64
        B_signed = B_fixed * b_sign[..., ks:ke].astype(np.float64)

        mul_35 = np.sum(
            np.expand_dims(A_signed, -2) * np.expand_dims(B_signed, -3), axis=-1
        )

        mul_scaled = np.where(
            shift >= 0,
            mul_35.astype(np.float64) / (np.int64(1) << shift),
            mul_35.astype(np.float64) * (np.int64(1) << (-shift)),
        ).astype(np.float64)

        C = np.clip(np.round(C.astype(np.float64) + mul_scaled), int32_min, int32_max)

    return C.astype(np.int32)


def F16_matmul_S8_v2(a, b, bias, fixed_shift_val=13, antiquant_scale=None):
    """F16 x S8 batched matmul
    bias should already be scaled to the accumulator domain by the caller
    (i.e. bias_int32 = round (fp16(bias) -> fp32 / antiquant_scale)).
    If antiquant_scale is provided and bias is the raw fp32 bias, this
    function performs the scaling internally for convenience.
    """
    BaseK = 16
    scale_6bit = fixed_shift_val
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max

    *batch_a, M_in, K_a = a.shape
    *batch_b, N_in, K_b = b.shape
    assert K_a == K_b, f"K dim mismatch: a.K={K_a} vs b.K={K_b}"

    broadcast_batch = np.broadcast_shapes(batch_a, batch_b)

    if tuple(batch_a) != broadcast_batch:
        a = np.broadcast_to(a, broadcast_batch + (M_in, K_a))
    if tuple(batch_b) != broadcast_batch:
        b = np.broadcast_to(b, broadcast_batch + (N_in, K_b))

    K = K_a
    if K % BaseK != 0:
        pad_k = BaseK - K % BaseK
        pad_a = [(0, 0)] * len(broadcast_batch) + [(0, 0), (0, pad_k)]
        pad_b = [(0, 0)] * len(broadcast_batch) + [(0, 0), (0, pad_k)]

        a = np.pad(a, pad_a, mode="constant", constant_values=0)
        b = np.pad(b, pad_b, mode="constant", constant_values=0)
        K = a.shape[-1]

    M = a.shape[-2]
    N = b.shape[-2]

    a_u16 = a.view(np.uint16)
    a_sign_mask = (a_u16 >> 15) & 1
    a_exp_raw = (a_u16 >> 10) & 0x1F
    a_man_raw = a_u16 & 0x3FF

    a_hw_exp = a_exp_raw.astype(np.int32)

    a_is_normal = a_exp_raw > 0

    a_exp_base = np.where(a_is_normal, np.int32(4), np.int32(5))

    a_man_val = np.where(
        a_is_normal, a_man_raw.astype(np.int32) + 1024, a_man_raw.astype(np.int32)
    )

    a_sign = np.where(a_sign_mask == 1, np.int32(-1), np.int32(1))
    b_val = b.astype(np.int32)
    # step 4 per-tile computation
    num_tiles = K // BaseK

    C = np.zeros(broadcast_batch + (M, N), dtype=np.int32)

    if antiquant_scale is not None:
        bias = np.asarray(bias, dtype=np.float32)
        bias = bias.astype(np.float16).astype(np.float32)
        ascale = np.asarray(antiquant_scale, dtype=np.float32)
        bias_scaled = bias / ascale
        bias_arr = np.round(bias_scaled).astype(np.int64)
    else:
        bias_arr = np.asarray(bias, dtype=np.float64)
        bias_arr = np.round(bias_arr).astype(np.int64)

    bias_f2 = bias_arr.reshape(-1, N) if bias_arr.ndim > 1 else bias_arr.reshape(1, N)
    bc_prod = np.prod(broadcast_batch, dtype=int) if broadcast_batch else 1
    bias_bc2 = np.broadcast_to(bias_f2, (bc_prod, N))
    bias_4d = bias_bc2.reshape(broadcast_batch + (N,) if broadcast_batch else (N,))
    C = C + bias_4d.reshape(broadcast_batch + (1, N) if broadcast_batch else (1, N))

    for t in range(num_tiles):
        ks = t * BaseK
        ke = ks + BaseK

        A_t_exp = a_hw_exp[..., ks:ke]

        A_max = np.max(A_t_exp, axis=-1)

        shift_raw = scale_6bit - A_max

        shift = np.expand_dims(shift_raw.astype(np.int64), -1)

        A_delta = np.expand_dims(A_max, -1) - A_t_exp
        A_exp = np.clip(a_exp_base[..., ks:ke] - A_delta, -16, 15)

        A_man = a_man_val[..., ks:ke].astype(np.int64)
        A_fix_pos = A_man << A_exp  # exp>=0: left shift
        A_fix_neg = A_man >> (-A_exp)  # exp<0: right shift
        A_fixed = np.where(A_exp >= 0, A_fix_pos, A_fix_neg)

        # ai bug 用高精度浮点，不是用int64
        A_signed = A_fixed * a_sign[..., ks:ke].astype(np.float64)

        B_signed = b_val[..., ks:ke].astype(np.float64)

        mul_28 = np.sum(
            np.expand_dims(A_signed, -2) * np.expand_dims(B_signed, -3), axis=-1
        ).astype(np.float64)

        mul_scaled = np.where(
            shift >= 0,
            mul_28 / (np.int64(1) << shift),
            mul_28 * (np.int64(1) << (-shift)),
        )

        C = np.clip(np.round(C.astype(np.float64) + mul_scaled), int32_min, int32_max)

    return C.astype(np.int32)


def int32_dequant_to_fp16(inputs, fixed_shift_val=42):
    inputs = inputs.astype(np.float32) / (1 << (58 - fixed_shift_val))
    return inputs.astype(np.float16)


def int32_dequant_to_fp16_s8(int32_acc, antiquant_scale):
    int32_f = int32_acc.astype(np.float32)
    scale = np.asarray(antiquant_scale, dtype=np.float32)
    if scale.ndim == 1 and scale.size == int32_f.shape[-1]:
        scale = scale.reshape(1, -1)
    out = int32_f * scale
    return out.astype(np.float16)
