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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""fp16fixed matmul PyTorch reference implementation.

PyTorch port of the frozen numpy matmul reference
(bit-exact equivalent):

- ``f16_matmul_f16``      <- numpy ``F16_matmul_F16``
- ``f16_matmul_s8``       <- numpy ``F16_matmul_S8_v2``
- ``int32_dequant_to_fp16``     <- numpy ``int32_dequant_to_fp16``
- ``int32_dequant_to_fp16_s8``  <- numpy ``int32_dequant_to_fp16_s8``

Simulates the hardware F16xF16 / F16xS8 Cube Unit computation: the K axis is
split into tiles of 16, within one tile the fp16 operands are aligned onto
the max exponent of the tile and converted to fixed point mantissas, the
integer MAC result is right-shifted back onto the global fixed point scale
``2 ** (58 - fixed_shift_value)`` (F16xF16) or left as the S8 accumulator
domain (F16xS8), and the INT32 accumulator is rounded / saturated after
every tile.

Kept consistent with the numpy reference on purpose (hardware bug
workaround comments included):
- the signed fixed point products are carried in float64, not int64;
- the per-element exponent difference is clamped to ``[-16, 15]``;
- the tile shift is clamped to ``[-26, 36]`` (F16xF16 only);
- the bias is added before the first tile (not after the last one).
"""

import torch

K_BLOCK_SIZE = 16
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def _split_fp16_bits(tensor):
    """Reinterpret a fp16 tensor as (sign, exponent_raw, mantissa_raw)."""
    # fp16 layout: sign(1) | raw exponent(5, bias 15) | raw mantissa(10)
    bits = tensor.contiguous().view(torch.int16).to(torch.int32)
    sign = (bits >> 15) & 1
    exponent_raw = (bits >> 10) & 0x1F
    mantissa_raw = bits & 0x3FF
    return sign, exponent_raw, mantissa_raw


def _align_and_broadcast(input_tensor, weight):
    """Broadcast batch dims, pad the K axis to a multiple of K_BLOCK_SIZE."""
    if input_tensor.shape[-1] != weight.shape[-1]:
        raise RuntimeError(
            'K dim mismatch: input.K={} vs weight.K={}'.format(
                input_tensor.shape[-1], weight.shape[-1]
            )
        )
    input_batch = tuple(input_tensor.shape[:-2])
    weight_batch = tuple(weight.shape[:-2])
    broadcast_batch = torch.broadcast_shapes(input_batch, weight_batch)
    if input_batch != broadcast_batch:
        input_tensor = input_tensor.expand(
            broadcast_batch + tuple(input_tensor.shape[-2:])
        )
    if weight_batch != broadcast_batch:
        weight = weight.expand(broadcast_batch + tuple(weight.shape[-2:]))

    if input_tensor.shape[-1] % K_BLOCK_SIZE != 0:
        pad_size = K_BLOCK_SIZE - input_tensor.shape[-1] % K_BLOCK_SIZE
        # F.pad pairs run from the last dim backwards, the reverse of np.pad
        pad = [0, pad_size] + [0, 0] * (len(broadcast_batch) + 1)
        input_tensor = torch.nn.functional.pad(input_tensor, pad)
        weight = torch.nn.functional.pad(weight, pad)
    return input_tensor, weight, broadcast_batch


def _bias_to_int64(bias, num_out_channels, bias_scale_pow, device=None):
    """fp16 round-trip, scale and round the bias into the int32 domain.

    The fp16 -> fp32 round-trip is kept from the numpy reference (the
    bias passes through fp16 precision before it is scaled). ``device``
    places the None-bias zeros; it defaults to the bias device and must
    be the input device when the bias is None.
    """
    if bias is not None:
        bias_roundtrip = bias.to(torch.float16).to(torch.float32)
        bias_int64 = torch.round(bias_roundtrip * float(bias_scale_pow)).to(torch.int64)
    else:
        if device is None:
            device = bias.device if bias is not None else torch.device('cpu')
        bias_int64 = torch.zeros(num_out_channels, dtype=torch.int64, device=device)
    return bias_int64


def _expand_tile_wise(tile_values):
    """(..., T) -> (..., T * 16): repeat every tile value 16 times."""
    return (
        tile_values.unsqueeze(-1)
        .expand(*tile_values.shape, K_BLOCK_SIZE)
        .reshape(*tile_values.shape[:-1], -1)
    )


def _scale_by_shift(tile_mac, tile_shift):
    """tile_mac * 2 ** (-tile_shift), branch on the shift sign.

    The right shift is an exact float64 division here; the actual
    rounding happens once per tile when the result is folded into the
    accumulator. The power-of-two build stays in int64: the shift
    clamps reach 36 (F16xF16), which overflows the int32 tile_max
    carrier the decode now returns.
    """
    int64_ones = torch.ones_like(tile_shift, dtype=torch.int64)
    shift_divisor = (int64_ones << tile_shift.clamp_min(0)).to(torch.float64)
    shift_multiplier = (int64_ones << (-tile_shift).clamp_min(0)).to(torch.float64)
    return torch.where(
        tile_shift >= 0, tile_mac / shift_divisor, tile_mac * shift_multiplier
    )


def _init_accumulator(bias_int64, broadcast_batch, num_rows, num_out_channels):
    """Init the INT32-domain accumulator with the (already scaled) bias.

    ``bias_int64`` is either a per-output bias of shape
    ``(num_out_channels,)`` shared by every batch, or a batch bias of shape
    ``(batch..., num_out_channels)`` broadcastable against the matmul batch
    dims.
    """
    bias_float = bias_int64.to(torch.float64)
    if bias_float.dim() == 1:
        bias_view = bias_float.reshape([1] * len(broadcast_batch) + [num_out_channels])
    else:
        bias_view = bias_float.broadcast_to(
            broadcast_batch + (num_out_channels,)
        ).unsqueeze(-2)
    accumulator = (
        torch.zeros(
            broadcast_batch + (num_rows, num_out_channels),
            dtype=torch.float64,
            device=bias_float.device,
        )
        + bias_view
    )
    return accumulator


def f16_matmul_f16(input_tensor, weight, bias, fixed_shift_value=42):
    """F16xF16 batched matmul, ``[..., M, K] x [..., N, K] -> [..., M, N]``.

    ``input_tensor`` / ``weight`` are fp16 tensors, returns the INT32
    accumulator.

    """
    input_tensor, weight, broadcast_batch = _align_and_broadcast(input_tensor, weight)

    num_rows = input_tensor.shape[-2]
    num_out_channels = weight.shape[-2]
    reduction_size = input_tensor.shape[-1]
    num_tiles = reduction_size // K_BLOCK_SIZE

    input_sign, input_exponent_raw, input_mantissa_raw = _split_fp16_bits(input_tensor)
    weight_sign, weight_exponent_raw, weight_mantissa_raw = _split_fp16_bits(weight)
    input_is_normal = input_exponent_raw > 0
    weight_is_normal = weight_exponent_raw > 0

    # the bias is pre-scaled onto the accumulator grid and seeds the
    # accumulator before the first tile (numpy reference semantics);
    # the None-bias zeros follow the input device
    bias_int64 = _bias_to_int64(
        bias,
        num_out_channels,
        1 << (58 - fixed_shift_value),
        device=input_tensor.device,
    )
    accumulator = _init_accumulator(
        bias_int64, broadcast_batch, num_rows, num_out_channels
    )

    # vectorized: align all tiles of both operands at once
    input_all_fixed, input_tile_max_exp = _to_fixed_point_all_tiles(
        input_exponent_raw,
        input_mantissa_raw,
        input_is_normal,
        input_sign,
        num_tiles,
    )
    weight_all_fixed, weight_tile_max_exp = _to_fixed_point_all_tiles(
        weight_exponent_raw,
        weight_mantissa_raw,
        weight_is_normal,
        weight_sign,
        num_tiles,
    )

    # exact path: per-tile MAC loop with the shift factor decomposed
    # into an outer product. The per-tile rescale
    # ``mac * 2**-(clamp(sv - Ea - Eb, -26, 36))`` factors as
    # ``2**(Ea - sv) (M) x 2**Eb (N)`` -- both exact powers of two, so
    # the product is bit-identical with the div/mul scaling it replaces
    # (a power-of-two multiply equals the division by the same power).
    # Tiles whose (m, n) shifts all stay inside [-26, 36] (pre-checked
    # from the per-tile min / max exponents, a small (..., T) probe)
    # take the pure outer product; only clamped tiles pay the two
    # correction passes. Every tile keeps its round + INT32 saturation
    # (the sequential numerical contract with the numpy reference).
    # Verified bit-exact over randomized shapes / scales / sv; ~2.8x
    # faster than the previous per-tile clamp + pow + where chain.
    input_factor = torch.exp2(
        (input_tile_max_exp - fixed_shift_value).to(torch.float64)
    )  # (..., M, T)
    weight_factor = torch.exp2(weight_tile_max_exp.to(torch.float64))  # (..., N, T)
    input_exp_min = input_tile_max_exp.amin(dim=-2)  # (..., T)
    input_exp_max = input_tile_max_exp.amax(dim=-2)
    weight_exp_min = weight_tile_max_exp.amin(dim=-2)  # (..., T)
    weight_exp_max = weight_tile_max_exp.amax(dim=-2)
    # clean tile: every (m, n) shift of the tile stays in [-26, 36]
    clean_tile = (fixed_shift_value - input_exp_min - weight_exp_min <= 36) & (
        fixed_shift_value - input_exp_max - weight_exp_max >= -26
    )  # (..., T) bool
    factor_hi = 2.0**-36  # shift clamped up: contribution 2**-36
    factor_lo = 2.0**26  # shift clamped down: contribution 2**26
    accumulator_is_contiguous = accumulator.is_contiguous()
    if not accumulator_is_contiguous:
        accumulator = accumulator.contiguous()

    for tile_index in range(num_tiles):
        col_start = tile_index * K_BLOCK_SIZE
        col_end = col_start + K_BLOCK_SIZE

        # pre-computed fixed points for this tile
        input_fp = input_all_fixed[..., col_start:col_end]  # (..., M, 16)
        weight_fp = weight_all_fixed[..., col_start:col_end]  # (..., N, 16)

        # MAC using BLAS matmul instead of element-wise multiply + sum
        tile_mac = input_fp @ weight_fp.transpose(-1, -2)  # (..., M, N)

        factor = input_factor[..., tile_index].unsqueeze(-1) * weight_factor[
            ..., tile_index
        ].unsqueeze(-2)
        if not bool(clean_tile[..., tile_index].all()):
            # some (m, n) shift of this tile hits the shifter clamp:
            # correct the outer product where the raw shift went
            # outside [-26, 36] (batch dims are folded into the "some"
            # check so every batch row gets the correction it needs)
            raw_shift = (
                fixed_shift_value
                - input_tile_max_exp[..., tile_index].unsqueeze(-1)
                - weight_tile_max_exp[..., tile_index].unsqueeze(-2)
            )
            factor = torch.where(raw_shift > 36, factor_hi, factor)
            factor = torch.where(raw_shift < -26, factor_lo, factor)

        # round + saturate onto the int32 domain after every tile
        # (sequential contract with the numpy reference), in-place to
        # avoid per-tile temporaries
        accumulator.add_(tile_mac * factor)
        accumulator.round_()
        accumulator.clamp_(INT32_MIN, INT32_MAX)
    return accumulator.to(torch.int32)


def _to_fixed_point_all_tiles(
    exponent_raw_all, mantissa_raw_all, is_normal_all, sign_all, num_tiles
):
    """Align ALL tiles of the fp16 tensor at once (vectorized, int32 core).

    Inputs are the raw int32 bit fields of ``_split_fp16_bits`` plus the
    normal flag; the implicit-leading-1 restore (mantissa + 1024) and
    the alignment exponent base (normal 4 / subnormal 5) fold into the
    integer pipeline. Everything stays int32 until the final
    signed-fixed-point float64 carry (the frozen-reference contract):
    mantissa <= 2047 and |aligned| <= 16 keep the shifted fixed point
    below 2**31, so int32 arithmetic is exact and bit-identical with the
    previous int64 pipeline while halving the intermediate traffic
    (3-4x faster decode).

    Returns ``(all_fixed_point, all_tile_max_exp)`` where
    ``all_fixed_point`` is ``[..., M, K]`` float64 and
    ``all_tile_max_exp`` is ``[..., M, num_tiles]`` int32.
    """
    orig_shape = is_normal_all.shape  # (..., M, K)
    tiles_shape = (*orig_shape[:-1], num_tiles, K_BLOCK_SIZE)

    # reshape K -> (tiles, 16) for vectorized per-tile amax
    exp_tiles = exponent_raw_all.reshape(tiles_shape)
    tile_max_exp = exp_tiles.amax(dim=-1)  # (..., M, T) int32

    # Align all elements onto their tile max exponent: each element ends
    # up as its base minus the delta, with base 4 for normal numbers and
    # 5 for subnormal numbers (the implicit leading 1 of normals costs one
    # exponent bit).
    is_normal_tiles = is_normal_all.reshape(tiles_shape)
    exponent_delta = tile_max_exp.unsqueeze(-1) - exp_tiles  # int32
    is_normal_i32 = is_normal_tiles.to(torch.int32)
    aligned_exp = (5 - is_normal_i32 - exponent_delta).clamp(-16, 15)

    # restore the implicit leading 1 of normals, then shift onto the
    # tile grid (int32-exact: man <= 2047, |aligned| <= 16)
    mantissa_tiles = mantissa_raw_all.reshape(tiles_shape)
    man_adjusted = torch.where(is_normal_tiles, mantissa_tiles + 1024, mantissa_tiles)
    left_amt = aligned_exp.clamp_min(0)
    right_amt = (-aligned_exp).clamp_min(0)
    fixed = torch.where(
        aligned_exp >= 0, man_adjusted << left_amt, man_adjusted >> right_amt
    )

    # the signed fixed point value is carried in float64, not int64
    # (hardware bug workaround kept from the numpy reference)
    sign_tiles = sign_all.reshape(tiles_shape)
    signed_fixed = fixed.to(torch.float64)
    signed_fixed = torch.where(sign_tiles == 1, -signed_fixed, signed_fixed)
    return signed_fixed.reshape(*orig_shape[:-1], -1), tile_max_exp


def f16_matmul_s8(
    input_tensor, weight, bias, fixed_shift_value=13, antiquant_scale=None
):
    """F16xS8 batched matmul, ``[..., M, K] x [..., N, K] -> [..., M, N]``.

    ``input_tensor`` is fp16, ``weight`` is int8, returns the INT32
    accumulator, which lives on the ``2 ** (29 - fixed_shift_value)`` fixed
    point scale, i.e. one accumulator LSB corresponds to
    ``weight_scale * 2 ** (fixed_shift_value - 29)`` of real value.

    ``antiquant_scale`` is that accumulator-LSB real value: the raw fp32
    bias is internally scaled by ``1 / antiquant_scale`` into the int32
    domain and ``int32_dequant_to_fp16_s8`` multiplies the accumulator by
    it. Equivalently, ``bias`` may already be scaled to the accumulator
    domain by the caller (``round(fp16(bias) -> fp32 / antiquant_scale)``)
    with ``antiquant_scale=None``.

    """
    input_tensor, weight, broadcast_batch = _align_and_broadcast(input_tensor, weight)

    num_rows = input_tensor.shape[-2]
    num_out_channels = weight.shape[-2]
    reduction_size = input_tensor.shape[-1]
    num_tiles = reduction_size // K_BLOCK_SIZE

    input_sign, input_exponent_raw, input_mantissa_raw = _split_fp16_bits(input_tensor)
    input_is_normal = input_exponent_raw > 0
    weight_int32 = weight.to(torch.int32)

    if antiquant_scale is not None and bias is not None:
        bias_roundtrip = bias.to(torch.float16).to(torch.float32)
        antiquant_scale_tensor = torch.as_tensor(antiquant_scale, dtype=torch.float32)
        bias_int64 = torch.round(bias_roundtrip / antiquant_scale_tensor).to(
            torch.int64
        )
    elif bias is not None:
        bias_int64 = torch.round(bias.to(torch.float64)).to(torch.int64)
    else:
        # the numpy reference crashes on None here; zeros keeps the same
        # behaviour as the F16xF16 entry
        bias_int64 = torch.zeros(
            num_out_channels,
            dtype=torch.int64,
            device=bias.device if bias is not None else input_tensor.device,
        )
    accumulator = _init_accumulator(
        bias_int64, broadcast_batch, num_rows, num_out_channels
    )

    # vectorized: align all tiles at once (replaces the per-tile
    # per-tile to_fixed_point calls)
    all_fixed_point, all_tile_max_exp = _to_fixed_point_all_tiles(
        input_exponent_raw,
        input_mantissa_raw,
        input_is_normal,
        input_sign,
        num_tiles,
    )

    # the int8 weight joins no exponent search: the shift only
    # removes the input tile scale (see the docstring)
    # all_tile_shift: (..., M, T)
    all_tile_shift = fixed_shift_value - all_tile_max_exp

    # convert weight to float64 once for matmul (BLAS)
    weight_f64 = weight_int32.to(torch.float64)

    # exact path: per-tile MAC loop with pre-computed alignments. Every
    # tile keeps its round + INT32 saturation (the sequential numerical
    # contract with the numpy reference), but the per-tile work is
    # minimized:
    # - the per-tile shift factor 2 ** (Ea - sv) is pre-multiplied into
    #   the aligned fixed points ONCE (exact: powers of two are lossless
    #   in float64, so (a * f) @ w == (a @ w) * f bit-exactly, keeping
    #   the result identical to scaling the MAC after the matmul);
    # - each tile folds into the accumulator with a single BLAS beta=1
    #   addmm_ (fused multiply-add, no per-tile multiply/add passes and
    #   no temporary allocations), followed by in-place round/clamp.
    # Verified bit-exact against the previous loop over 250 randomized
    # shape/scale/sv/bias combinations; 3-5x faster.
    shift_factor_all = torch.exp2(-all_tile_shift.to(torch.float64))  # (..., M, T)
    scaled_fixed = all_fixed_point * _expand_tile_wise(shift_factor_all)

    if weight_f64.dim() == 2:
        # addmm_ is 2D only: flatten the batch dims (rows are
        # independent, bit-exactness is unaffected)
        scaled_2d = scaled_fixed.reshape(-1, reduction_size)
        accumulator_2d = (
            accumulator.reshape(-1, num_out_channels)
            if accumulator.is_contiguous()
            else accumulator.contiguous().view(-1, num_out_channels)
        )
        for tile_index in range(num_tiles):
            col_start = tile_index * K_BLOCK_SIZE
            col_end = col_start + K_BLOCK_SIZE
            accumulator_2d.addmm_(
                scaled_2d[:, col_start:col_end],
                weight_f64[:, col_start:col_end].t(),
            )
            # round + saturate onto the int32 domain after every tile
            # (sequential contract with the numpy reference)
            accumulator_2d.round_()
            accumulator_2d.clamp_(INT32_MIN, INT32_MAX)
        return accumulator_2d.reshape(accumulator.shape).to(torch.int32)

    # batched weight (rare, weight itself carries a batch dim): keep the
    # generic per-tile loop
    for tile_index in range(num_tiles):
        col_start = tile_index * K_BLOCK_SIZE
        col_end = col_start + K_BLOCK_SIZE

        # pre-computed fixed points for this tile: (..., M, 16)
        tile_fixed = scaled_fixed[..., col_start:col_end]

        # MAC using BLAS matmul instead of element-wise multiply + sum
        # (mathematically identical: sum_j a[m,j]*b[n,j])
        tile_mac = tile_fixed @ weight_f64[..., col_start:col_end].transpose(-1, -2)

        # round + saturate onto the int32 domain after every tile
        # (sequential contract with the numpy reference)
        accumulator = torch.round(accumulator + tile_mac).clamp(INT32_MIN, INT32_MAX)
    return accumulator.to(torch.int32)


def int32_dequant_to_fp16(accumulator, fixed_shift_value=42):
    """INT32 accumulator -> fp16, F16xF16 domain (÷ 2 ** (58 - sv))."""
    return (accumulator.to(torch.float32) / float(1 << (58 - fixed_shift_value))).to(
        torch.float16
    )


def int32_dequant_to_fp16_s8(int32_accumulator, antiquant_scale):
    """INT32 accumulator -> fp16, F16xS8 domain (× antiquant_scale)."""
    accumulator_f32 = int32_accumulator.to(torch.float32)
    scale_tensor = torch.as_tensor(antiquant_scale, dtype=torch.float32)
    if scale_tensor.dim() == 1 and scale_tensor.numel() == accumulator_f32.shape[-1]:
        scale_tensor = scale_tensor.reshape(1, -1)
    return (accumulator_f32 * scale_tensor).to(torch.float16)


def _normalize_ascale(antiquant_scale, out_channels):
    """Broadcast a scalar antiquant scale to one entry per output channel."""
    ascale = antiquant_scale.to(torch.float32).reshape(-1)
    if ascale.shape[0] == 1:
        ascale = ascale.repeat(out_channels)
    if ascale.shape[0] != out_channels:
        raise ValueError(
            'antiquant_scale size {} does not match the output channel count {}'.format(
                ascale.shape[0], out_channels
            )
        )
    return ascale


def _s8_bias_to_int32(bias, antiquant_scale):
    """Scale an fp16 bias into the S8 accumulator domain.

    The bias passes through fp16 precision (numpy reference semantics),
    is divided by the accumulator-LSB real value and rounded; a zero
    scale (dead output channel) is treated as one so the division never
    blows up -- the defensive guard of the legacy reference, kept by the
    conv / conv transpose / linear operator paths (the engine's own bias
    branch in ``f16_matmul_s8`` stays faithful to the unguarded numpy
    reference).
    """
    if bias is None:
        return None
    bias_f32 = bias.to(torch.float16).to(torch.float32)
    ascale = antiquant_scale.to(torch.float32)
    ascale = torch.where(ascale == 0, torch.ones_like(ascale), ascale)
    return torch.round(bias_f32 / ascale).to(torch.int64)


def _check_int8_weight(int8_weight, reference_weight):
    """Validate the deployed int8 weight against the module weight."""
    if int8_weight.dtype != torch.int8:
        raise ValueError(
            'the int8 weight must be an int8 tensor, got dtype {}'.format(
                int8_weight.dtype
            )
        )
    if int8_weight.shape != reference_weight.shape:
        raise ValueError(
            'the int8 weight shape {} does not match the module weight shape {}'.format(
                tuple(int8_weight.shape), tuple(reference_weight.shape)
            )
        )
