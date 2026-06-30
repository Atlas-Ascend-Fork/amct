"""
Pure-torch MXFP4 quantizer aligned with Quark's OCP MXFP4 export path.

This implementation matches the static weight export path:
1. Compute per-group scale with OCP even-rounding.
2. Quantize to FP4 e2m1 with round-half-to-even.
3. Pack 2 FP4 values into one uint8.
4. Store scale in e8m0 uint8 format.
"""

from __future__ import annotations

import torch

BLOCK_SIZE = 32

E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _right_shift_unsigned(x: torch.Tensor, shift: int) -> torch.Tensor:
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def _compute_scale_even(max_abs: torch.Tensor) -> torch.Tensor:
    f32_min_normal = 2.0 ** (-126)
    eps = f32_min_normal * (max_abs == 0).to(max_abs.dtype)

    nan_mask = torch.isnan(max_abs)
    max_abs_int = max_abs.to(torch.float32).view(torch.int32)

    # fp4 e2m1 => mbits=1
    val_to_add = 1 << 21
    fp32_sign_exponent_mask = ((1 << 9) - 1) << 23

    rounded = (max_abs_int + val_to_add) & fp32_sign_exponent_mask
    rounded_float = rounded.view(torch.float32)
    rounded_float = rounded_float.masked_fill(nan_mask, float("nan"))

    scale_e8m0_unbiased = torch.floor(torch.log2(rounded_float + eps)) - 2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-127, max=127)
    return torch.pow(2.0, scale_e8m0_unbiased)


def scale_float_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    return (torch.log2(scale).round().to(torch.int16).clamp(-127, 127) + 127).to(
        torch.uint8
    )


def scale_e8m0_to_float(scale_uint8: torch.Tensor) -> torch.Tensor:
    return (scale_uint8.to(torch.int32) << 23).view(torch.float32)


def _fp32_to_e2m1_packed(tensor: torch.Tensor) -> torch.Tensor:
    q_int = tensor.contiguous().view(torch.int32)
    signs = q_int & 0x80000000
    exponents = _right_shift_unsigned(q_int, 23) & 0xFF
    mantissas = q_int & 0x7FFFFF

    e8_bias = 127
    e2_bias = 1
    mantissas = torch.where(
        exponents < e8_bias,
        (0x400000 | _right_shift_unsigned(mantissas, 1))
        >> (e8_bias - exponents - 1).clamp(min=0),
        mantissas,
    )
    exponents = torch.maximum(
        exponents, torch.tensor(e8_bias - e2_bias, device=tensor.device)
    ) - (e8_bias - e2_bias)

    # Round-half-to-even to match Quark C++ kernel.
    combined = (exponents << 2) | _right_shift_unsigned(mantissas, 21)
    remainder_bit = combined & 1
    trailing_bits = mantissas & 0x1FFFFF
    is_odd_result = _right_shift_unsigned(combined, 1) & 1
    round_up = remainder_bit & torch.where(
        trailing_bits > 0, torch.ones_like(combined), is_odd_result
    )
    e2m1_tmp = _right_shift_unsigned(combined + round_up, 1)
    e2m1_tmp = torch.minimum(e2m1_tmp, torch.tensor(0x7, device=tensor.device))

    sign_bits = _right_shift_unsigned(signs, 28)
    sign_bits = torch.where(e2m1_tmp == 0, torch.zeros_like(sign_bits), sign_bits)
    e2m1_value = (sign_bits | e2m1_tmp).to(torch.uint8)

    axis_shape = e2m1_value.shape[-1]
    e2m1_value = e2m1_value.view(*e2m1_value.shape[:-1], axis_shape // 2, 2)
    evens = e2m1_value[..., 0]
    odds = e2m1_value[..., 1]
    return evens | (odds << 4)


def _e2m1_packed_to_fp32(packed: torch.Tensor) -> torch.Tensor:
    lut = E2M1_LUT.to(packed.device)
    packed_int = packed.to(torch.int32)
    evens = packed_int & 0xF
    odds = (packed_int >> 4) & 0xF
    even_floats = lut[evens]
    odd_floats = lut[odds]
    output = torch.stack([even_floats, odd_floats], dim=-1)
    return output.view(*packed.shape[:-1], -1)


def mxfp4_quantize(
    tensor: torch.Tensor, axis: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    ndim = tensor.ndim
    axis = axis if axis >= 0 else axis + ndim

    perm = list(range(ndim))
    if axis != ndim - 1:
        perm[axis], perm[-1] = perm[-1], perm[axis]
    src = tensor.permute(perm).to(torch.float32)

    axis_shape = src.shape[-1]
    if axis_shape % 2 != 0:
        raise ValueError(
            "Quantization axis must have an even length for packed FP4 output."
        )

    next_multiple = (axis_shape + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    pad_amount = next_multiple - axis_shape
    if pad_amount > 0:
        padded = torch.nn.functional.pad(src, (0, pad_amount))
        valid_mask = torch.nn.functional.pad(
            torch.ones_like(src, dtype=torch.bool), (0, pad_amount)
        )
    else:
        padded = src
        valid_mask = torch.ones_like(src, dtype=torch.bool)

    abs_vals = padded.abs()
    abs_vals = torch.where(
        valid_mask,
        abs_vals,
        torch.tensor(-1.0, device=padded.device, dtype=padded.dtype),
    )
    groups_shape = padded.shape[:-1] + (padded.shape[-1] // BLOCK_SIZE, BLOCK_SIZE)
    grouped = abs_vals.view(*groups_shape)
    max_abs = grouped.max(dim=-1)[0]
    scale_float = _compute_scale_even(max_abs)
    scale_e8m0 = scale_float_to_e8m0(scale_float)

    quant_scale = torch.where(
        scale_float == 0, torch.tensor(0.0, device=padded.device), 1.0 / scale_float
    )
    quant = padded.view(*groups_shape) * quant_scale.unsqueeze(-1)
    quant = quant.view(padded.shape)[..., :axis_shape]

    packed = _fp32_to_e2m1_packed(quant)
    packed = packed.permute(perm)
    scale_e8m0 = scale_e8m0.permute(perm)
    return packed.contiguous(), scale_e8m0.contiguous()


def mxfp4_dequantize(
    packed: torch.Tensor,
    scale_e8m0: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
    axis: int = -1,
) -> torch.Tensor:
    ndim = packed.ndim
    axis = axis if axis >= 0 else axis + ndim

    perm = list(range(ndim))
    if axis != ndim - 1:
        perm[axis], perm[-1] = perm[-1], perm[axis]

    packed_p = packed.permute(perm)
    scale_p = scale_e8m0.permute(perm)

    dq_scale = scale_e8m0_to_float(scale_p)
    fp4_vals = _e2m1_packed_to_fp32(packed_p)

    axis_shape = fp4_vals.shape[-1]
    padded_axis_shape = dq_scale.shape[-1] * BLOCK_SIZE
    pad_size = padded_axis_shape - axis_shape
    if pad_size > 0:
        fp4_vals = torch.nn.functional.pad(fp4_vals, (0, pad_size))

    groups_shape = fp4_vals.shape[:-1] + (padded_axis_shape // BLOCK_SIZE, BLOCK_SIZE)
    dequant = fp4_vals.view(*groups_shape) * dq_scale.unsqueeze(-1)
    dequant = dequant.view(*fp4_vals.shape[:-1], padded_axis_shape)[..., :axis_shape]
    return dequant.permute(perm).to(target_dtype)
