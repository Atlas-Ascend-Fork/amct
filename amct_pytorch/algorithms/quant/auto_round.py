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

import math

import torch
import torch.nn as nn

from amct_pytorch.algorithms.quant.base import QuantAlgorithmBase
from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.quantization.dtypes.hifp_impl import NG, _to_bf16


def _ste_floor(x: torch.Tensor) -> torch.Tensor:
    return (torch.floor(x) - x).detach() + x


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return (torch.round(x) - x).detach() + x


def _ste_clamp(x: torch.Tensor, min_val, max_val) -> torch.Tensor:
    return (torch.clamp(x, min_val, max_val) - x).detach() + x


_K_MAX = 2.0
_V_MAX = 2.0 - 2.0 ** (-NG)


def _quant_hifp(x, qdim, k=None, v=None):
    """HiAutoRound HiF4 fake-quant aligned with the repo's canonical HiF4 grid.

    Ports hifp_impl._hif4_reference_quantize (same math as
    hifloat4_fake_quant/hif4_pack) but swaps its bit-exact, non-differentiable
    primitives for torch.log2/exp2 + STE floor/round so gradients reach the
    learnable `k` (per innermost 4-element group) and `v` (per element)
    compensation terms. Both are clamped (_K_MAX/_V_MAX) to stop unbounded
    drift under gradient descent once downstream clamps kill their gradient.
    """
    xg = x.to(torch.float32).unflatten(qdim, (-1, 8, 2, 4))
    special_mask = ~torch.isfinite(xg)
    x_finite = torch.where(special_mask, torch.zeros_like(xg), xg)
    x_unsigned = torch.abs(x_finite)
    sign = torch.sign(x_finite)

    max_lv3 = torch.amax(x_unsigned, dim=qdim, keepdim=True)
    max_lv2 = torch.amax(max_lv3, dim=qdim - 1, keepdim=True)
    max_lv1 = torch.amax(max_lv2, dim=qdim - 2, keepdim=True)
    zero_group = max_lv1 == 0

    div7 = _to_bf16(torch.ones_like(max_lv1) / 7.0)
    scale_factor = max_lv1 * div7
    safe_scale = torch.where(
        scale_factor > 0, scale_factor, torch.ones_like(scale_factor)
    )
    e_sf = _ste_floor(torch.log2(safe_scale))
    mant_sf = safe_scale * torch.exp2(-e_sf + 7.0)
    scale_factor = _ste_round(mant_sf) * torch.exp2(e_sf - 7.0)
    scale_factor = scale_factor.clamp(min=2.0 ** (-48), max=49152.0)

    e_sf = _ste_floor(torch.log2(scale_factor))
    scale_factor = _ste_round(scale_factor * torch.exp2(2.0 - e_sf)) * torch.exp2(
        e_sf - 2.0
    )
    scale_factor = torch.where(
        zero_group, torch.full_like(scale_factor, 2.0 ** (-48)), scale_factor
    )

    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x_unsigned.dtype)

    scale_lv2_exp = _ste_floor((max_lv2 * rec_sf).clamp(0, 4) * 0.25)
    scale_lv2 = torch.exp2(scale_lv2_exp)
    scale_lv3_input = max_lv3 * rec_sf * torch.exp2(-scale_lv2_exp)
    if k is not None:
        k = _ste_clamp(k, -_K_MAX, _K_MAX)
        scale_lv3_input = scale_lv3_input + k.reshape(scale_lv3_input.shape)
    scale_lv3_exp = _ste_floor(scale_lv3_input.clamp(0, 2) * 0.5)
    scale_lv3 = torch.exp2(scale_lv3_exp)

    mant = x_unsigned * rec_sf * torch.exp2(-scale_lv2_exp - scale_lv3_exp)
    mant = _to_bf16(mant)
    if v is not None:
        v = _ste_clamp(v, -_V_MAX, _V_MAX)
        mant = mant + v.reshape(mant.shape)
    mant = _ste_floor(mant * (2**NG) + 0.5) / (2**NG)
    mant = _ste_clamp(mant, -2.0 + 2.0 ** (-NG), 2.0 - 2.0 ** (-NG))

    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor

    group_nan = special_mask.any(dim=qdim, keepdim=True)
    group_nan = group_nan.any(dim=qdim - 1, keepdim=True)
    group_nan = group_nan.any(dim=qdim - 2, keepdim=True)
    out = torch.where(
        torch.broadcast_to(group_nan, out.shape), torch.full_like(out, torch.nan), out
    )

    return out.flatten(qdim - 3, qdim).to(x.dtype)


def _reshape_pad_tensor_by_group_size(
    tensor: torch.Tensor, group_size: int, pad_value: float = 0.0
):
    orig_shape = tensor.shape
    if tensor.ndim != 2:
        raise ValueError(
            f"AutoRound currently expects 2D weight tensors, got shape {orig_shape}"
        )

    if group_size in (0, -1) or tensor.shape[-1] <= group_size:
        return tensor.reshape(tensor.shape[0], -1), orig_shape, 0

    pad_len = (group_size - tensor.shape[-1] % group_size) % group_size
    if pad_len > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_len), value=pad_value)
    return tensor.reshape(-1, group_size), orig_shape, pad_len


def _revert_tensor_by_pad(tensor: torch.Tensor, orig_shape, pad_len: int):
    if len(orig_shape) != 2:
        raise ValueError(
            f"AutoRound currently expects 2D weight tensors, got shape {orig_shape}"
        )
    if pad_len > 0:
        rows, cols = orig_shape
        tensor = tensor.reshape(rows, cols + pad_len)[..., :cols]
    else:
        tensor = tensor.reshape(orig_shape)
    return tensor


def _get_scale_shape(weight_shape, group_size: int):
    rows, cols = weight_shape
    if group_size == 0:
        return 1
    if group_size == -1 or cols <= group_size:
        return rows
    return rows * math.ceil(cols / group_size)


@ALGO_REGISTRY.register(
    name="autoround",
    description="Learnable rounding offsets inspired by Intel AutoRound",
    targets=("weight",),
)
class AutoRound(QuantAlgorithmBase):
    HIFX4_GROUP_SIZE = 64

    def __init__(self, args, w_bits):
        super().__init__()
        self.args = args
        self.bits = w_bits
        self.group_size = self._get_group_size(args)
        self.q_scale_thresh = 1e-5

        weight_shape = tuple(args.w_size)
        self.rows, self.columns = weight_shape
        self.is_hif4 = args.quant_dtype == "hifp"
        if self.is_hif4 and w_bits != 4:
            raise ValueError(
                f"AutoRound: only HiFloat4 (w_bits=4) is currently supported for "
                f"quant_dtype='hifp', got w_bits={w_bits!r}"
            )
        if self.is_hif4 and self.columns % self.HIFX4_GROUP_SIZE != 0:
            raise ValueError(
                "AutoRound: HiFloat4 requires the weight's column count to be a "
                "multiple of {}, got {}.".format(self.HIFX4_GROUP_SIZE, self.columns)
            )

        if self.is_hif4:
            # HiAutoRound: per-element mantissa compensation v + per-4-column-group
            # exponent compensation k. No separate learnable clip range like
            # min_scale/max_scale below -- HiF4 self-scales; k/v themselves are
            # bounded to [-_K_MAX, _K_MAX] / [-_V_MAX, _V_MAX] inside _quant_hifp.
            self.value = nn.Parameter(
                torch.zeros(self.rows, self.columns), requires_grad=True
            )
            self.k = nn.Parameter(
                torch.zeros(self.rows, self.columns // 4), requires_grad=True
            )
            return

        dummy_weight = torch.zeros(weight_shape)
        grouped_weight, _, _ = _reshape_pad_tensor_by_group_size(
            dummy_weight, self.group_size
        )
        scale_shape = _get_scale_shape(weight_shape, self.group_size)

        # Core AutoRound idea: learn per-group rounding offsets.
        self.value = nn.Parameter(torch.zeros_like(grouped_weight), requires_grad=True)
        self.min_scale = nn.Parameter(
            torch.ones(scale_shape, dtype=torch.float32),
            requires_grad=True,
        )
        self.max_scale = nn.Parameter(
            torch.ones(scale_shape, dtype=torch.float32),
            requires_grad=True,
        )

    @staticmethod
    def _get_group_size(args):
        if args.quant_dtype == "int":
            group_size = -1
        elif args.quant_dtype == "mxfp":
            group_size = 32
        elif args.quant_dtype == "hifp":
            group_size = 64
        else:
            raise ValueError(f"Not supported quant_dtype: {args.quant_dtype!r}")
        return group_size

    def prepare_deploy_weight(self, weight: torch.Tensor):
        grouped_weight, orig_shape, pad_len = _reshape_pad_tensor_by_group_size(
            weight, self.group_size
        )
        clip_min, clip_max = self._compute_clip_range(grouped_weight)

        clipped_weight = torch.clamp(grouped_weight, min=clip_min, max=clip_max)
        v = self.value.to(grouped_weight.device, grouped_weight.dtype)
        clipped_weight = _revert_tensor_by_pad(clipped_weight, orig_shape, pad_len)
        v = _revert_tensor_by_pad(v, orig_shape, pad_len)
        return clipped_weight, v

    def export_deploy(self, weight: torch.Tensor, quant_obj):
        if self.is_hif4:
            weight_q = _quant_hifp(weight, -1, self.k, self.value)
            return quant_obj.export_deploy(weight_q)
        clipped_weight, v = self.prepare_deploy_weight(weight)
        return quant_obj.export_deploy(clipped_weight, v=v)

    def quantize(self, weight: torch.Tensor, quant_obj):
        if self.is_hif4:
            return _quant_hifp(weight, -1, self.k, self.value)
        clipped_weight, v = self.prepare_deploy_weight(weight)
        return quant_obj(clipped_weight, v=v)

    def forward(self, weight: torch.Tensor):
        return weight

    def _reshape_scale(self, scale_param: torch.Tensor, grouped_weight: torch.Tensor):
        return scale_param.to(grouped_weight.device, grouped_weight.dtype).reshape(
            -1, 1
        )

    def _compute_clip_range(self, grouped_weight: torch.Tensor):
        min_scale = torch.clamp(
            self._reshape_scale(self.min_scale, grouped_weight), 0.0, 1.0
        )
        max_scale = torch.clamp(
            self._reshape_scale(self.max_scale, grouped_weight), 0.0, 1.0
        )

        group_min = torch.clamp(grouped_weight.amin(dim=-1, keepdim=True), max=0)
        group_max = torch.clamp(grouped_weight.amax(dim=-1, keepdim=True), min=0)
        tuned_min = -(group_min.abs() * min_scale)
        tuned_max = group_max * max_scale
        max_abs = torch.maximum(tuned_min.abs(), tuned_max.abs()).clamp(
            min=self.q_scale_thresh
        )
        clip_min = -max_abs
        clip_max = max_abs
        return clip_min, clip_max
