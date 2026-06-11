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

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY


def _reshape_pad_tensor_by_group_size(tensor: torch.Tensor, group_size: int, pad_value: float = 0.0):
    orig_shape = tensor.shape
    if tensor.ndim != 2:
        raise ValueError(f"AutoRound currently expects 2D weight tensors, got shape {orig_shape}")

    if group_size in (0, -1) or tensor.shape[-1] <= group_size:
        return tensor.reshape(tensor.shape[0], -1), orig_shape, 0

    pad_len = (group_size - tensor.shape[-1] % group_size) % group_size
    if pad_len > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_len), value=pad_value)
    return tensor.reshape(-1, group_size), orig_shape, pad_len


def _revert_tensor_by_pad(tensor: torch.Tensor, orig_shape, pad_len: int):
    if len(orig_shape) != 2:
        raise ValueError(f"AutoRound currently expects 2D weight tensors, got shape {orig_shape}")
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
class AutoRound(nn.Module):
    def __init__(self, args, w_bits):
        super().__init__()
        self.args = args
        self.bits = w_bits
        self.group_size = self._get_group_size(args)
        self.q_scale_thresh = 1e-5

        weight_shape = tuple(args.w_size)
        dummy_weight = torch.zeros(weight_shape)
        grouped_weight, _, _ = _reshape_pad_tensor_by_group_size(dummy_weight, self.group_size)
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
        else:
            group_size = 64
            raise ValueError("Not supported hifx for now")
        return group_size

    def trainable_params(self):
        params = [self.value, self.min_scale, self.max_scale]
        return params

    def export_ptq_params(self):
        return {
            "value": self.value.detach().cpu(),
            "min_scale": self.min_scale.detach().cpu(),
            "max_scale": self.max_scale.detach().cpu(),
        }

    def load_ptq_params(self, params):
        self.value.data.copy_(params["value"].to(device=self.value.device, dtype=self.value.dtype))
        self.min_scale.data.copy_(params["min_scale"].to(device=self.min_scale.device, dtype=self.min_scale.dtype))
        self.max_scale.data.copy_(params["max_scale"].to(device=self.max_scale.device, dtype=self.max_scale.dtype))

    def prepare_deploy_weight(self, weight: torch.Tensor):
        grouped_weight, orig_shape, pad_len = _reshape_pad_tensor_by_group_size(weight, self.group_size)
        clip_min, clip_max = self._compute_clip_range(grouped_weight)

        clipped_weight = torch.clamp(grouped_weight, min=clip_min, max=clip_max)
        v = self.value.to(grouped_weight.device, grouped_weight.dtype)
        clipped_weight = _revert_tensor_by_pad(clipped_weight, orig_shape, pad_len)
        v = _revert_tensor_by_pad(v, orig_shape, pad_len)
        return clipped_weight, v

    def export_deploy(self, weight: torch.Tensor, quant_obj):
        clipped_weight, v = self.prepare_deploy_weight(weight)
        return quant_obj.export_deploy(clipped_weight, v=v)

    def quantize(self, weight: torch.Tensor, quant_obj):
        clipped_weight, v = self.prepare_deploy_weight(weight)
        return quant_obj(clipped_weight, v=v)

    def forward(self, weight: torch.Tensor):
        return weight

    def _reshape_scale(self, scale_param: torch.Tensor, grouped_weight: torch.Tensor):
        return scale_param.to(grouped_weight.device, grouped_weight.dtype).reshape(-1, 1)

    def _compute_clip_range(self, grouped_weight: torch.Tensor):
        min_scale = torch.clamp(self._reshape_scale(self.min_scale, grouped_weight), 0.0, 1.0)
        max_scale = torch.clamp(self._reshape_scale(self.max_scale, grouped_weight), 0.0, 1.0)

        group_min = torch.clamp(grouped_weight.amin(dim=-1, keepdim=True), max=0)
        group_max = torch.clamp(grouped_weight.amax(dim=-1, keepdim=True), min=0)
        tuned_min = -(group_min.abs() * min_scale)
        tuned_max = group_max * max_scale
        max_abs = torch.maximum(tuned_min.abs(), tuned_max.abs()).clamp(min=self.q_scale_thresh)
        clip_min = -max_abs
        clip_max = max_abs
        return clip_min, clip_max
