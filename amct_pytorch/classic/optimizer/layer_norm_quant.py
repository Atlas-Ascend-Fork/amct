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
'''Pre-op module owning shared-input activation quantization parameters.'''

import torch
from torch import nn

from amct_pytorch.classic.quantize_op.utils import calculate_scale_offset
from amct_pytorch.common.utils.log import LOGGER


class LayerNormQuant(nn.Module):
    '''Wrap a Norm pre_op and calibrate one shared-input quantization group.

    The wrapper owns only the original pre_op. Scaling-layer modules stay in
    their original model positions. A plain tuple holds references without
    registering them as children; paths are kept for conversion and logging.
    '''

    def __init__(
        self,
        pre_op,
        scaling_layers,
        member_refs,
        target_ref,
        adapter,
        algorithm,
        act_granularity,
        batch_num,
        dynamic_per_token=False,
    ):
        super().__init__()
        if act_granularity == 'token' and not dynamic_per_token:
            raise ValueError(
                'static per-token quantization is unsupported: deployment scales '
                'depend on runtime batch and sequence length'
            )
        self.pre_op = pre_op
        # Do not use ModuleList: projections belong to their attention/MLP.
        # Whole-model deepcopy preserves these aliases through its shared memo.
        self.scaling_layers = tuple(scaling_layers)
        self.member_refs = tuple(member_refs)
        self.target_ref = target_ref
        self.adapter = adapter
        self.algorithm = algorithm
        self.act_granularity = act_granularity
        self.batch_num = batch_num
        self.dynamic_per_token = dynamic_per_token

        self.register_buffer('act_max', None, persistent=False)
        self.register_buffer('act_min', None, persistent=False)
        self.register_buffer('smooth_factor', None)
        self.register_buffer('activation_scale', None)
        self.register_buffer('activation_offset', None)
        self.batch_count = 0
        self.status = 'pending'
        self.training = pre_op.training
        for member in self.scaling_layers:
            member.shared_param_mode = 'pending'

    def forward(self, x):
        out = self.pre_op(x)
        if self.status == 'pending':
            self.update_activation_stat(out)
            if self.batch_count == self.batch_num:
                self.finalize_quant_params()
        return out

    @torch.no_grad()
    def update_activation_stat(self, inputs):
        if self.status != 'pending':
            return
        inputs = inputs.detach()
        if self.algorithm == 'minmax' and self.dynamic_per_token:
            self.batch_count += 1
            return
        batch_max = inputs.reshape(-1, inputs.shape[-1]).amax(dim=0, keepdim=True)
        batch_min = inputs.reshape(-1, inputs.shape[-1]).amin(dim=0, keepdim=True)
        self.act_max = (
            batch_max
            if self.act_max is None
            else torch.maximum(self.act_max, batch_max)
        )
        self.act_min = (
            batch_min
            if self.act_min is None
            else torch.minimum(self.act_min, batch_min)
        )
        self.batch_count += 1

    @torch.no_grad()
    def finalize_quant_params(self):
        if self.status == 'calibrated':
            return True
        if self.batch_count < self.batch_num:
            return False
        members = self.scaling_layers
        first = members[0]
        if self.algorithm == 'smoothquant':
            act_max = torch.maximum(self.act_max.abs(), self.act_min.abs())
            wts_max = members[0].weight.detach().abs().amax(dim=0, keepdim=True)
            for member in members[1:]:
                wts_max = torch.maximum(
                    wts_max, member.weight.detach().abs().amax(dim=0, keepdim=True)
                )
            alpha = first.quant_config['algorithm']['smoothquant']['smooth_strength']
            self.smooth_factor = _safe_smooth_factor(act_max, wts_max, alpha).to(
                first.weight.dtype
            )
        if not self.dynamic_per_token:
            maximum, minimum = self.act_max, self.act_min
            if self.smooth_factor is not None:
                maximum = maximum / self.smooth_factor
                minimum = minimum / self.smooth_factor
            maximum, minimum = maximum.max().reshape(-1), minimum.min().reshape(-1)
            self.activation_scale, self.activation_offset = calculate_scale_offset(
                maximum, minimum, first.act_symmetric, first.act_type
            )
        mode = 'dynamic' if self.dynamic_per_token else 'static'
        for member in members:
            member.accept_shared_params(
                mode,
                smooth_factor=self.smooth_factor,
                activation_scale=self.activation_scale,
                activation_offset=self.activation_offset,
            )
        self.status = 'calibrated'
        self.act_max = self.act_min = None
        LOGGER.logd(
            f'LayerNormQuant finalized: members={list(self.member_refs)}, '
            f'batch_count={self.batch_count}',
            'LayerNormQuant',
        )
        return True


def _safe_smooth_factor(act_max, wts_max, alpha):
    factor = ((act_max.float() ** alpha) / (wts_max.float() ** (1 - alpha))).to(
        act_max.dtype
    )
    invalid = ~torch.isfinite(factor) | (factor < torch.finfo(act_max.dtype).eps)
    if invalid.any():
        factor = torch.where(
            invalid,
            torch.ones_like(factor),
            factor,
        )
    return factor
