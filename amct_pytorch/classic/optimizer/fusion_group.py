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
'''Shared-input fusion group registry.'''

import torch

from amct_pytorch.classic.quantize_op.utils import calculate_scale_offset


class FusionGroup:
    '''A pre_op and the quantization modules sharing its output.'''

    def __init__(
        self,
        group_id,
        group_type,
        pre_op_ref,
        scaling_layer_refs,
        algorithm,
        fusion_rule,
    ):
        self.group_id = group_id
        self.group_type = group_type
        self.pre_op_ref = pre_op_ref
        self.scaling_layer_refs = list(scaling_layer_refs)
        self.algorithm = algorithm
        self.fusion_rule = fusion_rule
        self.activation_scale = None
        self.activation_offset = None
        self.act_max = None
        self.act_min = None
        self.batch_count = 0
        self.wts_max_by_layer = {}
        self.smooth_factor = None
        self.status = 'pending'
        self.skip_reason = None
        self.last_input = None
        self.dynamic_per_token = False
        self.layer_ref = None

    def update_activation_stat(self, inputs):
        if inputs is self.last_input:
            return
        batch_max, batch_min = _reduce_channel_extrema(inputs)
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
        self.last_input = inputs

    def finalize_quant_params(self, members):
        if self.act_max is None or self.act_min is None:
            self.status = 'skipped'
            self.skip_reason = 'activation statistics are incomplete'
            return False

        if self.algorithm == 'smoothquant':
            wts_max = None
            for member in members:
                member_max = member.weight.detach().abs().amax(dim=0, keepdim=True)
                wts_max = (
                    member_max
                    if wts_max is None
                    else torch.maximum(wts_max, member_max)
                )
            alpha = members[0].quant_config['algorithm']['smoothquant'][
                'smooth_strength'
            ]
            smooth = _safe_smooth_factor(self.act_max, wts_max, alpha)
            self.smooth_factor = smooth
            scaled_max = (self.act_max / smooth).max().reshape(-1)
            scaled_min = (self.act_min / smooth).min().reshape(-1)
            scale_d, offset_d = calculate_scale_offset(
                scaled_max, scaled_min, members[0].act_symmetric, members[0].act_type
            )
            self.activation_scale = scale_d
            for member in members:
                weight = member.weight.detach() * smooth.to(member.weight.dtype)
                member.scale_w, member.offset_w = member.calculate_weights_scale_factor(
                    weight, member.wts_granularity
                )
                member.scale = smooth.to(member.scale_w.dtype)
                member.scale_d = scale_d.to(member.scale_w.dtype)
                member.offset_d = (
                    None if offset_d is None else offset_d.to(member.scale_w.dtype)
                )
                member.fake_quant_cache_ready = False
        else:
            scale_d, offset_d = calculate_scale_offset(
                self.act_max,
                self.act_min,
                members[0].act_symmetric,
                members[0].act_type,
            )
            self.activation_scale = scale_d
            for member in members:
                member.scale_d = scale_d
                member.offset_d = offset_d
                member.fake_quant_cache_ready = False

        self.status = 'calibrated'
        return True


class FusionRegistry:
    '''Model-level container for fusion groups.'''

    def __init__(self):
        self.groups = []
        self._next_group_id = 0

    def add_group(self, group):
        self.groups.append(group)

    def next_group_id(self):
        group_id = f'fusion_group_{self._next_group_id}'
        self._next_group_id += 1
        return group_id


def _reduce_channel_extrema(inputs):
    flattened = inputs.reshape(-1, inputs.shape[-1])
    return flattened.amax(dim=0, keepdim=True), flattened.amin(dim=0, keepdim=True)


def _safe_smooth_factor(act_max, wts_max, alpha):
    factor = (act_max.float() ** alpha) / (wts_max.float() ** (1 - alpha))
    eps = torch.finfo(act_max.dtype).eps
    invalid = ~torch.isfinite(factor) | (factor < eps)
    if invalid.any():
        factor = torch.where(
            invalid,
            torch.ones_like(factor),
            factor,
        )
    return factor
