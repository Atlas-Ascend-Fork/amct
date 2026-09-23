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

from amct_pytorch.common.utils.log import LOGGER
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
        self.batch_num = None
        self.act_granularity = None
        self.batch_input = None
        self._finalized = False
        self._writeback_done = {}
        self._member_refs = list(self.scaling_layer_refs)

    def bind_member(self, member):
        '''Record the shared quant config so calibration can use group state.'''
        self.batch_num = getattr(member, 'batch_num', None)
        if self.batch_num is None:
            cfg = getattr(member, 'quant_config', None) or {}
            self.batch_num = cfg.get('batch_num')
        self.act_granularity = getattr(member, 'act_granularity', None) or (
            (getattr(member, 'quant_config', None) or {})
            .get('inputs_cfg', {})
            .get('strategy', 'tensor')
        )
        self._member_refs = list(self.scaling_layer_refs)

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

    def finalize_if_ready(self, caller):
        '''Finalize group params once the group batch count reaches batch_num.

        Called from member forwards. Any member may trigger the shared
        calculation exactly once; group params are stored on the group and
        written back to each member via writeback_if_pending at its own next
        forward, so no cross-module resolution is needed.
        Returns True when this call performed the finalization.
        '''
        if self._finalized or self.status == 'skipped':
            return False
        if self.batch_num is None or self.batch_count < self.batch_num:
            return False
        if self.algorithm == 'smoothquant' and len(self.wts_max_by_layer) < len(
            self._member_refs
        ):
            return False
        self._finalized = True
        return self.finalize_quant_params([caller])

    def writeback_if_pending(self, member):
        '''Apply stored group params to a member that has not received them.

        Runs at the top of each bound member's forward after finalize, so
        every member picks up the group results without module references.
        '''
        if self.status != 'calibrated' or self._writeback_done.get(id(member)):
            return
        self._apply_group_params(member)
        self._writeback_done[id(member)] = True

    def finalize_quant_params(self, members):
        '''Compute common group params and write them back to the given members.

        Params are stored on the group first; other members receive them via
        writeback_if_pending at their next forward.
        '''
        if self.act_max is None or self.act_min is None:
            self.status = 'skipped'
            self.skip_reason = 'activation statistics are incomplete'
            return False

        if self.algorithm == 'smoothquant':
            if self.smooth_factor is None:
                wts_max = None
                for member in members:
                    member_max = member.weight.detach().abs().amax(dim=0, keepdim=True)
                    wts_max = (
                        member_max
                        if wts_max is None
                        else torch.maximum(wts_max, member_max)
                    )
                if len(members) < len(self._member_refs):
                    # The triggering member only sees its own weights; other
                    # members' weight stats are collected at their writeback.
                    # Fallback: use per-layer stats already reported by each
                    # member via update_weight_stat (kept on the group).
                    wts_max = self._collect_wts_max(members, wts_max)
                alpha = members[0].quant_config['algorithm']['smoothquant'][
                    'smooth_strength'
                ]
                self.smooth_factor = _safe_smooth_factor(self.act_max, wts_max, alpha)
            smooth = self.smooth_factor
            if self.act_granularity == 'token' and self.batch_input is not None:
                smooth_input = self.batch_input / smooth.cpu()
                per_token_max = smooth_input.amax(dim=-1, keepdim=True)
                per_token_min = smooth_input.amin(dim=-1, keepdim=True)
                scale_d, offset_d = calculate_scale_offset(
                    per_token_max,
                    per_token_min,
                    members[0].act_symmetric,
                    members[0].act_type,
                )
            else:
                scaled_max = (self.act_max / smooth).max().reshape(-1)
                scaled_min = (self.act_min / smooth).min().reshape(-1)
                scale_d, offset_d = calculate_scale_offset(
                    scaled_max,
                    scaled_min,
                    members[0].act_symmetric,
                    members[0].act_type,
                )
            self.activation_scale = scale_d
            self.activation_offset = offset_d
        else:
            if self.act_granularity == 'tensor':
                max_val = self.act_max.max().reshape(-1)
                min_val = self.act_min.min().reshape(-1)
            else:
                max_val, min_val = self.act_max, self.act_min
            scale_d, offset_d = calculate_scale_offset(
                max_val, min_val, members[0].act_symmetric, members[0].act_type
            )
            self.activation_scale = scale_d
            self.activation_offset = offset_d

        for member in members:
            self._apply_group_params(member)
            self._writeback_done[id(member)] = True
        if self.status != 'skipped':
            self.status = 'calibrated'
            LOGGER.logd(
                f'Group {self.group_id} params finalized: '
                f'batch_count={self.batch_count}',
                'FusionGroup',
            )
        return self.status == 'calibrated'

    def _collect_wts_max(self, members, fallback):
        '''Combine reported per-layer weight stats with the fallback tensor.'''
        combined = fallback
        for layer_max in self.wts_max_by_layer.values():
            combined = (
                layer_max if combined is None else torch.maximum(combined, layer_max)
            )
        return combined

    def update_weight_stat(self, member):
        '''Record a member's per-input-channel weight max for the common stat.'''
        self.wts_max_by_layer[id(member)] = (
            member.weight.detach().abs().amax(dim=0, keepdim=True)
        )

    def _apply_group_params(self, member):
        '''Write common params onto one member (group-calibrated values only).'''
        if self.algorithm == 'smoothquant':
            smooth = self.smooth_factor
            param_dtype = member.weight.dtype
            member.scale = smooth.to(param_dtype)
            member.scale_d = self.activation_scale.to(param_dtype)
            member.offset_d = (
                None
                if self.activation_offset is None
                else self.activation_offset.to(param_dtype)
            )
        else:
            member.scale_d = self.activation_scale
            member.offset_d = self.activation_offset
        member.fake_quant_cache_ready = False


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
