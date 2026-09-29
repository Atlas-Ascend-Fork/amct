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
'''Fold calibrated shared-input parameters before deployment replacement.'''

import torch

from amct_pytorch.classic.deploy_op.local_dynamic import build_local_adapter
from amct_pytorch.classic.deploy_op.shared_quantize import FusedPreOp, SharedQuantize
from amct_pytorch.classic.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.classic.optimizer.layer_norm_quant import LayerNormQuant
from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.utils.model_util import ModuleHelper


class ApplyScalingFusionPass(BaseModuleFusionPass):
    '''Fold common scaling parameters into one local model structure.'''

    @torch.no_grad()
    def run(self, model):
        wrappers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, LayerNormQuant)
        ]
        # Refuse conversion before any local group is changed if any group is
        # still uncalibrated; otherwise the model would be only partly fused.
        for name, module in wrappers:
            if module.status != 'calibrated':
                raise RuntimeError(
                    f'LayerNormQuant {name} has not finished calibration'
                )
        super().run(model)

    @staticmethod
    def match_pattern(module, name):
        return isinstance(module, LayerNormQuant)

    def do_pass(self, model, object_module, object_name):
        members = [model.get_submodule(ref) for ref in object_module.member_refs]
        try:
            replacement, target, updates = self._prepare(model, object_module, members)
        except (ValueError, RuntimeError, OSError, AttributeError) as exc:
            ModuleHelper.replace_module_by_name(
                model, object_name, object_module.pre_op
            )
            LOGGER.logw(f'Skip scaling fusion {object_name}: {exc}', 'LayerNormQuant')
            return

        for parameter, value in updates:
            parameter.copy_(value)
        if target is not None:
            ModuleHelper.replace_module_by_name(model, object_module.target_ref, target)
        ModuleHelper.replace_module_by_name(model, object_name, replacement)
        for member in members:
            member.smooth_scale_skipped = object_module.algorithm == 'smoothquant'
            member.input_quant_disabled = (
                object_module.dynamic_per_token or len(members) > 1
            )
        object_module.status = 'fused'
        LOGGER.logd(
            f'Fused pre_op {object_name}, '
            f'members={list(object_module.member_refs)}, '
            f'dynamic_per_token={object_module.dynamic_per_token}',
            'LayerNormQuant',
        )

    @staticmethod
    def _prepare(model, module, members):
        _check_member_parameters(module, members)
        target = (
            build_local_adapter(
                module.adapter,
                model.get_submodule(module.target_ref),
                members[0].act_type,
            )
            if module.dynamic_per_token
            else None
        )
        return _build_pre_op(module, members), target, _build_updates(module, members)


def _check_member_parameters(module, members):
    for member in members:
        pairs = [
            (member.scale_d, module.activation_scale),
            (member.offset_d, module.activation_offset),
        ]
        if module.algorithm == 'smoothquant':
            pairs.append((member.scale, module.smooth_factor))
        if any(not _same_parameter(actual, expected) for actual, expected in pairs):
            raise ValueError(
                'member parameters differ from calibrated pre_op parameters'
            )


def _build_pre_op(module, members):
    if module.dynamic_per_token or len(members) == 1:
        return module.pre_op
    if module.activation_scale is None:
        raise ValueError('activation_scale is missing')
    shared = SharedQuantize(
        module.activation_scale,
        module.activation_offset,
        act_type=members[0].act_type,
        output_dtype=members[0].weight.dtype,
        act_granularity=module.act_granularity,
    ).to(members[0].weight.device)
    replacement = FusedPreOp(module.pre_op, shared)
    replacement.train(module.training)
    return replacement


def _build_updates(module, members):
    if module.algorithm != 'smoothquant':
        return []
    smooth = module.smooth_factor
    if smooth is None:
        raise ValueError('smooth_factor is missing')
    factor = smooth.reshape(-1)
    norm = module.pre_op
    if norm.weight is None or norm.weight.numel() != factor.numel():
        raise ValueError('norm channels do not match smooth factor')

    updates = []
    for parameter in (norm.weight, getattr(norm, 'bias', None)):
        if parameter is not None:
            value = parameter.float() / factor.to(parameter.device)
            updates.append((parameter, value.to(parameter.dtype)))
    for member in members:
        weight = member.weight
        value = weight.float() * smooth.to(weight.device).float()
        updates.append((weight, value.to(weight.dtype)))
    return updates


def _same_parameter(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return actual.shape == expected.shape and torch.equal(actual, expected)
