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
        for name, module in model.named_modules():
            if isinstance(module, LayerNormQuant) and module.status != 'calibrated':
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
            self._check_shared_parameters(object_module, members)
            replacement, target = self._build_replacement(model, object_module, members)
            updates = self._build_smooth_updates(object_module, members)
        except (ValueError, RuntimeError, OSError, AttributeError) as exc:
            ModuleHelper.replace_module_by_name(
                model, object_name, object_module.pre_op
            )
            LOGGER.logw(f'Skip scaling fusion {object_name}: {exc}', 'LayerNormQuant')
            return

        saved_parameters = [
            (parameter, parameter.detach().clone()) for parameter, _ in updates
        ]
        saved_target = (
            model.get_submodule(object_module.target_ref)
            if target is not None
            else None
        )
        try:
            for parameter, value in updates:
                parameter.copy_(value)
            if target is not None:
                ModuleHelper.replace_module_by_name(
                    model, object_module.target_ref, target
                )
            ModuleHelper.replace_module_by_name(model, object_name, replacement)
        except (ValueError, RuntimeError, OSError, AttributeError) as exc:
            for parameter, value in saved_parameters:
                parameter.copy_(value)
            if target is not None:
                ModuleHelper.replace_module_by_name(
                    model, object_module.target_ref, saved_target
                )
            ModuleHelper.replace_module_by_name(
                model, object_name, object_module.pre_op
            )
            LOGGER.logw(f'Skip scaling fusion {object_name}: {exc}', 'LayerNormQuant')
            return

        for member in members:
            member.smooth_scale_skipped = object_module.algorithm == 'smoothquant'
            member.input_quant_disabled = (
                object_module.dynamic_per_token or len(members) > 1
            )
        object_module.status = 'fused'
        LOGGER.logi(
            f'Fused shared-input group {object_name}: '
            f'algorithm={object_module.algorithm}, '
            f'mode={"dynamic" if object_module.dynamic_per_token else "static"}, '
            f'members={list(object_module.member_refs)}, '
            f'pre_op_wrapped={isinstance(replacement, FusedPreOp)}',
            'LayerNormQuant',
        )

    def _check_shared_parameters(self, module, members):
        expected_mode = 'dynamic' if module.dynamic_per_token else 'static'
        for member in members:
            if getattr(member, 'shared_param_mode', None) != expected_mode:
                raise ValueError(
                    f'member shared parameters mode must be {expected_mode!r}'
                )

        if module.dynamic_per_token:
            if any(member.scale_d is not None for member in members):
                raise ValueError('dynamic shared input must not have static scales')
        elif module.activation_scale is None:
            raise ValueError('activation_scale is missing')

        if module.algorithm == 'smoothquant':
            if module.smooth_factor is None:
                raise ValueError('smooth_factor is missing')
            if (
                not torch.isfinite(module.smooth_factor).all()
                or (module.smooth_factor <= 0).any()
            ):
                raise ValueError('smooth_factor must be finite and positive')

    def _build_replacement(self, model, module, members):
        if module.dynamic_per_token:
            target = build_local_adapter(
                module.adapter,
                model.get_submodule(module.target_ref),
                members[0].act_type,
            )
            return module.pre_op, target

        if len(members) == 1:
            return module.pre_op, None
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
        return replacement, None

    def _build_smooth_updates(self, module, members):
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
