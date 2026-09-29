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
"""Fuse calibrated pre-op parameters before replacing deployment projections."""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch

from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.utils.model_util import ModuleHelper
from amct_pytorch.classic.optimizer.layer_norm_quant import LayerNormQuant
from amct_pytorch.classic.deploy_op.shared_quantize import FusedPreOp, SharedQuantize


@lru_cache(maxsize=None)
def _load_adapter(adapter):
    # Avoid importing model package initializers and their optional dependencies.
    locations = {
        'qwen3_attention': (
            'llm/qwen/qwen3/fused_attention.py',
            'Qwen3FusedAttention',
            'from_attention',
        ),
        'qwen3_mlp': ('llm/qwen/qwen3/fused_mlp.py', 'Qwen3FusedMLP', 'from_mlp'),
        'vit_attention': (
            'vit/fused_attention.py',
            'ViTFusedAttention',
            'from_attention',
        ),
        'vit_mlp': ('vit/fused_mlp.py', 'ViTFusedMLP', 'from_mlp'),
    }
    if adapter not in locations:
        raise ValueError(f'unsupported dynamic adapter: {adapter}')
    relative, class_name, method = locations[adapter]
    path = Path(__file__).resolve().parents[2] / 'common/models' / relative
    spec = importlib.util.spec_from_file_location(f'amct_{adapter}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Dynamo resolves a traced function's globals through its module name.
    # Register only after execution succeeds, so an unavailable adapter cannot
    # leave a partially initialized module in the import table.
    sys.modules[spec.name] = module
    return getattr(getattr(module, class_name), method)


def _build_dynamic_target(target, adapter, act_type):
    return _load_adapter(adapter)(target, act_type)


def _same_parameter(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return actual.shape == expected.shape and torch.equal(actual, expected)


class ApplyScalingFusionPass:
    """Prepare each local replacement fully before modifying model parameters."""

    @torch.no_grad()
    def run(self, model):
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, LayerNormQuant)
        ]
        # A partially calibrated model must not be converted into a partly
        # fused model. Check all calibration states before committing any group.
        for name, module in modules:
            if module.status != 'calibrated':
                raise RuntimeError(
                    f'LayerNormQuant {name} has not finished calibration'
                )
        for name, module in modules:
            members = [model.get_submodule(ref) for ref in module.member_refs]
            try:
                replacement, target, updates = self._prepare(model, module, members)
            except (ValueError, RuntimeError, OSError, AttributeError) as exc:
                # No parameter has changed. Members can still deploy with their
                # calibrated common parameters and independent input quantization.
                ModuleHelper.replace_module_by_name(model, name, module.pre_op)
                LOGGER.logw(f'Skip scaling fusion {name}: {exc}', 'LayerNormQuant')
                continue
            for parameter, value in updates:
                parameter.copy_(value)
            if target is not None:
                ModuleHelper.replace_module_by_name(model, module.target_ref, target)
            ModuleHelper.replace_module_by_name(model, name, replacement)
            for member in members:
                member.smooth_scale_skipped = module.algorithm == 'smoothquant'
                member.input_quant_disabled = (
                    module.dynamic_per_token or len(members) > 1
                )
            module.status = 'fused'
            LOGGER.logd(
                f'Fused pre_op {name}, members={list(module.member_refs)}, '
                f'dynamic_per_token={module.dynamic_per_token}',
                'LayerNormQuant',
            )

    @staticmethod
    def _prepare(model, module, members):
        for member in members:
            values = [
                (member.scale_d, module.activation_scale),
                (member.offset_d, module.activation_offset),
            ]
            if module.algorithm == 'smoothquant':
                values.append((member.scale, module.smooth_factor))
            if any(
                not _same_parameter(actual, expected) for actual, expected in values
            ):
                raise ValueError(
                    'member parameters differ from calibrated pre_op parameters'
                )
        target = None
        replacement = module.pre_op
        if module.dynamic_per_token:
            target = _build_dynamic_target(
                model.get_submodule(module.target_ref),
                module.adapter,
                members[0].act_type,
            )
        elif len(members) > 1:
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
        updates = []
        if module.algorithm == 'smoothquant':
            smooth = module.smooth_factor
            if smooth is None:
                raise ValueError('smooth_factor is missing')
            factor = smooth.reshape(-1)
            norm = module.pre_op
            if norm.weight is None or norm.weight.numel() != factor.numel():
                raise ValueError('norm channels do not match smooth factor')
            for parameter in (norm.weight, getattr(norm, 'bias', None)):
                if parameter is not None:
                    value = parameter.float() / factor.to(parameter.device)
                    updates.append((parameter, value.to(parameter.dtype)))
            for member in members:
                weight = member.weight
                value = weight.float() * smooth.to(weight.device).float()
                updates.append((weight, value.to(weight.dtype)))
        return replacement, target, updates
