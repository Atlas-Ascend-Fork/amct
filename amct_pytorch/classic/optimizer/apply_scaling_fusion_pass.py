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
'''Apply shared-input scaling fusion before deployment replacement.'''

import sys

from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.utils.model_util import ModuleHelper
from amct_pytorch.classic.deploy_op.shared_quantize import (
    FusedPreOp,
    SharedDynamicQuantize,
    SharedQuantize,
)


def _load_qwen3_fused_layer():
    '''Load the Qwen3 fused layer replica by direct file access.

    Going through the package import would execute the model registry
    __init__ (register_llm_models -> transformers and optional heavy deps);
    the replica itself only needs torch, so load the file directly and cache
    the module in sys.modules.
    '''
    cache_key = 'amct_qwen3_fused_layer'
    cached = sys.modules.get(cache_key)
    if cached is not None:
        return cached.Qwen3FusedDecoderLayer
    import importlib.util
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[2]
        / 'common/models/llm/qwen/qwen3/fused_layer.py'
    )
    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    return module.Qwen3FusedDecoderLayer


def _member_weight(member):
    '''Return the member's linear weight (real modules expose it directly,
    wrapped members keep the original nn.Linear under .linear).'''
    weight = getattr(member, 'weight', None)
    if weight is None:
        weight = member.linear.weight
    return weight


class ApplyScalingFusionPass:
    '''Fuse group parameters and insert shared activation quantization.'''

    def run(self, model):
        registry = getattr(model, '_amct_fusion_registry', None)
        if registry is None:
            return

        # Dynamic per-token groups that share one decoder layer must be fused
        # into a single layer replacement: replacing the layer once per group
        # would nest wrappers and replay the layer forward twice (B2).
        dynamic_by_layer = {}
        for group in registry.groups:
            if group.status == 'fused':
                LOGGER.logd(
                    f'Skip repeated fusion for group {group.group_id}',
                    'ScalingFusion',
                )
                continue
            if group.status != 'calibrated':
                continue
            if getattr(group, 'dynamic_per_token', False):
                dynamic_by_layer.setdefault(group.layer_ref, []).append(group)

        for group in registry.groups:
            if group.status == 'fused':
                continue
            if group.status != 'calibrated':
                continue
            if getattr(group, 'dynamic_per_token', False):
                continue
            try:
                self._apply_group(model, group)
                group.status = 'fused'
                LOGGER.logd(
                    f'Fused group {group.group_id}: pre_op={group.pre_op_ref}, '
                    f'scaling_layers={group.scaling_layer_refs}',
                    'ScalingFusion',
                )
            except (ValueError, RuntimeError) as exc:
                group.status = 'skipped'
                group.skip_reason = str(exc)
                LOGGER.logw(
                    f'Skip fusion group {group.group_id}: {exc}',
                    'ScalingFusion',
                )

        for layer_ref, groups in dynamic_by_layer.items():
            pending = [g for g in groups if g.status == 'calibrated']
            if not pending:
                continue
            try:
                self._apply_dynamic_layer(model, layer_ref, pending)
                for group in pending:
                    group.status = 'fused'
                    LOGGER.logd(
                        f'Fused dynamic group {group.group_id}: '
                        f'layer={layer_ref}, '
                        f'scaling_layers={group.scaling_layer_refs}',
                        'ScalingFusion',
                    )
            except (ValueError, RuntimeError, OSError) as exc:
                for group in pending:
                    group.status = 'skipped'
                    group.skip_reason = str(exc)
                LOGGER.logw(
                    f'Skip dynamic groups {[g.group_id for g in pending]} '
                    f'on layer {layer_ref}: {exc}',
                    'ScalingFusion',
                )

    def _apply_group(self, model, group):
        pre_op = model.get_submodule(group.pre_op_ref)
        members = [model.get_submodule(name) for name in group.scaling_layer_refs]
        self._fuse_scaling(group, pre_op, members)

        if len(members) >= 2:
            output_dtype = _member_weight(members[0]).dtype
            shared = SharedQuantize(
                group.activation_scale,
                group.activation_offset,
                act_type=members[0].act_type,
                output_dtype=output_dtype,
            )
            fused = FusedPreOp(pre_op, shared)
            ModuleHelper.replace_module_by_name(model, group.pre_op_ref, fused)
            for member in members:
                member.input_quant_disabled = True

    @staticmethod
    def _fuse_scaling(group, pre_op, members):
        if group.algorithm == 'smoothquant':
            smooth = group.smooth_factor
            if smooth is None:
                raise ValueError('smooth_factor is missing')
            if group.group_type == 'norm':
                ApplyScalingFusionPass._fuse_norm(pre_op, smooth)
            elif group.group_type == 'linear':
                ApplyScalingFusionPass._fuse_linear(pre_op, smooth)
            else:
                raise ValueError(f'unsupported pre_op type: {group.group_type}')
            for member in members:
                weight = _member_weight(member)
                weight.data.copy_(
                    (weight.data.float() * smooth.float()).to(weight.dtype)
                )
                member.smooth_scale_skipped = True
        elif group.algorithm == 'minmax':
            return
        else:
            raise ValueError(f'unsupported algorithm: {group.algorithm}')

    @staticmethod
    def _apply_dynamic_layer(model, layer_ref, groups):
        layer = model.get_submodule(layer_ref)
        if not all(
            hasattr(layer, attr)
            for attr in (
                'input_layernorm',
                'self_attn',
                'post_attention_layernorm',
                'mlp',
            )
        ):
            raise ValueError(
                f'layer {layer_ref} does not match the Qwen3 decoder structure'
            )

        Qwen3FusedDecoderLayer = _load_qwen3_fused_layer()

        # One shared quantizer per pre_op: the attention group shares the
        # input_layernorm output, the mlp group shares
        # post_attention_layernorm. Each takes act_type from its members.
        shared_by_pre_op = {}
        for group in groups:
            pre_op = model.get_submodule(group.pre_op_ref)
            members = [model.get_submodule(name) for name in group.scaling_layer_refs]
            ApplyScalingFusionPass._fuse_scaling(group, pre_op, members)
            first_member = model.get_submodule(group.scaling_layer_refs[0])
            shared_by_pre_op[group.pre_op_ref] = SharedDynamicQuantize(
                first_member.act_type
            )

        fused_layer = Qwen3FusedDecoderLayer.from_shared_map(
            layer, shared_by_pre_op, layer_ref
        )
        ModuleHelper.replace_module_by_name(model, layer_ref, fused_layer)
        for group in groups:
            for member_name in group.scaling_layer_refs:
                model.get_submodule(member_name).input_quant_disabled = True

    @staticmethod
    def _fuse_norm(pre_op, smooth):
        if hasattr(pre_op, 'weight'):
            factor = smooth.reshape(-1).to(pre_op.weight.device)
            if factor.numel() != pre_op.weight.numel():
                raise ValueError('norm pre_op channels do not match smooth factor')
            pre_op.weight.data.copy_(
                (pre_op.weight.data.float() / factor).to(pre_op.weight.dtype)
            )
        if hasattr(pre_op, 'bias') and pre_op.bias is not None:
            factor = smooth.reshape(-1).to(pre_op.bias.device)
            pre_op.bias.data.copy_(
                (pre_op.bias.data.float() / factor).to(pre_op.bias.dtype)
            )

    @staticmethod
    def _fuse_linear(pre_op, smooth):
        factor = smooth.reshape(-1)
        if factor.numel() != pre_op.weight.shape[0]:
            raise ValueError('linear pre_op output channels do not match smooth factor')
        pre_op.weight.data.copy_(
            (pre_op.weight.data.float() / factor[:, None]).to(pre_op.weight.dtype)
        )
        if pre_op.bias is not None:
            pre_op.bias.data.copy_(
                (pre_op.bias.data.float() / factor).to(pre_op.bias.dtype)
            )
