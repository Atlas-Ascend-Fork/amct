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

from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.utils.model_util import ModuleHelper
from amct_pytorch.classic.deploy_op.shared_quantize import (
    FusedPreOp,
    SharedDynamicQuantize,
    SharedQuantize,
)


class ApplyScalingFusionPass:
    '''Fuse group parameters and insert shared activation quantization.'''

    def run(self, model):
        registry = getattr(model, '_amct_fusion_registry', None)
        if registry is None:
            return
        for group in registry.groups:
            if group.status == 'fused':
                LOGGER.logd(
                    f'Skip repeated fusion for group {group.group_id}',
                    'ScalingFusion',
                )
                continue
            if group.status != 'calibrated':
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

    def _apply_group(self, model, group):
        pre_op = model.get_submodule(group.pre_op_ref)
        members = [model.get_submodule(name) for name in group.scaling_layer_refs]
        self._fuse_scaling(group, pre_op, members)

        if getattr(group, 'dynamic_per_token', False):
            self._apply_dynamic_group(model, group, members)
            return

        if len(members) >= 2:
            shared = SharedQuantize(group.activation_scale, group.activation_offset)
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
                member.linear.weight.data.copy_(
                    (member.linear.weight.data.float() * smooth.float()).to(
                        member.linear.weight.dtype
                    )
                )
                member.smooth_scale_skipped = True
        elif group.algorithm == 'minmax':
            return
        else:
            raise ValueError(f'unsupported algorithm: {group.algorithm}')

    @staticmethod
    def _apply_dynamic_group(model, group, members):
        if group.layer_ref is None:
            raise ValueError('dynamic per-token layer reference is missing')
        layer = model.get_submodule(group.layer_ref)
        if not all(
            hasattr(layer, attr)
            for attr in (
                'input_layernorm',
                'self_attn',
                'post_attention_layernorm',
                'mlp',
            )
        ):
            raise ValueError('layer does not match Qwen3 decoder structure')
        if not members:
            raise ValueError('dynamic per-token group has no scaling layers')

        import importlib.util
        from pathlib import Path

        module_path = (
            Path(__file__).resolve().parents[2]
            / 'common/models/llm/qwen/qwen3/fused_layer.py'
        )
        spec = importlib.util.spec_from_file_location(
            'amct_qwen3_fused_layer', module_path
        )
        if spec is None or spec.loader is None:
            raise ValueError('Qwen3 dynamic layer replica is unavailable')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Qwen3FusedDecoderLayer = module.Qwen3FusedDecoderLayer

        shared = SharedDynamicQuantize(members[0].act_type)
        fused_layer = Qwen3FusedDecoderLayer(layer, shared)
        ModuleHelper.replace_module_by_name(model, group.layer_ref, fused_layer)
        for member in members:
            member.input_quant_disabled = True

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
