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
'''Pattern-whitelist based shared-input fusion group discovery.'''

from dataclasses import dataclass

from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.classic.optimizer.fusion_group import FusionGroup, FusionRegistry


@dataclass(frozen=True)
class FusionGroupPattern:
    '''One pre_op and the relative member paths of its shared-input group.

    A member can declare multiple candidate paths because upstream model
    implementations may expose equivalent projections at different locations.
    '''

    pre_op: str
    members: tuple[str, ...]
    group_type: str


@dataclass(frozen=True)
class FusionLayerPattern:
    '''Relative paths inside a model layer.'''

    name: str
    required_attrs: tuple[str, ...]
    groups: tuple[FusionGroupPattern, ...]
    dynamic_layer: str
    layer_types: tuple[str, ...]


class FusionPatternRegistry:
    '''Registry for explicitly supported layer patterns.'''

    def __init__(self):
        self._patterns = {}

    def register(self, pattern):
        if pattern.name in self._patterns:
            raise ValueError(f'Fusion pattern {pattern.name!r} is already registered')
        self._patterns[pattern.name] = pattern

    def get(self, name):
        return self._patterns[name]

    def patterns(self):
        return tuple(self._patterns.values())


FUSION_PATTERNS = FusionPatternRegistry()
FUSION_PATTERNS.register(
    FusionLayerPattern(
        name='qwen3',
        required_attrs=('input_layernorm', 'post_attention_layernorm'),
        groups=(
            FusionGroupPattern(
                pre_op='input_layernorm',
                members=(
                    'self_attn.q_proj',
                    'self_attn.k_proj',
                    'self_attn.v_proj',
                ),
                group_type='norm',
            ),
            FusionGroupPattern(
                pre_op='post_attention_layernorm',
                members=('mlp.up_proj', 'mlp.gate_proj'),
                group_type='norm',
            ),
        ),
        dynamic_layer='decoder_layer',
        layer_types=('transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer',),
    )
)
FUSION_PATTERNS.register(
    FusionLayerPattern(
        name='vit',
        required_attrs=('layernorm_before', 'layernorm_after'),
        groups=(
            FusionGroupPattern(
                pre_op='layernorm_before',
                members=(
                    ('attention.q_proj', 'attention.attention.query'),
                    ('attention.k_proj', 'attention.attention.key'),
                    ('attention.v_proj', 'attention.attention.value'),
                ),
                group_type='norm',
            ),
            FusionGroupPattern(
                pre_op='layernorm_after',
                members=(('mlp.fc1', 'intermediate.dense'),),
                group_type='norm',
            ),
        ),
        dynamic_layer='encoder_layer',
        layer_types=('transformers.models.vit.modeling_vit.ViTLayer',),
    )
)


def build_fusion_groups_from_patterns(model, patterns=None):
    '''Build fusion groups directly from the pattern whitelist.

    This function is intentionally static: it does not run a forward, register
    hooks, capture tensors, or infer structure from tensor identity.
    '''
    registry = FusionRegistry()
    if patterns is None:
        patterns = FUSION_PATTERNS.patterns()

    used_members = set()
    for layer_name, module in model.named_modules():
        for pattern in patterns:
            module_type = f'{type(module).__module__}.{type(module).__qualname__}'
            if module_type not in pattern.layer_types:
                continue
            if not all(hasattr(module, attr) for attr in pattern.required_attrs):
                continue
            for group_pattern in pattern.groups:
                _build_group(
                    model,
                    registry,
                    layer_name,
                    group_pattern,
                    used_members,
                    pattern.dynamic_layer,
                )
    return registry


def _build_group(
    model,
    registry,
    layer_name,
    group_pattern,
    used_members,
    dynamic_layer,
):
    pre_ref = _join_ref(layer_name, group_pattern.pre_op)
    resolved_members = [
        _resolve_member(model, layer_name, member) for member in group_pattern.members
    ]
    member_refs = [ref for ref, _ in resolved_members]
    try:
        pre_op = model.get_submodule(pre_ref)
        members = [module for _, module in resolved_members]
    except AttributeError as exc:
        LOGGER.logw(
            f'Skip fusion group {pre_ref} -> {member_refs}: module path is missing '
            f'({exc})',
            'FusionGroup',
        )
        return

    if group_pattern.group_type == 'norm' and not _is_norm(pre_op):
        LOGGER.logw(
            f'Skip fusion group {pre_ref}: pre_op is not a supported Norm',
            'FusionGroup',
        )
        return

    signatures = [_config_signature(member) for member in members]
    first = signatures[0] if signatures else None
    if not first or any(signature != first for signature in signatures[1:]):
        LOGGER.logw(
            f'Skip fusion candidates {member_refs}: quant configs are inconsistent',
            'FusionGroup',
        )
        return

    if not all(_is_quant_module(member) for member in members):
        LOGGER.logw(
            f'Skip fusion candidates {member_refs}: members are not all quantized',
            'FusionGroup',
        )
        return
    if any(ref in used_members for ref in member_refs):
        LOGGER.logw(
            f'Skip fusion candidates {member_refs}: quant module belongs to another group',
            'FusionGroup',
        )
        return
    if first['algorithm'] == 'minmax' and len(members) < 2:
        LOGGER.logw(
            f'Skip fusion candidate {member_refs}: minmax single-member group has no gain',
            'FusionGroup',
        )
        return

    group = FusionGroup(
        group_id=registry.next_group_id(),
        group_type=group_pattern.group_type,
        pre_op_ref=pre_ref,
        scaling_layer_refs=member_refs,
        algorithm=first['algorithm'],
        fusion_rule=group_pattern.group_type,
    )
    group.dynamic_per_token = first['strategy'] == 'token' and first['dynamic'] is True
    if group.dynamic_per_token:
        if not layer_name:
            LOGGER.logw(
                f'Skip dynamic fusion group {pre_ref}: root layer cannot be replaced in place',
                'FusionGroup',
            )
            return
        group.layer_ref = layer_name
        group.dynamic_layer = dynamic_layer
    registry.add_group(group)
    used_members.update(member_refs)
    for ref, member in zip(member_refs, members):
        member.fusion_group = group
        group.bind_member(member)
    LOGGER.logd(
        f'Built fusion group {group.group_id}: pre_op={pre_ref}, '
        f'scaling_layers={member_refs}',
        'FusionGroup',
    )


def _resolve_member(model, layer_name, member):
    candidates = (member,) if isinstance(member, str) else member
    for candidate in candidates:
        ref = _join_ref(layer_name, candidate)
        try:
            return ref, model.get_submodule(ref)
        except AttributeError:
            continue
    raise AttributeError(f'none of the member paths exists: {candidates}')


def _is_norm(module):
    return 'norm' in type(module).__name__.lower()


def _join_ref(layer_name, relative_ref):
    return relative_ref if not layer_name else f'{layer_name}.{relative_ref}'


def _is_quant_module(module):
    return hasattr(module, 'batch_num') or hasattr(module, 'act_type')


def _module_config(module):
    cfg = getattr(module, 'quant_config', None)
    if cfg is not None:
        return cfg
    return {
        'algorithm': getattr(module, 'algorithm', None),
        'batch_num': getattr(module, 'batch_num', None),
        'inputs_cfg': {
            'quant_type': getattr(module, 'act_type', None),
            'symmetric': getattr(module, 'act_symmetric', None),
            'strategy': getattr(module, 'act_granularity', None),
            'dynamic': getattr(module, 'dynamic', None),
        },
    }


def _config_signature(module):
    cfg = _module_config(module)
    inputs = cfg.get('inputs_cfg', {})
    signature = {
        'algorithm': cfg.get('algorithm'),
        'batch_num': cfg.get('batch_num'),
        'quant_type': inputs.get('quant_type'),
        'symmetric': inputs.get('symmetric'),
        'strategy': inputs.get('strategy'),
        'dynamic': inputs.get('dynamic'),
    }
    algorithm = signature['algorithm']
    if isinstance(algorithm, dict):
        algorithm = next(iter(algorithm))
    signature['algorithm'] = algorithm
    if algorithm == 'smoothquant':
        signature['smooth_strength'] = (
            cfg.get('algorithm', {}).get('smoothquant', {}).get('smooth_strength')
            if isinstance(cfg.get('algorithm'), dict)
            else cfg.get('smooth_strength')
        )
    return signature
