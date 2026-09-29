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
'''Explicit whitelist for pre-op calibration and local deployment adapters.'''

from dataclasses import dataclass

from amct_pytorch.classic.quantize_op import MinMaxQuant, SmoothQuant

from amct_pytorch.common.utils.log import LOGGER


@dataclass(frozen=True)
class LayerNormQuantSpec:
    '''Installation spec for one shared-input pre_op.'''

    pre_op_ref: str
    member_refs: tuple[str, ...]
    target_ref: str
    adapter: str
    algorithm: str
    act_granularity: str
    batch_num: int
    dynamic_per_token: bool


@dataclass(frozen=True)
class FusionGroupPattern:
    '''One pre_op and the relative member paths of its shared-input group.

    A member can declare multiple candidate paths because upstream model
    implementations may expose equivalent projections at different locations.
    '''

    pre_op: str
    members: tuple[str | tuple[str, ...], ...]
    group_type: str
    adapter: str


@dataclass(frozen=True)
class FusionLayerPattern:
    '''Relative paths inside a model layer.'''

    name: str
    required_attrs: tuple[str, ...]
    groups: tuple[FusionGroupPattern, ...]
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
                adapter='qwen3_attention',
            ),
            FusionGroupPattern(
                pre_op='post_attention_layernorm',
                members=('mlp.up_proj', 'mlp.gate_proj'),
                group_type='norm',
                adapter='qwen3_mlp',
            ),
        ),
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
                adapter='vit_attention',
            ),
            FusionGroupPattern(
                pre_op='layernorm_after',
                members=(('mlp.fc1', 'intermediate.dense'),),
                group_type='norm',
                adapter='vit_mlp',
            ),
        ),
        layer_types=('transformers.models.vit.modeling_vit.ViTLayer',),
    )
)


def build_layer_norm_quant_specs(model, patterns=None):
    '''Build LayerNormQuant installation specs from the whitelist.'''
    specs = []
    if patterns is None:
        patterns = FUSION_PATTERNS.patterns()
    used_members = set()
    for layer_name, module in model.named_modules():
        for pattern in patterns:
            module_type = f'{type(module).__module__}.{type(module).__qualname__}'
            if module_type not in pattern.layer_types:
                continue
            for group_pattern in pattern.groups:
                spec = _build_layer_norm_quant_spec(
                    model,
                    layer_name,
                    group_pattern,
                    used_members,
                )
                if spec is not None:
                    specs.append(spec)
    return specs


def _build_layer_norm_quant_spec(
    model,
    layer_name,
    group_pattern,
    used_members,
):
    pre_ref = _join_ref(layer_name, group_pattern.pre_op)
    try:
        pre_op = model.get_submodule(pre_ref)
        resolved_members = [
            _resolve_member(model, layer_name, member)
            for member in group_pattern.members
        ]
    except AttributeError as exc:
        LOGGER.logw(
            f'Skip LayerNormQuant {pre_ref}: {exc}. Keep independent quantization.',
            'LayerNormQuant',
        )
        return None
    if not _is_norm(pre_op) or getattr(pre_op, 'weight', None) is None:
        return None
    member_refs = [ref for ref, _ in resolved_members]
    members = [module for _, module in resolved_members]
    if not all(_is_quant_module(member) for member in members):
        LOGGER.logw(
            f'Skip LayerNormQuant {pre_ref}: members={member_refs} include unsupported or unquantized projections. Keep independent quantization.',
            'LayerNormQuant',
        )
        return None
    signatures = [_config_signature(member) for member in members]
    first = signatures[0]
    if any(signature != first for signature in signatures[1:]):
        LOGGER.logw(
            f'Skip LayerNormQuant {pre_ref}: members={member_refs} have inconsistent quant configs {signatures}. Keep independent quantization.',
            'LayerNormQuant',
        )
        return None
    if any(ref in used_members for ref in member_refs):
        return None
    if first['algorithm'] == 'minmax' and len(members) < 2:
        return None
    if first['strategy'] not in ('tensor', 'token'):
        return None
    if first['dynamic'] and (first['strategy'] != 'token' or not first['symmetric']):
        return None
    weights = [member.weight for member in members]
    if any(
        w.ndim != 2
        or w.shape[-1] != pre_op.weight.numel()
        or w.device != pre_op.weight.device
        or w.dtype != pre_op.weight.dtype
        for w in weights
    ):
        return None
    if any(
        w.device != weights[0].device or w.dtype != weights[0].dtype for w in weights
    ):
        return None
    target_ref = member_refs[0].rsplit('.', 1)[0]
    used_members.update(member_refs)
    return LayerNormQuantSpec(
        pre_op_ref=pre_ref,
        member_refs=tuple(member_refs),
        target_ref=target_ref,
        adapter=group_pattern.adapter,
        algorithm=first['algorithm'],
        act_granularity=first['strategy'],
        batch_num=first['batch_num'],
        dynamic_per_token=first['strategy'] == 'token' and first['dynamic'] is True,
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
    if type(module) not in (MinMaxQuant, SmoothQuant):
        return False
    inputs = module.quant_config.get('inputs_cfg', {})
    return (
        inputs.get('enable_quant', True) is not False
        and module.act_type == 'int8'
        and module.wts_type in ('int8', 'int4')
        and module.ori_module_type == 'Linear'
        and isinstance(module.batch_num, int)
        and module.batch_num > 0
    )


def _config_signature(module):
    cfg = module.quant_config
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
