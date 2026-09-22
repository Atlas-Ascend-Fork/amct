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
'''First-forward shared-input structure discovery.'''

import torch

from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.classic.optimizer.fusion_group import FusionGroup, FusionRegistry


class _HookRecords:
    def __init__(self):
        self.producers = {}
        self.quant_inputs = []
        self.consumer_inputs = []
        self.quant_names = set()
        self.handles = []

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, torch.Tensor) for item in value)
    ):
        return value[0]
    return None


def install_fusion_hooks(model):
    '''Install temporary hooks that discover groups after one model forward.'''
    records = _HookRecords()
    named_modules = list(model.named_modules())

    for name, module in named_modules:

        def output_hook(mod, args, output, module_name=name):
            tensor = _tensor(output)
            if tensor is not None:
                records.producers[id(tensor)] = (tensor, module_name)

        records.handles.append(module.register_forward_hook(output_hook))

    for name, module in named_modules:
        if hasattr(module, 'quant_config'):
            records.quant_names.add(name)

    for name, module in named_modules:

        def input_hook(mod, args, module_name=name):
            tensor = _tensor(args)
            if tensor is None:
                return
            if hasattr(mod, 'quant_config'):
                records.quant_inputs.append((id(tensor), module_name))
            elif not module_name.startswith(
                tuple(f'{name}.' for name in records.quant_names)
            ):
                records.consumer_inputs.append((id(tensor), module_name))

        records.handles.append(module.register_forward_pre_hook(input_hook))

    def root_hook(mod, args, output):
        try:
            registry = build_fusion_groups(model, records)
            model._amct_fusion_registry = registry
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.logw(f'Failed to build fusion groups: {exc}', 'FusionGroup')
            model._amct_fusion_registry = FusionRegistry()
        finally:
            records.clear()

    records.handles.append(model.register_forward_hook(root_hook))
    return records


def _config_signature(module):
    cfg = module.quant_config
    inputs = cfg.get('inputs_cfg', {})
    signature = {
        'algorithm': cfg.get('algorithm'),
        'batch_num': cfg.get('batch_num'),
        'quant_type': inputs.get('quant_type'),
        'symmetric': inputs.get('symmetric'),
        'strategy': inputs.get('strategy'),
    }
    if signature['algorithm'] == 'smoothquant':
        signature['smooth_strength'] = cfg.get('smooth_strength')
    return signature


def _group_type(pre_op):
    module_name = type(pre_op).__name__.lower()
    if 'norm' in module_name:
        return 'norm'
    if isinstance(pre_op, torch.nn.Linear):
        return 'linear'
    return None


def build_fusion_groups(model, records):
    '''Build groups from the tensor identities captured by hooks.'''
    registry = FusionRegistry()
    candidates = {}
    for tensor_id, quant_name in records.quant_inputs:
        candidates.setdefault(tensor_id, []).append(quant_name)

    consumers_by_tensor = {}
    for tensor_id, consumer_name in records.consumer_inputs:
        consumers_by_tensor.setdefault(tensor_id, set()).add(consumer_name)

    producer_refs = {}
    for tensor_id, members in candidates.items():
        if tensor_id not in records.producers:
            LOGGER.logw(
                f'Skip fusion candidates {members}: producer is not a module',
                'FusionGroup',
            )
            continue
        tensor, pre_op_ref = records.producers[tensor_id]
        producer_refs[tensor_id] = pre_op_ref

    # All module consumers of each producer output must belong to the candidate.
    producer_consumers = {}
    for tensor_id, producer_ref in producer_refs.items():
        producer_consumers[producer_ref] = set()
    for tensor_id, producer_ref in producer_refs.items():
        producer_consumers[producer_ref].update(candidates[tensor_id])

    seen_members = set()
    for tensor_id, members in candidates.items():
        producer = records.producers.get(tensor_id)
        if producer is None:
            continue
        tensor, pre_op_ref = producer
        pre_op = model.get_submodule(pre_op_ref)
        group_type = _group_type(pre_op)
        signatures = [_config_signature(model.get_submodule(name)) for name in members]
        first = signatures[0]

        valid = group_type is not None
        if valid and any(signature != first for signature in signatures[1:]):
            valid = False
            LOGGER.logw(
                f'Skip fusion candidates {members}: quant configs are inconsistent',
                'FusionGroup',
            )
        if valid and first['algorithm'] == 'minmax' and len(members) < 2:
            valid = False
            LOGGER.logw(
                f'Skip fusion candidate {members}: minmax single-member group has no gain',
                'FusionGroup',
            )
        if valid and any(name in seen_members for name in members):
            valid = False
            LOGGER.logw(
                f'Skip fusion candidates {members}: quant module belongs to another group',
                'FusionGroup',
            )
        if valid:
            outside = consumers_by_tensor.get(tensor_id, set()) - set(members)
            if outside:
                valid = False
                LOGGER.logw(
                    f'Skip fusion candidates {members}: outside consumers {sorted(outside)}',
                    'FusionGroup',
                )

        if not valid:
            continue

        group = FusionGroup(
            group_id=registry.next_group_id(),
            group_type=group_type,
            pre_op_ref=pre_op_ref,
            scaling_layer_refs=members,
            algorithm=first['algorithm'],
            fusion_rule=group_type,
        )
        registry.add_group(group)
        seen_members.update(members)
        for member_name in members:
            model.get_submodule(member_name).fusion_group = group
        LOGGER.logd(
            f'Built fusion group {group.group_id}: pre_op={pre_op_ref}, '
            f'scaling_layers={members}',
            'FusionGroup',
        )

    return registry
