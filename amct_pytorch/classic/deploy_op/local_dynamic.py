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
'''Construct local adapters without changing the parent transformer layer.'''

import copy
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from amct_pytorch.classic.deploy_op.shared_quantize import SharedDynamicQuantize


_ADAPTER_BUILDERS = {
    'qwen3_attention': (
        'llm/qwen/qwen3/fused_modules.py',
        'qwen3_attention',
    ),
    'qwen3_mlp': ('llm/qwen/qwen3/fused_modules.py', 'qwen3_mlp'),
    'vit_attention': ('vit/fused_modules.py', 'vit_attention'),
    'vit_mlp': ('vit/fused_modules.py', 'vit_mlp'),
}
_ADAPTER_FORWARDS = {
    'qwen3_attention': ('llm/qwen/qwen3/fused_modules.py', '_attention_forward'),
    'qwen3_mlp': ('llm/qwen/qwen3/fused_modules.py', '_mlp_forward'),
    'vit_attention': ('vit/fused_modules.py', '_attention_forward'),
    'vit_mlp': ('vit/fused_modules.py', '_mlp_forward'),
}
_ADAPTER_REDUCE_VERSION = 1


def build_local_adapter(adapter, original, act_type):
    '''Build a registered attention or MLP adapter for dynamic deployment.'''
    if adapter not in _ADAPTER_BUILDERS:
        raise ValueError(f'unsupported dynamic adapter: {adapter}')
    relative, builder_name = _ADAPTER_BUILDERS[adapter]
    builder = getattr(_load_adapter_module(relative), builder_name)
    return builder(original, act_type)


def make_local_adapter(adapter, original, act_type):
    '''Build a local adapter from an already loaded adapter builder.'''
    if adapter not in _ADAPTER_BUILDERS:
        raise ValueError(f'unsupported dynamic adapter: {adapter}')
    relative, forward_name = _ADAPTER_FORWARDS[adapter]
    forward = getattr(_load_adapter_module(relative), forward_name)
    # Keep native attributes, training state and hooks. Copy registration maps so
    # adding the shared quantizer does not modify the original module.
    fused = copy.copy(original)
    fused.__class__ = _adapter_type(type(original), forward, adapter)
    for name in ('_modules', '_parameters', '_buffers'):
        setattr(fused, name, getattr(original, name).copy())
    fused._amct_adapter_id = adapter
    fused.shared_quantize = SharedDynamicQuantize(act_type)
    fused.shared_quantize.train(original.training)
    return fused


def _reduce_adapter(self):
    # Pickle checks __reduce__ before it tries to import the dynamic class by
    # name. A stable adapter id and version let a fresh process rebuild the
    # adapter without relying on runtime module registration.
    return _rebuild_adapter, (
        self._amct_adapter_id,
        _ADAPTER_REDUCE_VERSION,
        type(self).__mro__[1],
        self.__dict__,
    )


def _rebuild_adapter(adapter, version, original_type, state):
    if version != _ADAPTER_REDUCE_VERSION:
        raise ValueError(f'unsupported local adapter serialization version: {version}')
    if adapter not in _ADAPTER_BUILDERS:
        raise ValueError(f'unsupported dynamic adapter: {adapter}')
    relative, forward_name = _ADAPTER_FORWARDS[adapter]
    forward = getattr(_load_adapter_module(relative), forward_name)
    adapter_type = _adapter_type(original_type, forward, adapter)
    instance = adapter_type.__new__(adapter_type)
    instance.__dict__.update(state)
    return instance


@lru_cache(maxsize=None)
def _adapter_type(original_type, forward, adapter):
    adapter_type = type(
        f'Fused{original_type.__name__}',
        (original_type,),
        {'forward': forward, '__reduce__': _reduce_adapter},
    )
    return adapter_type


@lru_cache(maxsize=None)
def _load_adapter_module(relative):
    # Avoid importing model package initializers and their optional dependencies.
    path = Path(__file__).resolve().parents[2] / 'common' / 'models' / relative
    module_name = 'amct_' + relative.removesuffix('.py').replace('/', '_')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Dynamo resolves a traced function's globals through its module name.
    # Register only after execution succeeds, so an unavailable adapter cannot
    # leave a partially initialized module in the import table.
    sys.modules[spec.name] = module
    return module
