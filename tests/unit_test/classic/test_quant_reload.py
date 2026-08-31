#!/usr/bin/env python3
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
"""Transactional whole-model reload of a published quant result."""

import copy
import json
import os

import pytest
import torch
import torch.nn as nn

from amct_pytorch.classic.quant_result import (
    MANIFEST_NAME,
    load_quant_manifest,
    read_quant_unit,
    reload_quant_params,
    resolve_layer_parent,
    validate_source_model,
)
from amct_pytorch.classic.quantize import quantize
from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
from amct_pytorch.classic.quantize_op.linear_awq_module import LinearAWQuant
from amct_pytorch.classic.quantize_op.minmax_module import MinMaxQuant


def _cfg(algorithm, activation=False):
    quant_cfg = {'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'}}
    if activation:
        quant_cfg['inputs'] = {'type': 'int8', 'symmetric': True, 'strategy': 'tensor'}
    return {
        'batch_num': 2 if activation else 1,
        'quant_cfg': quant_cfg,
        'algorithm': algorithm,
    }


MINMAX_CFG = _cfg({'minmax'})
MINMAX_ACT_CFG = _cfg({'minmax'}, activation=True)
AWQ_CFG = _cfg({'awq': {'grids_num': 4}})
# A finer grid actually picks a non-unit scale; with a coarse grid AWQ often
# settles on scale == 1, which would make any scaled-weight assertion vacuous.
AWQ_CFG_SEARCHED = _cfg({'awq': {'grids_num': 20}})
GPTQ_CFG = _cfg({'gptq'})
# MXFP4 GPTQ legitimately produces scale_w=None: it quantizes per group at deploy
# time. The artifact must still round-trip.
MXFP4_GPTQ_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'mxfp4_e2m1',
            'symmetric': True,
            'strategy': 'group',
            'group_size': 32,
        }
    },
    'algorithm': {'gptq'},
}

ALGO_CASES = (
    ('minmax', MINMAX_CFG, MinMaxQuant),
    ('awq', AWQ_CFG_SEARCHED, LinearAWQuant),
    ('gptq', GPTQ_CFG, GPTQuant),
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, inputs):
        return self.fc2(self.fc1(inputs))


class _NoBiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, inputs):
        return self.fc2(self.fc1(inputs))


class _NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))

    def forward(self, inputs):
        return self.block(inputs)


def _model(cls=_TinyModel):
    # int8 quantization only accepts bf16/fp16 source weights.
    return cls().to(torch.bfloat16)


def _inputs(batch=8, width=64):
    return torch.randn(batch, width).to(torch.bfloat16)


def _calibrate(model, config, out_dir, batches=2, width=64):
    """Run a full calibrate-and-dump cycle, returning the published directory."""
    os.mkdir(out_dir)
    quantize(model, config, quant_result_dir=str(out_dir))
    for _ in range(batches):
        model(_inputs(width=width))
    return str(out_dir)


def _artifact(tmp_path, config, name='art', model_cls=_TinyModel, batches=2, seed=0):
    """A published artifact plus the untouched float model it came from."""
    torch.manual_seed(seed)
    source = _model(model_cls)
    dumped = copy.deepcopy(source)
    out_dir = _calibrate(dumped, config, tmp_path / name, batches=batches)
    return source, dumped, out_dir


def _snapshot(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def _assert_unchanged(model, snapshot, module_objects):
    """The model must be byte-identical and hold the very same submodules."""
    after = model.state_dict()
    assert sorted(after) == sorted(snapshot)
    for key, value in snapshot.items():
        assert torch.equal(after[key], value), key
    assert [id(module) for module in model.children()] == module_objects


def _child_ids(model):
    return [id(module) for module in model.children()]


def _edit_manifest(result_dir, mutate):
    """Rewrite the manifest in place; used to forge invalid artifacts."""
    path = os.path.join(result_dir, MANIFEST_NAME)
    with open(path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    mutate(manifest)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle)
    return manifest


class TestReloadHappyPath:
    @pytest.mark.parametrize('name, config, quant_cls', ALGO_CASES)
    def test_reload_rebuilds_quant_modules(self, tmp_path, name, config, quant_cls):
        source, _, out_dir = _artifact(tmp_path, config, name=name)
        returned = reload_quant_params(source, out_dir)
        assert returned is source
        assert isinstance(source.fc1, quant_cls)
        assert isinstance(source.fc2, quant_cls)

    def test_reload_covers_activation_minmax(self, tmp_path):
        source, _, out_dir = _artifact(tmp_path, MINMAX_ACT_CFG, name='act')
        reload_quant_params(source, out_dir)
        assert isinstance(source.fc1, MinMaxQuant)
        # Activation calibration is what produces scale_d/offset_d.
        assert source.fc1.scale_d is not None
        assert source.fc1.scale_w is not None

    def test_reload_handles_nested_layer_names(self, tmp_path):
        source, _, out_dir = _artifact(
            tmp_path, MINMAX_CFG, name='nested', model_cls=_NestedModel
        )
        manifest = load_quant_manifest(out_dir)
        assert {unit['layer_name'] for unit in manifest['units']} == {
            'block.0',
            'block.1',
        }
        reload_quant_params(source, out_dir)
        assert isinstance(source.block[0], MinMaxQuant)
        assert isinstance(source.block[1], MinMaxQuant)


class TestReloadRejectsBadArtifact:
    """Every artifact defect must be caught before the model is touched."""

    @pytest.fixture
    def art(self, tmp_path):
        return _artifact(tmp_path, GPTQ_CFG, name='gptq')

    def _expect_rejected(self, model, result_dir, match=None):
        snapshot = _snapshot(model)
        children = _child_ids(model)
        with pytest.raises((ValueError, KeyError, FileNotFoundError), match=match):
            reload_quant_params(model, result_dir)
        _assert_unchanged(model, snapshot, children)

    def test_missing_manifest_is_rejected(self, art):
        source, _, out_dir = art
        os.remove(os.path.join(out_dir, MANIFEST_NAME))
        snapshot = _snapshot(source)
        children = _child_ids(source)
        with pytest.raises(FileNotFoundError):
            reload_quant_params(source, out_dir)
        _assert_unchanged(source, snapshot, children)

    def test_missing_unit_file_is_rejected(self, art):
        source, _, out_dir = art
        unit = load_quant_manifest(out_dir)['units'][0]['file']
        os.remove(os.path.join(out_dir, unit))
        self._expect_rejected(source, out_dir, match='missing file')

    def test_extra_unit_file_is_ignored(self, art):
        source, _, out_dir = art
        with open(os.path.join(out_dir, 'extra.safetensors'), 'wb') as handle:
            handle.write(b'junk')
        reload_quant_params(source, out_dir)

    def test_hidden_extra_file_is_ignored(self, art):
        source, _, out_dir = art
        with open(os.path.join(out_dir, '.hidden'), 'wb') as handle:
            handle.write(b'')
        reload_quant_params(source, out_dir)

    def test_truncated_unit_is_rejected(self, art):
        source, _, out_dir = art
        unit = load_quant_manifest(out_dir)['units'][0]['file']
        path = os.path.join(out_dir, unit)
        with open(path, 'r+b') as handle:
            handle.truncate(os.path.getsize(path) - 8)
        self._expect_rejected(source, out_dir, match='wrong size')

    def test_tampered_unit_bytes_are_rejected(self, art):
        source, _, out_dir = art
        unit = load_quant_manifest(out_dir)['units'][0]['file']
        with open(os.path.join(out_dir, unit), 'r+b') as handle:
            handle.seek(-4, os.SEEK_END)
            handle.write(b'\x00\x00\x00\x00')
        self._expect_rejected(source, out_dir, match='SHA-256')

    def test_entry_digest_mismatch_is_rejected(self, art):
        source, _, out_dir = art
        _edit_manifest(
            out_dir,
            lambda man: man['units'][0].update({'quant_module_type': 'MinMaxQuant'}),
        )
        self._expect_rejected(source, out_dir, match='entry_digest')

    def test_tensor_schema_mismatch_is_rejected(self, art):
        source, _, out_dir = art

        def drop_weight(manifest):
            entry = manifest['units'][0]
            entry['tensors'] = [
                tensor for tensor in entry['tensors'] if tensor['key'] != 'weight'
            ]

        _edit_manifest(out_dir, drop_weight)
        self._expect_rejected(source, out_dir)


class TestReloadRejectsWrongModel:
    @pytest.fixture
    def art(self, tmp_path):
        return _artifact(tmp_path, MINMAX_CFG, name='minmax')

    def _expect_rejected(self, model, result_dir, match=None):
        snapshot = _snapshot(model)
        children = _child_ids(model)
        with pytest.raises((ValueError, TypeError), match=match):
            reload_quant_params(model, result_dir)
        _assert_unchanged(model, snapshot, children)

    def test_different_weights_are_allowed_when_type_and_dtype_match(self, art):
        _, _, out_dir = art
        torch.manual_seed(123)
        model = _model()
        assert reload_quant_params(model, out_dir) is model
        assert isinstance(model.fc1, MinMaxQuant)

    def test_changed_weight_is_allowed_when_type_and_dtype_match(self, art):
        source, _, out_dir = art
        with torch.no_grad():
            source.fc1.weight[0, 0] += 1
        assert reload_quant_params(source, out_dir) is source
        assert isinstance(source.fc1, MinMaxQuant)

    def test_changed_dtype_is_rejected(self, art):
        source, _, out_dir = art
        self._expect_rejected(source.to(torch.float16), out_dir, match='not the model')

    def test_different_architecture_is_rejected(self, art):
        _, _, out_dir = art
        self._expect_rejected(_model(_NestedModel), out_dir)

    def test_different_model_type_is_rejected(self, art):
        source, _, out_dir = art

        class _Renamed(_TinyModel):
            pass

        renamed = _Renamed().to(torch.bfloat16)
        renamed.load_state_dict(source.state_dict())
        self._expect_rejected(renamed, out_dir, match='model_type')

    def test_validate_source_model_accepts_the_original(self, art):
        source, _, out_dir = art
        # Returns without raising; nothing else is asserted.
        validate_source_model(source, load_quant_manifest(out_dir)['source_model'])

    @pytest.mark.parametrize('bad', [None, [], 'x', 1])
    def test_validate_source_model_rejects_non_mapping(self, art, bad):
        source, _, _ = art
        with pytest.raises((ValueError, TypeError)):
            validate_source_model(source, bad)


class TestReloadRejectsBadStructure:
    @pytest.fixture
    def art(self, tmp_path):
        return _artifact(tmp_path, MINMAX_CFG, name='minmax')

    def _expect_rejected(self, model, result_dir, match=None):
        snapshot = _snapshot(model)
        children = _child_ids(model)
        with pytest.raises(ValueError, match=match):
            reload_quant_params(model, result_dir)
        _assert_unchanged(model, snapshot, children)

    def test_unknown_quant_module_type_is_rejected(self, art):
        source, _, out_dir = art
        _edit_manifest(
            out_dir,
            lambda man: [
                unit.update({'quant_module_type': 'NoSuchQuant'})
                for unit in man['units']
            ],
        )
        # Rejected by the entry digest before the name is even looked up: the
        # module type is covered by the digest, so it cannot be swapped silently.
        self._expect_rejected(source, out_dir)

    def test_unregistered_quant_module_name_cannot_be_resolved(self):
        from amct_pytorch.classic.quant_result import _resolve_quant_module_class

        entry = {
            'layer_name': 'fc1',
            'ori_module_type': 'Linear',
            'quant_module_type': 'NoSuchQuant',
            'quant_config': {'algorithm': {'minmax': {}}},
        }
        with pytest.raises(ValueError, match='does not match the registered class'):
            _resolve_quant_module_class(entry)

    def test_algorithm_and_source_type_resolve_the_registered_class(self):
        from amct_pytorch.classic.quant_result import _resolve_quant_module_class

        entry = {
            'layer_name': 'fc1',
            'ori_module_type': 'Linear',
            'quant_module_type': 'MinMaxQuant',
            'quant_config': {'algorithm': {'minmax': {}}},
        }
        assert _resolve_quant_module_class(entry) is MinMaxQuant

    def test_unregistered_algorithm_is_rejected(self):
        from amct_pytorch.classic.quant_result import _resolve_quant_module_class

        entry = {
            'layer_name': 'fc1',
            'ori_module_type': 'Linear',
            'quant_module_type': 'MinMaxQuant',
            'quant_config': {'algorithm': {'not_registered': {}}},
        }
        with pytest.raises(ValueError, match='not registered'):
            _resolve_quant_module_class(entry)

    def test_unregistered_source_type_is_rejected(self):
        from amct_pytorch.classic.quant_result import _resolve_quant_module_class

        entry = {
            'layer_name': 'fc1',
            'ori_module_type': 'Conv2d',
            'quant_module_type': 'MinMaxQuant',
            'quant_config': {'algorithm': {'minmax': {}}},
        }
        with pytest.raises(ValueError, match='no registered quantize module'):
            _resolve_quant_module_class(entry)

    def test_multiple_algorithms_are_rejected(self):
        from amct_pytorch.classic.quant_result import _resolve_quant_module_class

        entry = {
            'layer_name': 'fc1',
            'ori_module_type': 'Linear',
            'quant_module_type': 'MinMaxQuant',
            'quant_config': {'algorithm': {'minmax': {}, 'awq': {}}},
        }
        with pytest.raises(ValueError, match='exactly one algorithm'):
            _resolve_quant_module_class(entry)

    def test_algorithm_outside_the_artifact_contract_is_rejected(self):
        from amct_pytorch.classic.quant_result import _resolve_quant_module_class

        entry = {
            'layer_name': 'fc1',
            'ori_module_type': 'Linear',
            'quant_module_type': 'SmoothQuant',
            'quant_config': {'algorithm': {'smoothquant': {}}},
        }
        with pytest.raises(ValueError, match='does not support quant result reload'):
            _resolve_quant_module_class(entry)

    def test_ori_module_type_mismatch_is_rejected(self, art):
        source, _, out_dir = art
        _edit_manifest(
            out_dir,
            lambda man: [
                unit.update({'ori_module_type': 'Conv2d'}) for unit in man['units']
            ],
        )
        self._expect_rejected(source, out_dir, match='was produced from')

    def test_unknown_layer_is_rejected(self, art):
        source, _, out_dir = art
        _edit_manifest(
            out_dir,
            lambda man: [
                unit.update({'layer_name': unit['layer_name'] + '_gone'})
                for unit in man['units']
            ],
        )
        self._expect_rejected(source, out_dir, match='has no submodule')

    def test_unreachable_parent_path_is_rejected(self, art):
        source, _, out_dir = art
        _edit_manifest(
            out_dir,
            lambda man: [
                unit.update({'layer_name': 'missing.' + unit['layer_name']})
                for unit in man['units']
            ],
        )
        self._expect_rejected(source, out_dir, match='not a submodule')

    def test_nested_unit_paths_are_rejected(self, art):
        source, _, out_dir = art
        _edit_manifest(
            out_dir,
            lambda man: man['units'][1].update(
                {'layer_name': man['units'][0]['layer_name'] + '.inner'}
            ),
        )
        self._expect_rejected(source, out_dir, match='would discard')

    @pytest.mark.parametrize('bad_name', ['', 'fc1..w', '.fc1'])
    def test_resolve_layer_parent_rejects_bad_names(self, bad_name):
        with pytest.raises(ValueError):
            resolve_layer_parent(_model(), bad_name)

    def test_resolve_layer_parent_returns_parent_and_child(self):
        model = _model(_NestedModel)
        parent, attr_name, child = resolve_layer_parent(model, 'block.1')
        assert parent is model.block
        assert attr_name == '1'
        assert child is model.block[1]


class TestRequiredQuantParamsPerDtype:
    """Required params follow the dtype, because the writer's output does too."""

    @staticmethod
    def _awq(quant_type, group_size=None, symmetric=True):
        from amct_pytorch.classic.quantize_op.linear_awq_module import LinearAWQuant

        config = {
            'batch_num': 1,
            'weights_cfg': {
                'quant_type': quant_type,
                'symmetric': symmetric,
                'strategy': 'group' if group_size else 'channel',
                'group_size': group_size,
            },
            'inputs_cfg': {'enable_quant': False},
            'algorithm': {'awq': {'grids_num': 4}},
        }
        return LinearAWQuant(nn.Linear(64, 32).to(torch.bfloat16), 'fc1', config)

    @staticmethod
    def _gptq(quant_type, group_size=None, symmetric=True):
        config = {
            'batch_num': 1,
            'weights_cfg': {
                'quant_type': quant_type,
                'symmetric': symmetric,
                'strategy': 'group' if group_size else 'channel',
                'group_size': group_size,
            },
            'inputs_cfg': {'enable_quant': False},
            'algorithm': {'gptq'},
        }
        return GPTQuant(nn.Linear(64, 32).to(torch.bfloat16), 'fc1', config)

    @pytest.mark.parametrize('quant_type', ['int8', 'int4', 'float4_e2m1'])
    def test_awq_requires_scale_w_for_scaled_dtypes(self, quant_type):
        module = self._awq(quant_type)
        scale = torch.ones(1, 64).to(torch.bfloat16)
        with pytest.raises(ValueError, match='missing required quant param'):
            module.load_quant_params({'scale': scale}, {})

    def test_awq_allows_missing_scale_w_for_mxfp4(self):
        module = self._awq('mxfp4_e2m1', group_size=32)
        module.load_quant_params({'scale': torch.ones(1, 64).to(torch.bfloat16)}, {})
        assert module.calc_done is True
        assert getattr(module, 'scale_w', None) is None

    def test_awq_always_requires_scale(self):
        module = self._awq('int8')
        with pytest.raises(ValueError, match='missing required quant param'):
            module.load_quant_params({'scale_w': torch.ones(32, 1)}, {})

    def test_awq_asymmetric_quant_requires_offset_w(self):
        module = self._awq('int8', symmetric=False)
        with pytest.raises(ValueError, match='missing required quant param'):
            module.load_quant_params(
                {
                    'scale': torch.ones(1, 64).to(torch.bfloat16),
                    'scale_w': torch.ones(32, 1),
                },
                {},
            )

    def test_gptq_requires_scale_w_for_int8(self):
        module = self._gptq('int8')
        with pytest.raises(ValueError, match='missing required quant param'):
            module.load_quant_params({'weight': torch.zeros(32, 64).bfloat16()}, {})

    def test_gptq_allows_missing_scale_w_for_mxfp4(self):
        module = self._gptq('mxfp4_e2m1', group_size=32)
        module.load_quant_params({'weight': torch.zeros(32, 64).bfloat16()}, {})
        assert module.scale_w is None
        assert module.hessian is None

    def test_gptq_asymmetric_quant_requires_offset_w(self):
        module = self._gptq('int8', symmetric=False)
        with pytest.raises(ValueError, match='missing required quant param'):
            module.load_quant_params(
                {
                    'weight': torch.zeros(32, 64).bfloat16(),
                    'scale_w': torch.ones(32, 1),
                },
                {},
            )


class TestRequiredMinMaxParams:
    """MinMax reload rejects a unit that omits config-required state."""

    @staticmethod
    def _module(activation=False, symmetric=True):
        config = {
            'batch_num': 2 if activation else 1,
            'weights_cfg': {
                'quant_type': 'int8',
                'symmetric': symmetric,
                'strategy': 'channel',
            },
            'inputs_cfg': {
                'enable_quant': activation,
                'quant_type': 'int8',
                'symmetric': symmetric,
                'strategy': 'tensor',
            },
            'algorithm': {'minmax'},
        }
        return MinMaxQuant(nn.Linear(64, 32).to(torch.bfloat16), 'fc1', config)

    def test_weight_quant_requires_scale_w(self):
        module = self._module()
        with pytest.raises(ValueError, match='missing required quant param'):
            module.load_quant_params({}, {})

    def test_activation_quant_requires_scale_d(self):
        module = self._module(activation=True)
        with pytest.raises(ValueError, match='missing required quant param'):
            module.load_quant_params({'scale_w': module.scale_w.clone()}, {})

    def test_asymmetric_quant_requires_offsets(self):
        module = self._module(activation=True, symmetric=False)
        with pytest.raises(ValueError, match='missing required quant param'):
            module.load_quant_params(
                {
                    'scale_w': module.scale_w.clone(),
                    'scale_d': torch.ones(1),
                },
                {},
            )


class TestMxfp4Gptq:
    """MXFP4 GPTQ has no scale_w; a published artifact must still reload."""

    def test_mxfp4_artifact_reloads_without_scale_w(self, tmp_path):
        torch.manual_seed(0)
        base = _NoBiasModel().to(torch.bfloat16)
        dumped = copy.deepcopy(base)
        out_dir = str(tmp_path / 'mxfp4')
        os.mkdir(out_dir)
        quantize(dumped, MXFP4_GPTQ_CFG, quant_result_dir=out_dir)
        dumped(_inputs())
        assert dumped.fc1.scale_w is None

        units = load_quant_manifest(out_dir)['units']
        for unit in units:
            keys = {tensor['key'] for tensor in unit['tensors']}
            assert 'weight' in keys
            assert 'scale_w' not in keys

        source = copy.deepcopy(base)
        reload_quant_params(source, out_dir)
        assert isinstance(source.fc1, GPTQuant)
        assert source.fc1.scale_w is None
        assert torch.equal(source.fc1.weight.data, dumped.fc1.weight.data)

    def test_mxfp4_reloaded_forward_matches_direct(self, tmp_path):
        torch.manual_seed(0)
        base = _NoBiasModel().to(torch.bfloat16)
        probe = _inputs()
        dumped = copy.deepcopy(base)
        out_dir = str(tmp_path / 'mxfp4_eq')
        os.mkdir(out_dir)
        quantize(dumped, MXFP4_GPTQ_CFG, quant_result_dir=out_dir)
        dumped(probe)
        direct_out = dumped(probe)

        source = copy.deepcopy(base)
        reload_quant_params(source, out_dir)
        assert torch.equal(direct_out, source(probe))


class TestReloadIsTransactional:
    """A failure on any layer must leave the whole model untouched."""

    @pytest.mark.parametrize('name, config, _cls', ALGO_CASES)
    def test_second_layer_failure_leaves_first_layer_intact(
        self, tmp_path, name, config, _cls
    ):
        source, _, out_dir = _artifact(tmp_path, config, name=name)
        _edit_manifest(
            out_dir, lambda man: man['units'][1].update({'ori_module_type': 'Conv2d'})
        )
        snapshot = _snapshot(source)
        children = _child_ids(source)
        first_layer, second_layer = source.fc1, source.fc2
        with pytest.raises(ValueError):
            reload_quant_params(source, out_dir)
        # Not merely "still a Linear": the very same objects, byte-identical.
        assert source.fc1 is first_layer
        assert source.fc2 is second_layer
        _assert_unchanged(source, snapshot, children)

    def test_gptq_staging_does_not_mutate_the_source_weight(self, tmp_path):
        """GPTQ shares ori_module.weight, so staging must work on a copy."""
        source, _, out_dir = _artifact(tmp_path, GPTQ_CFG, name='gptq_tx')
        weight_before = source.fc1.weight.detach().clone()
        weight_obj = source.fc1.weight
        _edit_manifest(
            out_dir, lambda man: man['units'][1].update({'ori_module_type': 'Conv2d'})
        )
        with pytest.raises(ValueError):
            reload_quant_params(source, out_dir)
        assert source.fc1.weight is weight_obj
        assert torch.equal(source.fc1.weight, weight_before)

    def test_successful_reload_replaces_every_listed_layer(self, tmp_path):
        source, _, out_dir = _artifact(tmp_path, GPTQ_CFG, name='gptq_ok')
        reload_quant_params(source, out_dir)
        assert isinstance(source.fc1, GPTQuant)
        assert isinstance(source.fc2, GPTQuant)

    def test_setattr_failure_rolls_back_committed_layers(self, tmp_path, monkeypatch):
        """An unexpected commit-stage error restores the modules already swapped."""
        import amct_pytorch.classic.quant_result as module

        source, _, out_dir = _artifact(tmp_path, MINMAX_CFG, name='rollback')
        first_layer, second_layer = source.fc1, source.fc2
        snapshot = _snapshot(source)
        children = _child_ids(source)

        real_setattr = module.setattr if hasattr(module, 'setattr') else setattr
        calls = {'n': 0}

        def flaky_setattr(obj, name, value):
            calls['n'] += 1
            if calls['n'] == 2:
                raise RuntimeError('boom')
            return real_setattr(obj, name, value)

        monkeypatch.setattr(module, 'setattr', flaky_setattr, raising=False)
        with pytest.raises(RuntimeError, match='boom'):
            reload_quant_params(source, out_dir)
        assert source.fc1 is first_layer
        assert source.fc2 is second_layer
        _assert_unchanged(source, snapshot, children)


def _calibrated_state(model, batches, width=64):
    """Forward a few batches, then capture the post-calibration output."""
    for _ in range(batches):
        model(_inputs(width=width))
    torch.manual_seed(7)
    probe = _inputs(width=width)
    return model(probe), probe


class TestCalibrationStateEquivalence:
    """Reload must reproduce the calibrated module state, not approximate it."""

    @pytest.mark.parametrize(
        'name, config, quant_cls',
        [case for case in ALGO_CASES if case[0] != 'awq'],
    )
    def test_reloaded_forward_matches_direct_calibration(
        self, tmp_path, name, config, quant_cls
    ):
        source, dumped, out_dir = _artifact(tmp_path, config, name=name)
        direct_out, probe = _calibrated_state(dumped, batches=0)

        reload_quant_params(source, out_dir)
        reloaded_out = source(probe)
        assert isinstance(source.fc1, quant_cls)
        # Bit-for-bit: an ulp of drift here would mean the reloaded model is not
        # the calibrated model.
        assert torch.equal(direct_out, reloaded_out)

    def test_awq_reloaded_forward_is_close_but_not_required_bit_exact(self, tmp_path):
        """AWQ's calibration state is only recoverable up to one rounding step.

        Calibration scales ori_module.weight in place; a reloaded run has no such
        tensor and must recompute weight/scale, where scale is 1/scale_awq already
        rounded to the weight precision. That round trip is not exact, so the
        contract for AWQ is closeness here and bit-exactness at deploy, where the
        scale is applied to the unscaled weight instead.
        """
        source, dumped, out_dir = _artifact(
            tmp_path, AWQ_CFG_SEARCHED, name='awq_cal', seed=3
        )
        assert not torch.all(dumped.fc1.scale == 1.0), 'scale must be non-trivial'
        direct_out, probe = _calibrated_state(dumped, batches=0)

        reload_quant_params(source, out_dir)
        assert source.fc1._scale_reapplied is True
        reloaded_out = source(probe)
        torch.testing.assert_close(direct_out, reloaded_out, rtol=0.05, atol=0.05)

    def test_awq_does_not_persist_weight(self, tmp_path):
        _, _, out_dir = _artifact(tmp_path, AWQ_CFG, name='awq_keys')
        for unit in load_quant_manifest(out_dir)['units']:
            keys = {tensor['key'] for tensor in unit['tensors']}
            assert 'scale' in keys
            assert 'scale_w' in keys
            assert 'weight' not in keys

    def test_gptq_persists_optimized_weight(self, tmp_path):
        source, dumped, out_dir = _artifact(tmp_path, GPTQ_CFG, name='gptq_keys')
        for unit in load_quant_manifest(out_dir)['units']:
            keys = {tensor['key'] for tensor in unit['tensors']}
            assert 'weight' in keys
            assert 'scale_w' in keys
        # The persisted weight is the Hessian-corrected one, not the input weight.
        float_weight = source.fc1.weight.detach().clone()
        reload_quant_params(source, out_dir)
        assert torch.equal(source.fc1.weight.data, dumped.fc1.weight.data)
        assert not torch.equal(source.fc1.weight.data, float_weight)

    def test_awq_reload_does_not_search_scale_again(self, tmp_path):
        source, _, out_dir = _artifact(tmp_path, AWQ_CFG, name='awq_nosearch')
        reload_quant_params(source, out_dir)
        assert source.fc1.calc_done is True
        scale_before = source.fc1.scale.clone()
        source(_inputs())
        assert torch.equal(source.fc1.scale, scale_before)

    def test_gptq_reload_does_not_recompute_hessian(self, tmp_path):
        source, _, out_dir = _artifact(tmp_path, GPTQ_CFG, name='gptq_nohess')
        reload_quant_params(source, out_dir)
        assert source.fc1.hessian is None
        assert source.fc1.cur_batch > source.fc1.quant_config.get('batch_num')
        weight_before = source.fc1.weight.detach().clone()
        source(_inputs())
        assert source.fc1.hessian is None
        assert torch.equal(source.fc1.weight, weight_before)

    def test_gptq_unit_without_weight_is_rejected(self, tmp_path):
        """The constructor pre-fills weight, so absence must be caught explicitly."""
        from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant as _G

        module = _G(
            nn.Linear(8, 4).to(torch.bfloat16),
            'fc',
            dict(
                GPTQ_CFG,
                **{
                    'weights_cfg': {'quant_type': 'int8', 'group_size': None},
                    'inputs_cfg': {'enable_quant': False},
                },
            ),
        )
        with pytest.raises(ValueError, match='missing required quant param'):
            module.load_quant_params({'scale_w': torch.ones(4)}, {})


def _deploy_signature(model):
    """Layer types plus every deploy tensor, the shape equivalence is judged on."""
    types = {name: type(module).__name__ for name, module in model.named_modules()}
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return types, state


def _assert_same_deploy_model(left, right):
    left_types, left_state = _deploy_signature(left)
    right_types, right_state = _deploy_signature(right)
    assert left_types == right_types
    assert sorted(left_state) == sorted(right_state)
    for key, value in left_state.items():
        assert value.dtype == right_state[key].dtype, key
        assert value.shape == right_state[key].shape, key
        assert torch.equal(value, right_state[key]), key


def _real_torch_npu():
    """True when the genuine torch_npu is importable and not a test stub.

    Deploy conversion calls into real torch_npu ops, and several suites swap a
    MagicMock into sys.modules['torch_npu']; under that mock the deploy pass fails
    inside the mock rather than in the code under test, so these tests skip.
    """
    import sys

    module = sys.modules.get('torch_npu')
    if module is None:
        return False
    batchmatmul = getattr(module, 'npu_weight_quant_batchmatmul', None)
    schemas = getattr(batchmatmul, '_schemas', None)
    try:
        return bool(schemas) and bool(dict(schemas))
    except Exception:
        return False


class TestConvertFromQuantResult:
    """convert(model, quant_result_dir) on a float model == direct convert."""

    @pytest.fixture(autouse=True)
    def _needs_real_torch_npu(self):
        # Checked at call time, not collection time: other suites replace
        # sys.modules['torch_npu'] with a MagicMock while the session runs.
        if not _real_torch_npu():
            pytest.skip('deploy conversion requires the real torch_npu')

    @pytest.mark.parametrize('name, config, quant_cls', ALGO_CASES)
    def test_layerwise_equivalence(self, tmp_path, name, config, quant_cls):
        from amct_pytorch.classic.quantize import convert

        # Seed 3 with a fine AWQ grid actually searches a non-unit scale. Under
        # seed 0 AWQ settles on scale == 1, which makes the whole scaled-weight
        # path a no-op and any equivalence claim about it vacuous.
        torch.manual_seed(3)
        base = _model()
        # AWQ and GPTQ derive their params from the calibration data, so both
        # paths must be fed the very same batches for equivalence to mean
        # anything.
        batches = [_inputs(), _inputs()]

        # A: quantize + calibrate + convert, all in memory.
        direct = copy.deepcopy(base)
        quantize(direct, config)
        for batch in batches:
            direct(batch)
        assert isinstance(direct.fc1, quant_cls)
        if name == 'awq':
            # Guard against the vacuous case: with scale == 1 the AWQ transform
            # is the identity and this test would pass without exercising it.
            assert not torch.all(direct.fc1.scale == 1.0)
        convert(direct)

        # B: publish an artifact from a second copy, same batches.
        dumped = copy.deepcopy(base)
        out_dir = str(tmp_path / ('art_' + name))
        os.mkdir(out_dir)
        quantize(dumped, config, quant_result_dir=out_dir)
        for batch in batches:
            dumped(batch)

        # C: a still-float third copy goes straight from artifact to deploy.
        from_artifact = copy.deepcopy(base)
        assert convert(from_artifact, out_dir) is None
        _assert_same_deploy_model(direct, from_artifact)

    def test_convert_without_dir_keeps_legacy_behaviour(self, tmp_path):
        from amct_pytorch.classic.quantize import convert

        model = _model()
        quantize(model, MINMAX_CFG)
        model(_inputs())
        assert convert(model) is None
        assert type(model.fc1).__name__ != 'MinMaxQuant'

    def test_convert_rejects_non_str_dir(self):
        from amct_pytorch.classic.quantize import convert

        with pytest.raises(TypeError):
            convert(_model(), 123)

    def test_convert_rejects_a_calibrated_model(self, tmp_path):
        """The artifact path expects the original float model, not a quantized one."""
        from amct_pytorch.classic.quantize import convert

        source, _, out_dir = _artifact(tmp_path, MINMAX_CFG, name='already')
        quantize(source, MINMAX_CFG)
        with pytest.raises(ValueError, match='not the model'):
            convert(source, out_dir)

    def test_convert_propagates_artifact_errors_before_touching_model(self, tmp_path):
        from amct_pytorch.classic.quantize import convert

        source, _, out_dir = _artifact(tmp_path, MINMAX_CFG, name='bad')
        os.remove(os.path.join(out_dir, MANIFEST_NAME))
        snapshot = _snapshot(source)
        children = _child_ids(source)
        with pytest.raises(FileNotFoundError):
            convert(source, out_dir)
        _assert_unchanged(source, snapshot, children)


def _npu_available():
    """True only on a real NPU build, never on the suite-wide torch_npu mock.

    tests/conftest.py installs a MagicMock as torch_npu and as torch.npu, whose
    is_available() returns a truthy Mock. Requiring a real bool and a working
    device allocation keeps these tests skipped in mocked runs instead of failing
    inside the mock.
    """
    if not hasattr(torch, 'npu'):
        return False
    try:
        available = torch.npu.is_available()
        if available is not True:
            return False
        torch.zeros(1).npu()
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _npu_available(), reason='requires an NPU device')
class TestReloadOnNpu:
    """Units are CPU tensors on disk; a reloaded model must land back on device."""

    def test_reload_restores_params_onto_the_npu(self, tmp_path):
        torch.manual_seed(0)
        source = _model().npu()
        dumped = copy.deepcopy(source)
        out_dir = str(tmp_path / 'npu')
        os.mkdir(out_dir)
        quantize(dumped, MINMAX_CFG, quant_result_dir=out_dir)
        dumped(_inputs().npu())

        # Whatever the run device was, the artifact itself holds CPU tensors.
        tensors, _, _ = read_quant_unit(out_dir, 'fc1')
        assert all(value.device.type == 'cpu' for value in tensors.values())

        reload_quant_params(source, out_dir)
        assert source.fc1.weight.device.type == 'npu'
        assert source.fc1.scale_w.device.type == 'npu'
        source(_inputs().npu())

    def test_npu_reload_matches_direct_calibration(self, tmp_path):
        torch.manual_seed(0)
        base = _model().npu()
        batches = [_inputs().npu(), _inputs().npu()]

        dumped = copy.deepcopy(base)
        out_dir = str(tmp_path / 'npu_eq')
        os.mkdir(out_dir)
        quantize(dumped, MINMAX_CFG, quant_result_dir=out_dir)
        for batch in batches:
            dumped(batch)
        direct_out = dumped(batches[0])

        source = copy.deepcopy(base)
        reload_quant_params(source, out_dir)
        assert torch.equal(direct_out, source(batches[0]))


@pytest.mark.skipif(not _npu_available(), reason='requires an NPU device')
class TestDeployForwardOnNpu:
    """The deploy models must also run, on device, for all three algorithms."""

    @pytest.fixture(autouse=True)
    def _needs_real_torch_npu(self):
        if not _real_torch_npu():
            pytest.skip('deploy conversion requires the real torch_npu')

    @pytest.mark.parametrize('name, config, quant_cls', ALGO_CASES)
    def test_npu_deploy_forward_matches_direct(self, tmp_path, name, config, quant_cls):
        from amct_pytorch.classic.quantize import convert

        torch.manual_seed(3)
        base = _model().npu()
        batches = [_inputs().npu(), _inputs().npu()]

        direct = copy.deepcopy(base)
        quantize(direct, config)
        for batch in batches:
            direct(batch)
        assert isinstance(direct.fc1, quant_cls)
        convert(direct)

        dumped = copy.deepcopy(base)
        out_dir = str(tmp_path / ('npu_' + name))
        os.mkdir(out_dir)
        quantize(dumped, config, quant_result_dir=out_dir)
        for batch in batches:
            dumped(batch)

        from_artifact = copy.deepcopy(base)
        convert(from_artifact, out_dir)
        _assert_same_deploy_model(direct, from_artifact)

        # The real point of this test: both deploy models actually execute on the
        # device and agree there, not merely that they were constructed alike.
        probe = batches[0]
        assert torch.equal(direct(probe), from_artifact(probe))
