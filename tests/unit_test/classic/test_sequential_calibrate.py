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
"""Unit tests for sequential_calibrate orchestration and the block calibrator
registry.

These cover the four gaps the review called out: block partitioning + kind
dispatch, auto_smooth grid search + finalize, quant_result_dir persistence +
reload, and illegal-kind rejection by register_block_calibrator.
"""

import pytest
import torch
from torch import nn

from amct_pytorch.algorithms.quant.forward import (
    _has_search_contract,
    _is_forward_module,
    calibrate_forward,
)
from amct_pytorch.algorithms.quant.gradient import (
    _is_gradient_module,
    calibrate_gradient,
)
from amct_pytorch.classic.quantize import (
    _detect_blocks,
    _find_block_name,
    _resolve_blocks,
    _resolve_calib_kind,
    quantize,
    sequential_calibrate,
)
from amct_pytorch.classic.quantize_op import SmoothQuant
from amct_pytorch.common.block_calibrator import (
    FORWARD,
    GRADIENT,
    get_block_calibrator,
    register_block_calibrator,
    registered_kinds,
)


def _auto_smooth_config():
    return {
        'batch_num': 1,
        'weights_cfg': {'quant_type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs_cfg': {
            'enable_quant': True,
            'quant_type': 'int8',
            'symmetric': True,
            'strategy': 'tensor',
        },
        'algorithm': {'auto_smooth': {'grid_num': 5, 'smooth_range': [0.1, 0.9]}},
    }


def _search_block():
    """A single-Linear block exposing the auto_smooth grid-search contract."""
    module = SmoothQuant(nn.Linear(8, 8), 'proj', _auto_smooth_config())

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = module

        def forward(self, x):
            return self.proj(x)

    return Block()


class _GradientModule(nn.Module):
    """A gradient-based quantized module: quant_config + non-empty trainable_params."""

    def __init__(self):
        super().__init__()
        self.quant_config = {'algorithm': {'flatquant': {}}}
        self.value = nn.Parameter(torch.zeros(1))

    def trainable_params(self):
        return [self.value]


# ---------------------------------------------------------------------------
# register_block_calibrator: kind must be 'forward' or 'gradient'.
# ---------------------------------------------------------------------------


def test_register_block_calibrator_rejects_invalid_kind():
    with pytest.raises(ValueError, match="Invalid block calibrator kind 'bogus'"):
        register_block_calibrator('bogus')


def test_register_block_calibrator_accepts_only_forward_and_gradient():
    assert FORWARD == 'forward'
    assert GRADIENT == 'gradient'
    # forward/gradient calibrators are registered by import side effects.
    assert get_block_calibrator(FORWARD) is not None
    assert get_block_calibrator(GRADIENT) is not None
    assert set(registered_kinds()) == {FORWARD, GRADIENT}


# ---------------------------------------------------------------------------
# Block partitioning helpers.
# ---------------------------------------------------------------------------


def test_find_block_name_uses_nearest_numeric_segment():
    assert _find_block_name('model.layers.3.self_attn.q_proj') == 'model.layers.3'
    assert _find_block_name('encoder.layers.0.mlp.fc1') == 'encoder.layers.0'
    assert _find_block_name('fc1') == 'fc1'


def test_detect_blocks_groups_by_inferred_block_name():
    names = ['layers.0.fc1', 'layers.0.fc2', 'layers.1.fc1']
    groups = _detect_blocks(names)
    assert groups == {
        'layers.0': ['layers.0.fc1', 'layers.0.fc2'],
        'layers.1': ['layers.1.fc1'],
    }


def test_resolve_blocks_accepts_explicit_list_and_dict():
    names = ['layers.0.fc1', 'layers.0.fc2', 'layers.1.fc1']
    assert _resolve_blocks(names, ['layers.0']) == {
        'layers.0': ['layers.0.fc1', 'layers.0.fc2']
    }
    assert _resolve_blocks(names, {'layers.0': ['layers.0.fc1']}) == {
        'layers.0': ['layers.0.fc1']
    }


# ---------------------------------------------------------------------------
# Kind dispatch and matchers.
# ---------------------------------------------------------------------------


def test_resolve_calib_kind_dispatches_forward_vs_gradient():
    forward_module = SmoothQuant(nn.Linear(8, 8), 'proj', _auto_smooth_config())
    assert _resolve_calib_kind(forward_module) == FORWARD
    assert _resolve_calib_kind(_GradientModule()) == GRADIENT
    assert _resolve_calib_kind(nn.Linear(8, 8)) is None


def test_forward_matcher_excludes_shared_and_gradient_modules():
    forward_module = SmoothQuant(nn.Linear(8, 8), 'proj', _auto_smooth_config())
    assert _is_forward_module(forward_module) is True

    forward_module.shared_param_mode = 'static'
    assert _is_forward_module(forward_module) is False

    assert _is_forward_module(_GradientModule()) is False
    assert _is_forward_module(nn.Linear(8, 8)) is False


def test_gradient_matcher_requires_non_empty_trainable_params():
    assert _is_gradient_module(_GradientModule()) is True

    empty = _GradientModule()
    empty.trainable_params = list
    assert _is_gradient_module(empty) is False

    assert _is_gradient_module(nn.Linear(8, 8)) is False


def test_has_search_contract_requires_all_three_methods():
    module = SmoothQuant(nn.Linear(8, 8), 'proj', _auto_smooth_config())
    assert _has_search_contract(module) is True

    class NoContract(nn.Module):
        pass

    class PartialContract(nn.Module):
        def calib_param_grid(self):
            return []

    assert _has_search_contract(NoContract()) is False
    assert _has_search_contract(PartialContract()) is False


# ---------------------------------------------------------------------------
# auto_smooth grid-search contract + finalize.
# ---------------------------------------------------------------------------


def test_smoothquant_search_contract_and_finalize():
    module = SmoothQuant(nn.Linear(8, 8), 'proj', _auto_smooth_config())

    grid = module.calib_param_grid()
    assert len(grid) == 5
    assert grid[0] == pytest.approx(0.1)
    assert grid[-1] == pytest.approx(0.9)

    module(torch.randn(4, 8))  # accumulate activation statistics
    module.apply_calib_param(0.5)
    assert module.finalized is True
    assert module.smooth_strength == 0.5
    assert module.scale is not None  # smooth_factor
    assert module.scale_w is not None

    module.reset_calib()
    assert module.finalized is False


def test_calibrate_forward_grid_search_selects_best_alpha():
    block = _search_block()
    x = torch.randn(4, 8)
    block(x)  # calibration forward accumulates statistics

    quant_modules = [block.proj]
    best = calibrate_forward(
        model=None,
        block_name='layers.0',
        block=block,
        quant_modules=quant_modules,
        block_inputs=[(x,)],
        _verbose=False,
    )

    assert best is not None
    assert 0.1 <= best <= 0.9
    assert block.proj.finalized is True
    assert block.proj.smooth_strength == best


def test_calibrate_gradient_raises_not_implemented():
    with pytest.raises(NotImplementedError, match='FlatQuant'):
        calibrate_gradient(None, 'layers.0', None, [], [])


# ---------------------------------------------------------------------------
# sequential_calibrate: quant_result_dir persistence + reload.
# ---------------------------------------------------------------------------


_MINMAX_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs': {'type': 'int8', 'symmetric': True, 'strategy': 'tensor'},
    },
    'algorithm': {'minmax'},
}


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, inputs):
        return self.fc2(self.fc1(inputs))


def test_sequential_calibrate_persists_and_reloads_quant_result(tmp_path):
    from amct_pytorch.classic.quant_result import (
        load_quant_manifest,
        read_quant_unit,
        validate_result_directory,
    )

    model = _TinyModel().to(torch.bfloat16)
    quantize(model, _MINMAX_CFG)

    out = tmp_path / 'out'
    out.mkdir()
    sequential_calibrate(
        model, [torch.randn(4, 8).to(torch.bfloat16)], quant_result_dir=str(out)
    )

    manifest = load_quant_manifest(str(out))
    assert set(manifest) == {
        'format',
        'schema_version',
        'artifact_id',
        'source_model',
        'units',
    }
    assert sorted(unit['layer_name'] for unit in manifest['units']) == ['fc1', 'fc2']
    validate_result_directory(str(out), manifest)

    # Every unit is loadable through the reader (the reload contract).
    for unit in manifest['units']:
        tensors, _, _ = read_quant_unit(str(out), unit['layer_name'], manifest)
        assert tensors  # MinMaxQuant dumps at least its weight scale


_AUTO_SMOOTH_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs': {'type': 'int8', 'symmetric': True, 'strategy': 'tensor'},
    },
    'algorithm': {'auto_smooth': {'grid_num': 9, 'smooth_range': [0.1, 0.9]}},
}


def test_convert_rejects_uncalibrated_auto_smooth():
    from amct_pytorch.classic.quantize import convert

    model = _TinyModel().to(torch.bfloat16)
    quantize(model, _AUTO_SMOOTH_CFG)
    # Legacy quantize -> forward flow does not finalize auto_smooth layers.
    model(torch.randn(4, 8).to(torch.bfloat16))

    with pytest.raises(ValueError, match='sequential_calibrate'):
        convert(model)


def test_check_auto_smooth_calibrated_passes_after_finalize():
    from amct_pytorch.classic.quantize import _check_auto_smooth_calibrated

    module = SmoothQuant(nn.Linear(8, 8), 'proj', _auto_smooth_config())

    class Model(nn.Module):
        def __init__(self, member):
            super().__init__()
            self.proj = member

    model = Model(module)
    with pytest.raises(ValueError, match='sequential_calibrate'):
        _check_auto_smooth_calibrated(model)

    module.finalized = True
    _check_auto_smooth_calibrated(model)


def test_quantize_rejects_auto_smooth_quant_result_dir(tmp_path):
    model = _TinyModel().to(torch.bfloat16)
    out = tmp_path / 'out'
    out.mkdir()

    with pytest.raises(ValueError, match='quant_result_dir'):
        quantize(model, _AUTO_SMOOTH_CFG, quant_result_dir=str(out))

    # Nothing must be written when the combination is rejected up front.
    assert list(out.iterdir()) == []


def test_iter_tensors_collects_nested_kwargs_tensors():
    from amct_pytorch.algorithms.quant.forward import _iter_tensors

    a = torch.zeros(2)
    b = torch.ones(3)
    c = torch.empty(4)
    assert list(_iter_tensors([a, {'pos': (b, c)}])) == [a, b, c]
    assert list(_iter_tensors([a, [b], {'k': c}])) == [a, b, c]
    assert list(_iter_tensors(a)) == [a]
    assert list(_iter_tensors({'x': 1})) == []


def test_calibrate_forward_rejects_nan_in_keyword_input():
    block = _search_block()
    x = torch.randn(4, 8)
    block(x)  # calibration forward accumulates statistics

    pos = torch.randn(4, 8)
    pos[0, 0] = float('nan')
    # Qwen3-style keyword input: position_embeddings nested in a tuple.
    block_input = ((x,), {'position_embeddings': (pos,)})

    with pytest.raises(RuntimeError, match='Invalid value'):
        calibrate_forward(
            model=None,
            block_name='layers.0',
            block=block,
            quant_modules=[block.proj],
            block_inputs=[block_input],
            _verbose=False,
        )


def test_reset_calib_marks_stats_finished_so_replay_does_not_recount():
    module = SmoothQuant(nn.Linear(8, 8), 'proj', _auto_smooth_config())
    x = torch.randn(4, 8)

    module(x)
    assert module.cur_batch == 1

    module.reset_calib()
    assert module.finalized is False
    assert module._stats_finished is True

    # Replay (grid search) must not bump the statistics state.
    cur_batch_before = module.cur_batch
    batch_input_before = module.batch_input
    module(x)
    assert module.cur_batch == cur_batch_before
    assert module.batch_input is batch_input_before


def test_get_submodule_raises_value_error_for_missing_block():
    from amct_pytorch.classic.quantize import _get_submodule

    model = _TinyModel()
    assert _get_submodule(model, '') is model
    with pytest.raises(ValueError, match='not a submodule'):
        _get_submodule(model, 'no_such_block')


def test_sequential_calibrate_warns_uncovered_layers():
    from unittest.mock import patch

    import amct_pytorch.common.utils.log as log_module

    model = _TinyModel().to(torch.bfloat16)
    quantize(model, _MINMAX_CFG)

    warnings = []
    with patch.object(
        log_module.LOGGER,
        'logw',
        side_effect=lambda msg, *args, **kwargs: warnings.append(msg),
    ):
        sequential_calibrate(
            model,
            [torch.randn(4, 8).to(torch.bfloat16)],
            blocks=['fc1'],
        )

    assert any('uncovered' in msg for msg in warnings)


def test_sequential_calibrate_warns_when_block_captures_no_input():
    from unittest.mock import patch

    import amct_pytorch.common.utils.log as log_module

    model = _TinyModel().to(torch.bfloat16)
    quantize(model, _MINMAX_CFG)

    warnings = []
    with patch.object(
        log_module.LOGGER,
        'logw',
        side_effect=lambda msg, *args, **kwargs: warnings.append(msg),
    ):
        sequential_calibrate(model, [])  # 空校准集：所有 block 均无捕获输入

    assert any('captured no input' in msg for msg in warnings)
    assert any("block 'fc1'" in msg for msg in warnings)


def test_sequential_calibrate_rejects_invalid_blocks_type():
    model = _TinyModel().to(torch.bfloat16)
    quantize(model, _MINMAX_CFG)

    with pytest.raises(TypeError, match='blocks'):
        sequential_calibrate(
            model,
            [torch.randn(4, 8).to(torch.bfloat16)],
            blocks='fc1',
        )


def test_sequential_calibrate_rejects_unknown_layer_in_blocks_dict():
    model = _TinyModel().to(torch.bfloat16)
    quantize(model, _MINMAX_CFG)

    with pytest.raises(ValueError, match='calibratable set'):
        sequential_calibrate(
            model,
            [torch.randn(4, 8).to(torch.bfloat16)],
            blocks={'fc1': ['nonexistent_or_excluded']},
        )


def test_per_token_params_follow_actual_batch_columns():
    cfg = {
        'batch_num': 4,
        'weights_cfg': {'quant_type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs_cfg': {
            'enable_quant': True,
            'quant_type': 'int8',
            'symmetric': True,
            'strategy': 'token',
        },
        'algorithm': {'auto_smooth': {'grid_num': 5, 'smooth_range': [0.1, 0.9]}},
    }
    module = SmoothQuant(nn.Linear(8, 8), 'proj', cfg)

    # batch_num=4 but only one calibration sample is fed: the per-token scale
    # must follow the actual batch_input columns, not the configured batch_num.
    module(torch.randn(4, 8))
    assert module.cur_batch == 1
    assert module.batch_input.shape == (4, 8)

    module.apply_calib_param(0.5)
    assert module.finalized is True
    assert module.scale_d is not None


def test_calibrate_forward_rejects_inf_loss():
    from unittest.mock import patch

    from amct_pytorch.algorithms.quant import forward as forward_module
    from amct_pytorch.algorithms.quant.forward import calibrate_forward

    block = _search_block()
    x = torch.randn(4, 8)
    block(x)

    with (
        patch.object(
            forward_module.F, 'mse_loss', return_value=torch.tensor(float('inf'))
        ),
        pytest.raises(RuntimeError, match='invalid loss'),
    ):
        calibrate_forward(
            model=None,
            block_name='layers.0',
            block=block,
            quant_modules=[block.proj],
            block_inputs=[(x,)],
            _verbose=False,
        )


def test_calibrate_forward_rejects_mismatched_grids_in_one_block():
    from amct_pytorch.algorithms.quant.forward import calibrate_forward

    def cfg(grid_num, smooth_range):
        return {
            'batch_num': 1,
            'weights_cfg': {
                'quant_type': 'int8',
                'symmetric': True,
                'strategy': 'channel',
            },
            'inputs_cfg': {
                'enable_quant': True,
                'quant_type': 'int8',
                'symmetric': True,
                'strategy': 'tensor',
            },
            'algorithm': {
                'auto_smooth': {'grid_num': grid_num, 'smooth_range': smooth_range}
            },
        }

    a = SmoothQuant(nn.Linear(8, 8), 'a', cfg(3, [0.1, 0.3]))
    b = SmoothQuant(nn.Linear(8, 8), 'b', cfg(5, [0.6, 0.9]))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = a
            self.b = b

        def forward(self, x):
            return self.b(self.a(x))

    block = Block()
    block(torch.randn(4, 8))

    with pytest.raises(ValueError, match='consistent within a block'):
        calibrate_forward(
            model=None,
            block_name='layers.0',
            block=block,
            quant_modules=[a, b],
            block_inputs=[(torch.randn(4, 8),)],
            _verbose=False,
        )


def test_fixed_smoothquant_is_not_grid_searched():
    from amct_pytorch.algorithms.quant.forward import calibrate_forward

    cfg = {
        'batch_num': 1,
        'weights_cfg': {'quant_type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs_cfg': {
            'enable_quant': True,
            'quant_type': 'int8',
            'symmetric': True,
            'strategy': 'tensor',
        },
        'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
    }
    module = SmoothQuant(nn.Linear(8, 8), 'proj', cfg)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = module

        def forward(self, x):
            return self.proj(x)

    block = Block()
    assert module.is_auto_smooth is False

    # Fixed mode finalizes automatically once batch_num is reached.
    block(torch.randn(4, 8))
    assert module.smooth_strength == 0.5

    # A fixed smoothquant layer must not be grid-searched (its smooth_strength
    # is user-configured and must not be overwritten by the block grid search).
    best = calibrate_forward(
        model=None,
        block_name='layers.0',
        block=block,
        quant_modules=[module],
        block_inputs=[(torch.randn(4, 8),)],
        _verbose=False,
    )

    assert best is None
    assert module.smooth_strength == 0.5
    assert module.finalized is True


def test_sequential_calibrate_replays_one_shot_generator_per_block():
    model = _TinyModel().to(torch.bfloat16)
    quantize(model, _AUTO_SMOOTH_CFG)

    def gen():
        yield torch.randn(4, 8).to(torch.bfloat16)

    # A generator is a one-shot iterable: without materialization the first
    # block would consume it and fc2 would silently skip calibration.
    sequential_calibrate(model, gen())

    calibratable = [
        module
        for module in model.modules()
        if isinstance(module, SmoothQuant) and module.is_auto_smooth
    ]
    assert len(calibratable) == 2
    for module in calibratable:
        assert module.finalized is True
        assert module.smooth_strength is not None
