#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""AMCT structured-pruning core tests (consolidated): comprehensive HuggingFace-style
mini-model tests (MLP / CNN / MoE), previously-uncovered core-line unit tests
(config validation/normalization, batch adapters, calib logit extraction, finetune
helpers, backend-compat helpers, paramless MoE router patch), and public-surface
coverage (default config path, activation_count(MoE), FULL_STRUCTURED preset,
per_layer_sparsity, evaluator unification, diagnose).
"""

import copy
import logging
import os
import shutil
import sys
import tempfile
import unittest

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp, create_mini_cnn, create_mini_moe
from amct_pytorch.pruning import (
    DENSE_LOWVAR_PRUNE_CFG,
    CNN_VARIANCE_PRUNE_CFG,
    CNN_RECONSTRUCT_PRUNE_CFG,
    MOE_MASSVAR_PRUNE_CFG,
    prune,
    PruneReport,
    FULL_STRUCTURED_PRUNE_CFG,
    prune_diagnose,
    prune_finetune,
)
from amct_pytorch.pruning.accuracy_based_auto_prune import (
    _accuracy_based_auto_prune as accuracy_based_auto_prune,
)
from amct_pytorch.pruning.config import (
    MethodSpec,
    PruneConfig,
    _normalize_method_spec,
    _validate_allocation,
    _validate_method_kwargs,
)
from amct_pytorch.pruning.context import PruneContext, default_batch_adapter
from amct_pytorch.pruning.calib import _unwrap_logits, calib_nll
from amct_pytorch.pruning.finetune import (
    _default_loss,
    _move_to_device,
    _set_warmup_lr,
)
from amct_pytorch.pruning.compat import (
    detect_backend,
    _try_set_int_attr,
    patch_common_config,
)
from amct_pytorch.pruning.prune_op._moe_mass_common import patch_router_module
from amct_pytorch.pruning.prune_op.base import BasePruningMethod
from amct_pytorch.pruning.domains.moe import MoETarget
from amct_pytorch.pruning.domains.dense import DensePruningDomain
from amct_pytorch.pruning.diagnostics import DiagnosisReport
from amct_pytorch.pruning.pruner import AutoPruner
from amct_pytorch.pruning.registry import (
    DomainMethodBinding,
    create_default_registry,
)
from amct_pytorch.pruning import presets
from amct_pytorch.pruning.presets import MOE_VARIANCE_MENU_CFG

torch.set_num_threads(2)


def _calib_tokens():
    return [torch.randint(0, 1000, (4, 20)) for _ in range(8)]


def _tok(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


def _params(m):
    return sum(p.numel() for p in m.parameters())


class TestPruneRatios(unittest.TestCase):
    def test_increasing_ratio_increases_reduction(self):
        calib = _calib_tokens()
        reductions = []
        for ratio in (0.3, 0.5, 0.7):
            model, _ = create_mini_mlp()
            model.eval()
            cfg = {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": ratio}}
                },
                "missing_data_policy": "warn_skip",
            }
            report = PruneReport()
            prune(model, cfg, data=calib, report=report)
            self.assertIsNotNone(report)
            self.assertLess(report.params_after, report.params_before)
            reductions.append(1 - report.params_after / report.params_before)
        self.assertLess(reductions[0], reductions[1])
        self.assertLess(reductions[1], reductions[2])


class TestSaveLoadRoundtrip(unittest.TestCase):
    def test_state_dict_roundtrip_matches(self):
        calib = _calib_tokens()
        original_model, config = create_mini_mlp()
        original_model.eval()
        report = PruneReport()
        prune(original_model, DENSE_LOWVAR_PRUNE_CFG, data=calib, report=report)
        self.assertLess(report.params_after, report.params_before)

        test_input = torch.randint(0, 1000, (2, 20))
        with torch.no_grad():
            original_output = original_model(test_input)

        temp_dir = tempfile.mkdtemp(prefix="amct_mlp_")
        try:
            save_path = os.path.join(temp_dir, "pruned_mlp.pth")
            torch.save(
                {
                    "model_state_dict": original_model.state_dict(),
                    "config": config.__dict__,
                    "pruning_info": {
                        "params_before": report.params_before,
                        "params_after": report.params_after,
                    },
                },
                save_path,
            )

            loaded_model, _ = create_mini_mlp()
            prune(loaded_model, DENSE_LOWVAR_PRUNE_CFG, data=calib)
            checkpoint = torch.load(save_path, weights_only=True)
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            loaded_model.eval()
            with torch.no_grad():
                loaded_output = loaded_model(test_input)
        finally:
            shutil.rmtree(temp_dir)

        output_diff = torch.abs(original_output - loaded_output).max().item()
        self.assertLess(output_diff, 1e-6)


class TestDomainPruning(unittest.TestCase):
    def test_mlp_dense_pruning(self):
        model, config = create_mini_mlp()
        model.eval()
        report = PruneReport()
        prune(model, DENSE_LOWVAR_PRUNE_CFG, data=_calib_tokens(), report=report)
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))

    def test_reconstruct_preserves_output_better_than_low_variance(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        calib = [torch.randint(0, 1000, (8, 32)) for _ in range(40)]
        eval_b = torch.randint(0, 1000, (4, 24))
        with torch.no_grad():
            ref = model(eval_b)

        def rel_err(name):
            m = copy.deepcopy(model)
            prune(
                m,
                {"methods": {"dense": {"name": name, "kwargs": {"prune_ratio": 0.4}}}},
                data=calib,
            )
            with torch.no_grad():
                out = m(eval_b)
            return ((out - ref).norm() / ref.norm()).item()

        e_lv = rel_err("low_variance")
        e_rec = rel_err("reconstruct")
        self.assertLess(e_rec, e_lv)

    def test_conv1d_ffn_reconstruct_prunes_and_preserves(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

        class Conv1D(nn.Module):
            def __init__(self, nf, nx):
                super().__init__()
                self.nf = nf
                self.weight = nn.Parameter(torch.randn(nx, nf) * 0.02)
                self.bias = nn.Parameter(torch.zeros(nf))

            def forward(self, x):
                return x @ self.weight + self.bias

        class Block(nn.Module):
            def __init__(self, h=32):
                super().__init__()
                self.c_fc = Conv1D(4 * h, h)
                self.c_proj = Conv1D(h, 4 * h)

            def forward(self, x):
                return x + self.c_proj(F.gelu(self.c_fc(x)))

        model = Block().eval()
        calib = [torch.randn(8, 16, 32) for _ in range(20)]
        x = torch.randn(4, 16, 32)
        with torch.no_grad():
            ref = model(x)
        p0 = sum(p.numel() for p in model.parameters())
        prune(
            model,
            {
                "methods": {
                    "dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.4}}
                }
            },
            data=calib,
        )
        p1 = sum(p.numel() for p in model.parameters())
        self.assertLess(p1, p0)
        with torch.no_grad():
            out = model(x)
        self.assertLess(((out - ref).norm() / ref.norm()).item(), 0.25)

    def test_quant_aware_saliency_valid_and_distinct(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        calib = [torch.randint(0, 1000, (8, 32)) for _ in range(40)]
        probe = torch.randint(0, 1000, (2, 20))

        def run(**kw):
            m = copy.deepcopy(model)
            prune(
                m,
                {
                    "methods": {
                        "dense": {
                            "name": "reconstruct",
                            "kwargs": {"prune_ratio": 0.5, **kw},
                        }
                    }
                },
                data=calib,
            )
            with torch.no_grad():
                return m, m(probe)

        p_before = sum(p.numel() for p in model.parameters())
        m_blind, o_blind = run()
        m_aware, o_aware = run(quant_aware=True, quant_aware_bits=2)
        self.assertLess(sum(p.numel() for p in m_aware.parameters()), p_before)
        self.assertGreater((o_blind - o_aware).abs().max().item(), 1e-5)

    def test_cnn_channel_pruning(self):
        model, config = create_mini_cnn()
        model.eval()
        calib = [torch.randn(2, 3, 32, 32) for _ in range(8)]
        report = PruneReport()
        prune(model, CNN_VARIANCE_PRUNE_CFG, data=calib, report=report)
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(out.shape, (2, config.num_classes))

    def test_cnn_reconstruct_preset(self):
        model, config = create_mini_cnn()
        model.eval()
        calib = [torch.randn(2, 3, 32, 32) for _ in range(8)]
        report = PruneReport()
        prune(model, CNN_RECONSTRUCT_PRUNE_CFG, data=calib, report=report)
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(out.shape, (2, config.num_classes))

    def test_cnn_reconstruct_prunes_and_forwards(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        model, config = create_mini_cnn()
        model.eval()
        calib = [torch.randn(4, 3, 32, 32) for _ in range(8)]
        p0 = sum(p.numel() for p in model.parameters())

        def pruned_params(name):
            m = copy.deepcopy(model)
            prune(
                m,
                {"methods": {"cnn": {"name": name, "kwargs": {"prune_ratio": 0.4}}}},
                data=calib,
            )
            with torch.no_grad():
                out = m(torch.randn(2, 3, 32, 32))
            self.assertEqual(out.shape, (2, config.num_classes))
            return sum(p.numel() for p in m.parameters())

        n_rc = pruned_params("reconstruct")
        n_var = pruned_params("variance_channel")
        self.assertLess(n_rc, p0)
        self.assertEqual(n_rc, n_var)

    def test_moe_expert_pruning(self):
        model, config = create_mini_moe()
        model.eval()
        calib = [torch.randint(0, 1000, (4, 20)) for _ in range(8)]
        report = PruneReport()
        prune(model, MOE_MASSVAR_PRUNE_CFG, data=calib, report=report)
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))


class _FusedMlp(torch.nn.Module):
    """Phi-3/GLM-4 style fused FFN: gate_up_proj output is gate (first half) + up (second half)."""

    def __init__(self, hidden=32, inter=48):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(hidden, 2 * inter)
        self.down_proj = torch.nn.Linear(inter, hidden)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class _ParamlessAct(torch.nn.Module):
    """Bloom-style custom activation (parameterless module); must not break the two-layer FFN chain."""

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class _TwoLayerCustomActMlp(torch.nn.Module):
    def __init__(self, hidden=32, inter=48):
        super().__init__()
        self.dense_h_to_4h = torch.nn.Linear(hidden, inter)
        self.act = _ParamlessAct()
        self.dense_4h_to_h = torch.nn.Linear(inter, hidden)

    def forward(self, x):
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))


class TestDenseLayoutVariants(unittest.TestCase):
    CFG = {
        "methods": {"dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.5}}},
        "missing_data_policy": "warn_skip",
    }

    def test_fused_gate_up_pruned(self):
        model = _FusedMlp().eval()
        self._run(model)
        self.assertEqual(
            model.gate_up_proj.out_features, 2 * model.down_proj.in_features
        )

    def test_two_layer_custom_activation_pruned(self):
        self._run(_TwoLayerCustomActMlp().eval())

    def test_quant_cfg_amct_scale_saliency(self):
        import warnings

        from amct_pytorch.pruning.prune_op.dense_reconstruct import (
            _fake_quant_amct,
        )

        w = torch.randn(16, 32)
        cfg = {
            "weights_cfg": {"strategy": "channel", "symmetric": True, "dtype": "int8"}
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wq = _fake_quant_amct(w, cfg)
        self.assertLess(float((wq - w).abs().max() / w.abs().max()), 0.02)
        model = _TwoLayerCustomActMlp().eval()
        prune(
            model,
            {
                "methods": {
                    "dense": {
                        "name": "reconstruct",
                        "kwargs": {"prune_ratio": 0.5, "quant_cfg": cfg},
                    }
                },
                "missing_data_policy": "warn_skip",
            },
            data=[torch.randn(4, 32) for _ in range(4)],
        )
        self.assertEqual(model.dense_h_to_4h.out_features, 24)

    def _run(self, model, hidden=32):
        calib = [torch.randn(4, hidden) for _ in range(4)]
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, self.CFG, data=calib)
        self.assertLess(sum(p.numel() for p in model.parameters()), p0)
        with torch.no_grad():
            out = model(torch.randn(2, hidden))
        self.assertEqual(out.shape, (2, hidden))


def test_normalize_from_methodspec_clones_kwargs():
    src = MethodSpec(name="m", kwargs={"prune_ratio": 0.5})
    out = _normalize_method_spec(src)
    assert isinstance(out, MethodSpec)
    assert out.name == "m"
    assert out.kwargs == {"prune_ratio": 0.5}
    assert out.kwargs is not src.kwargs


def test_normalize_from_str():
    out = _normalize_method_spec("low_variance")
    assert out.name == "low_variance"
    assert out.kwargs == {}


def test_normalize_from_dict_merges_extra_keys():
    out = _normalize_method_spec(
        {"name": "activation_count", "kwargs": {"a": 1}, "prune_ratio": 0.3}
    )
    assert out.name == "activation_count"
    assert out.kwargs == {"a": 1, "prune_ratio": 0.3}


def test_normalize_dict_missing_name_raises():
    with pytest.raises(ValueError):
        _normalize_method_spec({"kwargs": {"prune_ratio": 0.1}})


def test_normalize_bad_type_raises():
    with pytest.raises(TypeError):
        _normalize_method_spec(123)


def test_validate_kwargs_empty_name_raises():
    with pytest.raises(ValueError):
        _validate_method_kwargs(MethodSpec(name="", kwargs={}))


def test_validate_kwargs_non_numeric_prune_ratio_raises():
    with pytest.raises(ValueError):
        _validate_method_kwargs(MethodSpec(name="m", kwargs={"prune_ratio": "abc"}))


def test_validate_kwargs_out_of_range_prune_ratio_raises():
    with pytest.raises(ValueError):
        _validate_method_kwargs(MethodSpec(name="m", kwargs={"prune_ratio": 1.0}))


def test_validate_kwargs_unknown_key_raises_for_a_known_method():
    with pytest.raises(ValueError):
        _validate_method_kwargs(
            MethodSpec(name="low_variance", kwargs={"prune_rate": 0.5}), "dense"
        )


def test_validate_kwargs_good_values_ok():
    _validate_method_kwargs(MethodSpec(name="m", kwargs={"prune_ratio": 0.5}))


def test_validate_allocation_bad_type_raises():
    with pytest.raises(TypeError):
        _validate_allocation([("strategy", "uniform")])


def test_validate_allocation_ratio_out_of_range_raises():
    with pytest.raises(ValueError):
        _validate_allocation({"ref_ratio": 1.5})


def test_validate_min_channels_raises():
    with pytest.raises(ValueError):
        PruneConfig(min_channels=0).validate()


def test_validate_min_neurons_raises():
    with pytest.raises(ValueError):
        PruneConfig(min_neurons=0).validate()


def test_validate_min_experts_raises():
    with pytest.raises(ValueError):
        PruneConfig(min_experts=0).validate()


def test_validate_bad_missing_data_policy_raises():
    with pytest.raises(ValueError):
        PruneConfig(missing_data_policy="explode").validate()


def test_validate_bad_stage_error_policy_raises():
    with pytest.raises(ValueError):
        PruneConfig(stage_error_policy="explode").validate()


def test_validate_good_config_ok():
    PruneConfig(methods={"dense": "low_variance"}).validate()


def test_iter_model_inputs_none_data_empty():
    ctx = PruneContext(data=None)
    assert list(ctx.iter_model_inputs()) == []


def test_default_adapter_dict_to_kwargs():
    args, kwargs = default_batch_adapter({"input_ids": torch.zeros(2)})
    assert args == ()
    assert "input_ids" in kwargs


def test_default_adapter_tensor():
    t = torch.zeros(3)
    args, kwargs = default_batch_adapter(t)
    assert len(args) == 1 and torch.is_tensor(args[0])
    assert kwargs == {}


def test_default_adapter_tuple():
    args, kwargs = default_batch_adapter((1, 2))
    assert args == (1, 2)
    assert kwargs == {}


def test_default_adapter_list():
    args, kwargs = default_batch_adapter([1, 2, 3])
    assert args == (1, 2, 3)
    assert kwargs == {}


def test_default_adapter_unsupported_raises():
    with pytest.raises(TypeError):
        default_batch_adapter(object())


def test_iter_model_inputs_uses_adapter():
    ctx = PruneContext(data=[torch.zeros(2), torch.ones(2)])
    out = list(ctx.iter_model_inputs())
    assert len(out) == 2
    assert all(isinstance(a, tuple) and isinstance(k, dict) for a, k in out)


def test_unwrap_logits_tensor_passthrough():
    t = torch.zeros(2, 3)
    assert _unwrap_logits(t) is t


def test_unwrap_logits_from_tuple():
    t = torch.zeros(2, 3)
    assert _unwrap_logits((t, "extra")) is t


def test_unwrap_logits_from_list():
    t = torch.zeros(2, 3)
    assert _unwrap_logits([t]) is t


def test_unwrap_logits_from_attr():
    class Out:
        def __init__(self, logits):
            self.logits = logits

    t = torch.zeros(2, 3)
    assert _unwrap_logits(Out(t)) is t


def test_unwrap_logits_bad_raises():
    with pytest.raises(TypeError):
        _unwrap_logits(object())


def test_calib_nll_empty_batches_raises():
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        calib_nll(model, [])


class _TokenLM(nn.Module):
    """Tiny LM: embed token ids -> logits over vocab."""

    def __init__(self, vocab=8, hidden=6):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, ids):
        return self.head(self.emb(ids))


def test_calib_nll_runs_on_integer_batches():
    model = _TokenLM()
    batches = [torch.randint(0, 8, (2, 5)) for _ in range(2)]
    val = calib_nll(model, batches)
    assert isinstance(val, float) and val >= 0.0


def test_calib_nll_float_batch_raises():
    model = _TokenLM()
    with pytest.raises(ValueError):
        calib_nll(model, [torch.randn(2, 5)])


class _LossOut:
    def __init__(self, loss=None):
        self.loss = loss


def test_default_loss_via_batch_adapter_ok():
    loss_tensor = torch.tensor(1.0, requires_grad=True)

    class ToyLossModel(nn.Module):
        def forward(self, *a, **k):
            return _LossOut(loss=loss_tensor)

    def adapter(batch):
        return (), {"x": batch}

    out = _default_loss(ToyLossModel(), torch.zeros(2), adapter)
    assert out is loss_tensor


def test_default_loss_via_batch_adapter_no_loss_raises():
    class ToyLossModel(nn.Module):
        def forward(self, *a, **k):
            return _LossOut(loss=None)

    def adapter(batch):
        return (), {}

    with pytest.raises(ValueError):
        _default_loss(ToyLossModel(), torch.zeros(2), adapter)


def test_default_loss_dict_labels():
    captured = {}

    class ToyLossModel(nn.Module):
        def forward(self, **batch):
            captured.update(batch)
            return _LossOut(loss=torch.tensor(2.0))

    batch = {"input_ids": torch.zeros(2), "labels": torch.zeros(2)}
    out = _default_loss(ToyLossModel(), batch, None)
    assert float(out) == 2.0
    assert "labels" in captured


def test_default_loss_dict_input_ids_only():
    seen = {}

    class ToyLossModel(nn.Module):
        def forward(self, **batch):
            seen.update(batch)
            return _LossOut(loss=torch.tensor(3.0))

    batch = {"input_ids": torch.zeros(2)}
    out = _default_loss(ToyLossModel(), batch, None)
    assert float(out) == 3.0
    assert "labels" in seen


def test_default_loss_dict_missing_keys_raises():
    class ToyLossModel(nn.Module):
        def forward(self, **batch):
            return _LossOut(loss=torch.tensor(0.0))

    with pytest.raises(ValueError):
        _default_loss(ToyLossModel(), {"foo": torch.zeros(2)}, None)


def test_default_loss_xy_tuple_classification():
    model = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    out = _default_loss(model, (x, y), None)
    assert out.ndim == 0


class _LogitsOut:
    def __init__(self, logits):
        self.logits = logits


def test_default_loss_xy_tuple_with_logits_attr():
    class ToyLogitsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 3)

        def forward(self, x):
            return _LogitsOut(self.lin(x))

    m = ToyLogitsModel()
    out = _default_loss(m, (torch.randn(2, 4), torch.tensor([0, 1])), None)
    assert out.ndim == 0


def test_default_loss_unsupported_raises():
    with pytest.raises(ValueError):
        _default_loss(nn.Linear(2, 2), 12345, None)


def test_move_to_device_dict():
    cpu = torch.device("cpu")
    batch = {"a": torch.zeros(2), "b": "str"}
    moved = _move_to_device(batch, cpu)
    assert torch.is_tensor(moved["a"])
    assert moved["b"] == "str"


def test_move_to_device_tuple_and_list():
    cpu = torch.device("cpu")
    out_t = _move_to_device((torch.zeros(2), torch.ones(2)), cpu)
    assert isinstance(out_t, tuple) and all(torch.is_tensor(x) for x in out_t)
    out_l = _move_to_device([torch.zeros(2)], cpu)
    assert isinstance(out_l, list)


def test_move_to_device_tensor_and_passthrough():
    cpu = torch.device("cpu")
    t = torch.zeros(2)
    assert torch.is_tensor(_move_to_device(t, cpu))
    assert _move_to_device("scalar", cpu) == "scalar"


def test_set_warmup_lr_scales():
    params = [torch.zeros(1, requires_grad=True)]
    opt = torch.optim.SGD(params, lr=0.1)
    _set_warmup_lr(opt, lr=0.1, step=0, warmup=4)
    assert abs(opt.param_groups[0]["lr"] - 0.1 * 1 / 4) < 1e-9


def test_set_warmup_lr_noop_when_done():
    params = [torch.zeros(1, requires_grad=True)]
    opt = torch.optim.SGD(params, lr=0.1)
    _set_warmup_lr(opt, lr=0.1, step=10, warmup=4)
    assert opt.param_groups[0]["lr"] == 0.1


def test_prune_finetune_empty_data_raises():
    with pytest.raises(ValueError):
        prune_finetune(nn.Linear(2, 2), [], steps=1)


def test_prune_finetune_no_trainable_params_raises():
    model = nn.Linear(2, 2)
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError):
        prune_finetune(model, [(torch.randn(2, 2), torch.tensor([0, 1]))], steps=1)


def test_prune_finetune_runs_with_warmup():
    model = nn.Linear(4, 3)
    data = [(torch.randn(2, 4), torch.tensor([0, 1])) for _ in range(2)]
    hist = prune_finetune(model, data, steps=3, lr=1e-2, warmup=2)
    assert hist["steps"] == 3
    assert len(hist["loss_history"]) == 3


def test_detect_backend_pretrained_module():
    class ToyPretrainedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = object()

        def save_pretrained(self, *a, **k):
            pass

    backend = detect_backend(ToyPretrainedModel())
    assert backend.name == "pretrained-module"


def test_detect_backend_plain_torch():
    backend = detect_backend(nn.Linear(2, 2))
    assert backend.name == "torch"


def _module_named(fake_module):
    """Build an nn.Module instance whose __class__.__module__ is ``fake_module``."""

    class ToyNamedModel(nn.Module):
        pass

    ToyNamedModel.__module__ = fake_module
    return ToyNamedModel()


def test_detect_backend_huggingface():
    backend = detect_backend(_module_named("transformers.models.foo"))
    assert backend.name == "huggingface"


def test_detect_backend_modelscope():
    backend = detect_backend(_module_named("modelscope.models.bar"))
    assert backend.name == "modelscope"


def test_try_set_int_attr_missing_attr_noop():
    obj = object()
    _try_set_int_attr(obj, "nonexistent", 5)
    assert not hasattr(obj, "nonexistent")


def test_try_set_int_attr_bad_value_swallowed():
    class AttrHolder:
        def __init__(self):
            self.field = 1

    obj = AttrHolder()
    _try_set_int_attr(obj, "field", "not_an_int")
    assert obj.field == 1


def test_try_set_int_attr_ok():
    class AttrHolder:
        def __init__(self):
            self.field = 0

    obj = AttrHolder()
    _try_set_int_attr(obj, "field", "7")
    assert obj.field == 7


def test_patch_common_config_no_config_returns():
    patch_common_config(nn.Linear(2, 2))


def test_patch_common_config_no_meta_returns():
    class EmptyConfig:
        pass

    class ConfiguredModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = EmptyConfig()

    m = ConfiguredModel()
    patch_common_config(m)


def test_patch_common_config_per_layer_widths_warns(caplog):
    class Cfg:
        intermediate_size = 100

    class ConfiguredModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()

    m = ConfiguredModel()
    setattr(m, "_amct_prune_meta", {"_dense_widths": {10, 20}})
    with caplog.at_level(logging.WARNING, logger="Log"):
        patch_common_config(m)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_patch_common_config_uniform_sets_scalar():
    class Cfg:
        intermediate_size = 100

    class ConfiguredModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()

    m = ConfiguredModel()
    setattr(m, "_amct_prune_meta", {"dense_hidden_size": 42, "moe_num_experts": 3})
    patch_common_config(m)
    assert m.config.intermediate_size == 42


def test_patch_common_config_moe_per_layer_counts_warns(caplog):
    class Cfg:
        num_experts = 8
        num_local_experts = 8

    class ConfiguredModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()

    m = ConfiguredModel()
    setattr(m, "_amct_prune_meta", {"moe_num_experts": 4, "_moe_widths": {4, 8}})
    with caplog.at_level(logging.WARNING, logger="Log"):
        patch_common_config(m)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert m.config.num_experts == 8
    assert m.config.num_local_experts == 8


def test_patch_common_config_moe_uniform_sets_scalar():
    class Cfg:
        num_experts = 8
        num_local_experts = 8

    class ConfiguredModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()

    m = ConfiguredModel()
    setattr(m, "_amct_prune_meta", {"moe_num_experts": 4, "_moe_widths": {4}})
    patch_common_config(m)
    assert m.config.num_experts == 4
    assert m.config.num_local_experts == 4


class _Router(nn.Module):
    """Toy paramless router exposing group_limited_topk + expert_bias."""

    def __init__(self, num_experts=8):
        super().__init__()
        self.top_k = 2
        self.expert_bias = torch.zeros(num_experts)

        def group_limited_topk(scores):
            return None

        self.group_limited_topk = group_limited_topk


class _MoEBlock(nn.Module):
    def __init__(self, num_experts=8):
        super().__init__()
        self.gate = _Router(num_experts)
        self.dummy = nn.Parameter(torch.zeros(1))


class _MoEModel(nn.Module):
    def __init__(self, num_experts=8):
        super().__init__()
        self.block = _MoEBlock(num_experts)


def test_patch_router_installs_simple_topk_and_bias():
    model = _MoEModel(num_experts=8)
    target = MoETarget(
        module_path="block",
        router_path="block.gate",
        experts_path="block.experts",
    )
    # Distinct non-zero bias so we can verify the kept experts' values are
    # preserved (not reset) and stay in keep_idx order after the slice.
    model.block.gate.expert_bias = torch.arange(10.0, 18.0)  # [10, 11, ..., 17]
    keep_idx = [0, 2, 4, 6]
    patch_router_module(model, target, keep_idx, top_k=2)

    router = model.block.gate
    scores = torch.randn(5, 4)
    vals, idx = router.group_limited_topk(scores)
    assert vals.shape[-1] == 2
    assert router.expert_bias.shape[0] == len(keep_idx)
    assert torch.equal(router.expert_bias, torch.tensor([10.0, 12.0, 14.0, 16.0]))


def test_patch_router_none_router_returns():
    class NoRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = NoRouter()

    model = Wrap()
    target = MoETarget(
        module_path="block", router_path=None, experts_path="block.experts"
    )
    patch_router_module(model, target, [0, 1], top_k=2)


class TestFinetune(unittest.TestCase):
    def test_finetune_reduces_loss(self):
        data, d, c = self._toy_task()
        model = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, c))
        hist = prune_finetune(
            model,
            data,
            loss_fn=lambda m, b: F.cross_entropy(m(b[0]), b[1]),
            steps=200,
            lr=1e-2,
        )
        self.assertEqual(hist["steps"], 200)
        self.assertLess(hist["final_loss"], hist["initial_loss"])
        self.assertLess(hist["final_loss"], 0.5)

    def test_finetune_default_loss_xy(self):
        data, d, c = self._toy_task()
        model = nn.Sequential(nn.Linear(d, c))
        hist = prune_finetune(model, data, steps=50, lr=1e-2)
        self.assertEqual(hist["steps"], 50)
        self.assertIsNotNone(hist["final_loss"])

    def test_finetune_zero_steps_noop(self):
        data, d, c = self._toy_task()
        model = nn.Sequential(nn.Linear(d, c))
        before = [p.clone() for p in model.parameters()]
        hist = prune_finetune(model, data, steps=0)
        self.assertEqual(hist["steps"], 0)
        for a, b in zip(before, model.parameters()):
            self.assertTrue(torch.equal(a, b))

    def _toy_task(self, n=256, d=16, c=4):
        torch.manual_seed(0)
        w_mat = torch.randn(d, c)
        x_in = torch.randn(n, d)
        y_lbl = (x_in @ w_mat).argmax(-1)
        batches = []
        for i in range(0, n, 32):
            sl = slice(i, i + 32)
            batches.append((x_in[sl], y_lbl[sl]))
        return batches, d, c


class TestDefaultAndMoEMethods(unittest.TestCase):
    def test_default_config_prunes(self):
        model, config = create_mini_moe()
        model.eval()
        p_before = _params(model)
        prune(model, data=_tok())
        self.assertLess(_params(model), p_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))

    def test_moe_activation_count_method(self):
        model, config = create_mini_moe()
        model.eval()
        cfg = {
            "methods": {
                "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.5}}
            },
            "missing_data_policy": "warn_skip",
        }
        report = PruneReport()
        prune(model, cfg, data=_tok(), report=report)
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))

    def test_full_structured_preset(self):
        model, _ = create_mini_moe()
        model.eval()
        p_before = _params(model)
        prune(model, FULL_STRUCTURED_PRUNE_CFG, data=_tok())
        self.assertLess(_params(model), p_before)


class TestReportAndPass(unittest.TestCase):
    def test_per_layer_sparsity_recorded(self):
        model, _ = create_mini_mlp()
        model.eval()
        report = PruneReport()
        prune(
            model,
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
                },
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
            report=report,
        )
        self.assertGreater(len(report.per_layer_sparsity), 0)
        for sparsity in report.per_layer_sparsity.values():
            self.assertGreater(sparsity, 0.0)
        self.assertIn("per_layer_sparsity", report.as_dict())


class TestSearchArgsOutsideSearchMode(unittest.TestCase):
    """Search-only arguments must be rejected, not silently dropped, in fixed-ratio mode."""

    CFG = {
        "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        "missing_data_policy": "warn_skip",
    }

    def test_each_search_arg_is_rejected(self):
        for name, value in (
            ("evaluator", lambda m: 1.0),
            ("eval_data", _tok()),
            ("ratio_grid", (0.2, 0.4)),
            ("finetune_fn", lambda m: None),
            ("quant_fn", lambda m: None),
        ):
            with self.subTest(arg=name):
                with self.assertRaises(ValueError) as ctx:
                    self._prune(**{name: value})
                self.assertIn(name, str(ctx.exception))

    def test_fixed_ratio_without_search_args_still_prunes(self):
        model, _ = create_mini_mlp()
        model.eval()
        p_before = _params(model)
        prune(model, self.CFG, data=_tok())
        self.assertLess(_params(model), p_before)

    def test_search_args_accepted_when_searching(self):
        model, _ = create_mini_mlp()
        model.eval()
        p_before = _params(model)
        prune(
            model,
            self.CFG,
            data=_tok(),
            tolerance=0.9,
            evaluator=lambda m: 1.0,
            ratio_grid=(0.2,),
        )
        self.assertLess(_params(model), p_before)

    def _prune(self, **kwargs):
        model, _ = create_mini_mlp()
        model.eval()
        prune(model, self.CFG, data=_tok(), **kwargs)


class TestUnknownMethodKwargs(unittest.TestCase):
    """A kwarg no method reads must raise, not fall back to the default prune ratio."""

    @staticmethod
    def _cfg(spec):
        return {"methods": {"dense": spec}, "missing_data_policy": "warn_skip"}

    def test_typo_kwarg_is_rejected(self):
        # 'prune_rate' used to be dropped in silence, leaving the 0.50 default in force:
        # a caller asking for 10% got 50%.
        for spec in (
            {"name": "low_variance", "kwargs": {"prune_rate": 0.1}},
            {"name": "low_variance", "prune_rate": 0.1},  # flat form folds in too
        ):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError) as ctx:
                    PruneConfig(**self._cfg(spec)).validate()
                self.assertIn("prune_rate", str(ctx.exception))
                self.assertIn("prune_ratio", str(ctx.exception))

    def test_seed_is_no_longer_advertised(self):
        # No prune_op ever read 'seed'; it must not look supported.
        with self.assertRaises(ValueError):
            PruneConfig(
                **self._cfg({"name": "low_variance", "kwargs": {"seed": 42}})
            ).validate()

    def test_accepted_kwargs_still_pass(self):
        PruneConfig(
            **self._cfg(
                {
                    "name": "reconstruct",
                    "kwargs": {"prune_ratio": 0.5, "ridge": 1e-2, "recovery": "ls"},
                }
            )
        ).validate()

    def test_every_shipped_preset_validates(self):
        for name in (
            "CNN_RECONSTRUCT_PRUNE_CFG",
            "CNN_VARIANCE_PRUNE_CFG",
            "CNN_RECOVERY_MENU_CFG",
            "DENSE_LOWVAR_PRUNE_CFG",
            "DENSE_RECOVERY_MENU_CFG",
            "FULL_STRUCTURED_PRUNE_CFG",
            "MOE_MASSVAR_PRUNE_CFG",
            "MOE_OUTPUT_MERGE_PRUNE_CFG",
            "MOE_VARIANCE_MENU_CFG",
            "SENSITIVITY_ALLOC_PRUNE_CFG",
        ):
            with self.subTest(preset=name):
                PruneConfig(**getattr(presets, name)).validate()

    def test_typo_inside_a_menu_variant_is_rejected(self):
        cfg = copy.deepcopy(MOE_VARIANCE_MENU_CFG)
        menu = list(cfg["methods"]["moe"]["menu"])
        menu[1] = (
            menu[1][0],
            {"name": "mass_variance", "kwargs": {"varianc_score": "cond"}},
        )
        cfg["methods"]["moe"]["menu"] = tuple(menu)
        with self.assertRaises(ValueError) as ctx:
            PruneConfig(**cfg).validate()
        self.assertIn("varianc_score", str(ctx.exception))

    def test_method_from_a_caller_registry_is_not_checked(self):
        # Only the shipped methods declare accepted_kwargs; a caller-supplied one
        # must keep working with kwargs this package knows nothing about.
        class CustomMethod(BasePruningMethod):
            domain = "dense"
            name = "custom_method"
            requires_data = False
            requires_targets = False

            def apply(self, model, domain, targets, context, report, config, spec):
                return None

        registry = create_default_registry()
        registry[("dense", "custom_method")] = DomainMethodBinding(
            domain=DensePruningDomain(), method=CustomMethod()
        )
        config = PruneConfig(
            **self._cfg({"name": "custom_method", "kwargs": {"anything": 1}})
        )
        AutoPruner(config, registry=registry)


class TestQuantFnReachesSearch(unittest.TestCase):
    """quant_fn is documented on prune(); it must actually reach the search."""

    CFG = {
        "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        "missing_data_policy": "warn_skip",
    }

    def test_quant_fn_called_during_tolerance_search(self):
        model, _ = create_mini_mlp()
        model.eval()
        seen = []
        prune(
            model,
            self.CFG,
            data=_tok(),
            tolerance=0.9,
            ratio_grid=(0.2,),
            quant_fn=lambda m: seen.append(m),
        )
        self.assertTrue(seen, "quant_fn was never applied to a pruned candidate")


class TestPinnedDomainSurvivesSearch(unittest.TestCase):
    """A domain pinned to prune_ratio 0.0 must stay untouched while the search varies the rest."""

    def test_with_ratio_keeps_pins(self):
        from amct_pytorch.pruning.accuracy_based_auto_prune import _with_ratio

        cfg = {
            "methods": {
                "moe": {"name": "mass_variance", "kwargs": {"prune_ratio": 0.1}},
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
                "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
            }
        }
        out = _with_ratio(cfg, 0.4)
        ratios = {d: s["kwargs"]["prune_ratio"] for d, s in out["methods"].items()}
        self.assertEqual(ratios, {"moe": 0.4, "dense": 0.0, "cnn": 0.0})

    def test_with_ratio_accepts_flat_pin(self):
        """prune_ratio may sit flat next to name instead of under kwargs; the pin still holds."""
        from amct_pytorch.pruning.accuracy_based_auto_prune import _with_ratio

        out = _with_ratio(
            {
                "methods": {
                    "moe": {"name": "mass_variance", "kwargs": {}},
                    "dense": {"name": "low_variance", "prune_ratio": 0.0},
                }
            },
            0.4,
        )
        self.assertEqual(out["methods"]["dense"]["prune_ratio"], 0.0)
        self.assertEqual(out["methods"]["moe"]["kwargs"]["prune_ratio"], 0.4)

    def test_search_leaves_pinned_dense_untouched(self):
        model, _ = create_mini_mlp()
        model.eval()
        before = _params(model)
        prune(
            model,
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}}
                },
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
            tolerance=0.9,
            ratio_grid=(0.5,),
        )
        self.assertEqual(_params(model), before)

    def test_pinned_domain_is_not_counted_as_prunable_mass(self):
        """The size-budget accounting must agree with what the search will actually prune."""
        from amct_pytorch.pruning.accuracy_based_auto_prune import _prunable_param_count

        model, _ = create_mini_mlp()
        model.eval()
        free = _prunable_param_count(model, {})
        pinned = _prunable_param_count(
            model,
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}}
                }
            },
        )
        self.assertGreater(free, 0)
        self.assertEqual(pinned, 0)

    def test_diagnose_respects_pin(self):
        model, _ = create_mini_mlp()
        model.eval()
        before = _params(model)
        prune_diagnose(
            model,
            data=_tok(),
            config={
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}}
                },
                "missing_data_policy": "warn_skip",
            },
        )
        self.assertEqual(_params(model), before)


class TestOnlyNamedDomainsArePruned(unittest.TestCase):
    """One rule across every mode: methods names the targets, everything else stays put."""

    NAMED_MOE = {
        "methods": {
            "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.5}}
        },
        "missing_data_policy": "warn_skip",
    }

    def test_resolved_methods_pins_unnamed(self):
        resolved = PruneConfig(**self.NAMED_MOE).resolved_methods()
        self.assertEqual(resolved["moe"].kwargs["prune_ratio"], 0.5)
        self.assertEqual(resolved["dense"].kwargs["prune_ratio"], 0.0)
        self.assertEqual(resolved["cnn"].kwargs["prune_ratio"], 0.0)

    def test_no_methods_still_uses_defaults(self):
        resolved = PruneConfig().resolved_methods()
        self.assertEqual(resolved["cnn"].kwargs["prune_ratio"], 0.30)
        self.assertEqual(resolved["dense"].kwargs["prune_ratio"], 0.50)
        self.assertEqual(resolved["moe"].kwargs["prune_ratio"], 0.50)

    def test_fixed_mode_leaves_inner_ffn_alone(self):
        """A moe-only config must not shrink the experts' inner dense FFN."""
        model, _ = create_mini_moe()
        model.eval()
        before = model.layers[0].experts[0].fc1.weight.shape
        n_before = len(model.layers[0].experts)
        prune(model, self.NAMED_MOE, data=_tok())
        self.assertLess(len(model.layers[0].experts), n_before)
        self.assertEqual(model.layers[0].experts[0].fc1.weight.shape, before)

    def test_prunable_mass_matches_named_domains(self):
        from amct_pytorch.pruning.accuracy_based_auto_prune import _prunable_param_count

        model, _ = create_mini_mlp()
        model.eval()
        self.assertEqual(_prunable_param_count(model, self.NAMED_MOE), 0)


class TestMenuRejectsSearchArgs(unittest.TestCase):
    """A menu config plus a search argument must fail loudly, not drop the argument."""

    def test_tolerance_with_menu_raises(self):
        from amct_pytorch.pruning import DENSE_RECOVERY_MENU_CFG

        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(ValueError) as ctx:
            prune(model, DENSE_RECOVERY_MENU_CFG, data=_tok(), tolerance=0.0)
        self.assertIn("tolerance", str(ctx.exception))


class TestFlatPruneRatioForm(unittest.TestCase):
    """prune_ratio may sit flat beside name; it outranks kwargs, so every path must honour it."""

    def test_search_overrides_flat_ratio(self):
        from amct_pytorch.pruning.accuracy_based_auto_prune import _with_ratio

        out = _with_ratio(
            {"methods": {"dense": {"name": "low_variance", "prune_ratio": 0.1}}}, 0.8
        )
        spec = out["methods"]["dense"]
        self.assertNotIn("prune_ratio", spec)
        self.assertEqual(spec["kwargs"]["prune_ratio"], 0.8)

    def test_menu_reads_flat_ratio(self):
        from amct_pytorch.pruning.accuracy_based_auto_prune import _menu_spec

        _menu, ratio, _common = _menu_spec(
            {
                "methods": {
                    "dense": {
                        "name": "reconstruct",
                        "prune_ratio": 0.5,
                        "menu": (("none", {"name": "reconstruct", "kwargs": {}}),),
                    }
                }
            },
            "dense",
        )
        self.assertEqual(ratio, 0.5)


class TestAllocationWithoutDense(unittest.TestCase):
    """Sensitivity allocation must work off whichever domain the config actually names."""

    def test_uniform_cut_uses_active_domain(self):
        from amct_pytorch.pruning.allocation import _uniform_cut

        cfg = PruneConfig(
            methods={
                "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.3}}
            }
        )
        self.assertAlmostEqual(_uniform_cut(cfg), 0.3)

    def test_all_zero_raises_named_error(self):
        from amct_pytorch.pruning.allocation import _uniform_cut

        cfg = PruneConfig(
            methods={
                "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}}
            }
        )
        with self.assertRaises(ValueError) as ctx:
            _uniform_cut(cfg)
        self.assertIn("prune_ratio > 0", str(ctx.exception))


class TestStructuralEdgeCases(unittest.TestCase):
    """Shapes the domains accept but the pruners used to mishandle."""

    def test_batchnorm_without_affine(self):
        from amct_pytorch.pruning.utils import prune_batchnorm2d

        bn = nn.BatchNorm2d(8, affine=False)
        out = prune_batchnorm2d(bn, [0, 1, 2, 3])
        self.assertEqual(out.num_features, 4)
        self.assertFalse(out.affine)

    def test_conv1d_producer_is_pruned_by_low_variance(self):
        """The dense domain emits Conv1D targets; low_variance must not skip them."""

        class Conv1D(nn.Module):
            def __init__(self, nf, nx):
                super().__init__()
                self.nf = nf
                self.weight = nn.Parameter(torch.randn(nx, nf) * 0.02)
                self.bias = nn.Parameter(torch.zeros(nf))

            def forward(self, x):
                return x @ self.weight + self.bias

        class MLP(nn.Module):
            def __init__(self, h=32, i=128):
                super().__init__()
                self.c_fc = Conv1D(i, h)
                self.c_proj = Conv1D(h, i)

            def forward(self, x):
                return self.c_proj(torch.relu(self.c_fc(x)))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

            def forward(self, x):
                return self.mlp(x)

        model = Net().eval()
        prune(
            model,
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
                },
                "missing_data_policy": "warn_skip",
            },
            data=[torch.randn(4, 6, 32) for _ in range(4)],
        )
        self.assertLess(model.mlp.c_fc.nf, 128)

    def test_conv_not_feeding_head_is_not_paired_with_it(self):
        """A conv whose real consumer is the next block must not be attached to the head."""
        from amct_pytorch.pruning.domains.cnn import CNNPruningDomain

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.b1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
                self.b2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                x = self.pool(self.b2(self.b1(x)))
                return self.fc(x.flatten(1))

        model = Net().eval()
        targets = CNNPruningDomain().find_targets(model, PruneConfig())
        paired = {t.producer_path: t.consumer_path for t in targets}
        self.assertNotEqual(paired.get("b1.0"), "fc")


class TestCnnRecoveryMenuPreset(unittest.TestCase):
    """CNN_RECOVERY_MENU_CFG must actually select a recovery variant and prune the CNN."""

    def test_cnn_menu_preset_prunes_and_selects(self):
        from amct_pytorch.pruning import CNN_RECOVERY_MENU_CFG

        model, _ = create_mini_cnn()
        model.eval()
        before = _params(model)
        calib = [torch.randn(4, 3, 32, 32) for _ in range(4)]
        val = [torch.randn(2, 3, 32, 32) for _ in range(2)]
        report = PruneReport()
        prune(model, CNN_RECOVERY_MENU_CFG, data=calib, eval_data=val, report=report)
        self.assertLess(_params(model), before)
        self.assertTrue(
            any(e.stage == "menu-select" for e in report.events),
            "menu selection never ran for the CNN preset",
        )


class TestMenuNeutralisesUnlistedDomains(unittest.TestCase):
    """A menu config names one domain; the others must not be pruned at their defaults."""

    def test_unlisted_domains_are_neutralised(self):
        from amct_pytorch.pruning.accuracy_based_auto_prune import (
            _menu_candidate_config,
        )

        cfg = _menu_candidate_config(
            {"methods": {"moe": {"name": "mass_variance", "kwargs": {}}}},
            "moe",
            {"name": "mass_variance", "kwargs": {}},
            0.5,
            {},
            False,
        )
        ratios = {
            d: s.kwargs.get("prune_ratio") for d, s in cfg.resolved_methods().items()
        }
        self.assertEqual(ratios["moe"], 0.5)
        self.assertEqual(ratios["dense"], 0.0)
        self.assertEqual(ratios["cnn"], 0.0)


class _FakeEvaluator:
    """Mimics the quantization-side ModelEvaluator: evaluate(model, iterations) -> accuracy (here a param ratio)."""

    def __init__(self, p0):
        self.p0 = p0
        self.calls = 0

    def evaluate(self, model, iterations):
        self.calls += 1
        return sum(p.numel() for p in model.parameters()) / self.p0


class TestEvaluatorUnification(unittest.TestCase):
    def test_evaluator_drives_search(self):
        model, _ = create_mini_mlp()
        model.eval()
        p0 = _params(model)
        ev = _FakeEvaluator(p0)
        res = accuracy_based_auto_prune(
            model,
            {
                "methods": {"dense": {"name": "low_variance"}},
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
            tolerance=0.2,
            evaluator=ev,
        )
        self.assertGreater(ev.calls, 1)
        self.assertIsNotNone(res.chosen_ratio)
        self.assertLessEqual(res.quality_drop, 0.2 + 1e-9)
        self.assertLessEqual(res.weight_reduction, 0.2 + 1e-9)

    def test_bad_evaluator_raises(self):
        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(TypeError):
            accuracy_based_auto_prune(
                model, data=_tok(), tolerance=0.2, evaluator=object()
            )

    def test_single_arg_evaluator_works(self):
        model, _ = create_mini_mlp()
        model.eval()
        p0 = _params(model)

        class _OneArgEvaluator:
            def __init__(self):
                self.calls = 0

            def evaluate(self, model):
                self.calls += 1
                return sum(p.numel() for p in model.parameters()) / p0

        ev = _OneArgEvaluator()
        res = accuracy_based_auto_prune(
            model,
            {
                "methods": {"dense": {"name": "low_variance"}},
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
            tolerance=0.2,
            evaluator=ev,
        )
        self.assertGreater(ev.calls, 1)
        self.assertIsNotNone(res.chosen_ratio)

    def test_none_returning_evaluator_raises(self):
        model, _ = create_mini_mlp()
        model.eval()

        class _FeederEvaluator:
            @staticmethod
            def evaluate(model, iterations):
                return None

        with self.assertRaises(TypeError):
            accuracy_based_auto_prune(
                model, data=_tok(), tolerance=0.2, evaluator=_FeederEvaluator()
            )


class TestDiagnose(unittest.TestCase):
    def test_diagnose_functional_model(self):
        model, _ = create_mini_moe()
        model.eval()
        p_before = _params(model)
        rep = prune_diagnose(model, data=_tok())
        self.assertIsInstance(rep, DiagnosisReport)
        self.assertTrue(rep.any_domain_detected)
        self.assertGreater(rep.targets["dense"], 0)
        self.assertGreater(rep.targets["moe"], 0)
        self.assertTrue(rep.prune_works)
        self.assertGreater(rep.prune_reduction, 0.0)
        self.assertTrue(rep.prune_forward_ok)
        self.assertIsInstance(rep.search_works, bool)
        self.assertIsInstance(rep.summary(), str)
        self.assertEqual(_params(model), p_before)

    def test_diagnose_unsupported_model(self):
        model = nn.Sequential(nn.Embedding(50, 8))
        rep = prune_diagnose(model, data=None)
        self.assertFalse(rep.any_domain_detected)
        self.assertFalse(rep.prune_works)
        self.assertTrue(any("in any domain" in n for n in rep.notes))


class TestWarnSkipAtomicRollback(unittest.TestCase):
    """warn_skip atomic rollback (#16): a mid-stage failure leaves the model byte-unchanged."""

    def test_mid_stage_failure_leaves_model_unchanged(self):
        import amct_pytorch.pruning.pruner as pruner_mod

        model, cfg, moe_stage = self._moe_stage_pruner()
        calib = [torch.randint(0, 1000, (4, 20)) for _ in range(8)]
        context = PruneContext(data=calib)
        report = PruneReport()
        snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

        def boom(*args, **kwargs):
            raise RuntimeError("induced mid-stage failure")

        moe_stage.method.apply = boom
        returned = moe_stage.apply(model, context, report, cfg)

        self.assertIs(returned, model)
        current = model.state_dict()
        for name, ref in snapshot.items():
            self.assertTrue(
                torch.equal(ref, current[name]),
                msg=f"parameter '{name}' was mutated despite warn_skip rollback",
            )

        model2, cfg2, moe_stage2 = self._moe_stage_pruner()
        snapshot2 = {k: v.detach().clone() for k, v in model2.state_dict().items()}

        def half_then_raise(target, *args, **kwargs):
            with torch.no_grad():
                next(iter(target.parameters())).add_(1.0)
            raise RuntimeError("boom after partial mutation")

        moe_stage2.method.apply = half_then_raise
        original_deepcopy = pruner_mod.copy.deepcopy
        pruner_mod.copy.deepcopy = lambda obj: obj
        try:
            moe_stage2.apply(model2, context, PruneReport(), cfg2)
        finally:
            pruner_mod.copy.deepcopy = original_deepcopy

        current2 = model2.state_dict()
        leaked = any(
            not torch.equal(ref, current2[name]) for name, ref in snapshot2.items()
        )
        self.assertTrue(leaked, "broken rollback should leak the partial mutation")

    def _moe_stage_pruner(self):
        model, _ = create_mini_moe()
        model.eval()
        cfg = PruneConfig(
            methods={
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {"prune_ratio": 0.5, "boundary": 10},
                }
            },
            stage_error_policy="warn_skip",
            copy_model=False,
        )
        pruner = AutoPruner(cfg)
        moe_stage = next(s for s in pruner.stages if s.domain_name == "moe")
        return model, cfg, moe_stage


if __name__ == "__main__":
    unittest.main()
