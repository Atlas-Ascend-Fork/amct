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
"""AMCT structured-pruning comprehensive tests on HuggingFace-style mini models (MLP / CNN / MoE)."""

import os
import shutil
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp, create_mini_cnn, create_mini_moe
from amct_pytorch.pruning import (
    DENSE_LOWVAR_PRUNE_CFG,
    CNN_VARIANCE_PRUNE_CFG,
    MOE_MASSVAR_PRUNE_CFG,
    prune,
    get_last_report,
)


def _calib_tokens():
    return [torch.randint(0, 1000, (4, 20)) for _ in range(8)]


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
            prune(model, cfg, data=calib)
            report = get_last_report()
            self.assertIsNotNone(report)
            self.assertLess(report.params_after, report.params_before)
            reductions.append(1 - report.params_after / report.params_before)
        # reduction ratio increases monotonically with prune ratio
        self.assertLess(reductions[0], reductions[1])
        self.assertLess(reductions[1], reductions[2])


class TestSaveLoadRoundtrip(unittest.TestCase):
    def test_state_dict_roundtrip_matches(self):
        calib = _calib_tokens()
        original_model, config = create_mini_mlp()
        original_model.eval()
        prune(original_model, DENSE_LOWVAR_PRUNE_CFG, data=calib)
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)

        test_input = torch.randint(0, 1000, (2, 20))
        with torch.no_grad():
            original_output = original_model(test_input)

        temp_dir = tempfile.mkdtemp(prefix="amct_mlp_")
        try:
            save_path = os.path.join(temp_dir, "pruned_mlp.pth")
            # save state_dict only (do not pickle the whole nn.Module)
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
        prune(model, DENSE_LOWVAR_PRUNE_CFG, data=_calib_tokens())
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))

    def test_reconstruct_preserves_output_better_than_low_variance(self):
        import copy

        self.addCleanup(
            torch.set_rng_state, torch.get_rng_state()
        )  # don't pollute global RNG for later tests
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        # diverse calibration set (tokens >> intermediate dim); held-out eval batch disjoint from calib
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
        # reconstruction absorbs the removed neurons' contribution -> output closer to the original model
        self.assertLess(e_rec, e_lv)

    def test_conv1d_ffn_reconstruct_prunes_and_preserves(self):
        import torch.nn as nn
        import torch.nn.functional as F

        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

        class Conv1D(nn.Module):  # GPT-2 semantics: weight [in, out], nf=out
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
        calib = [
            torch.randn(8, 16, 32) for _ in range(20)
        ]  # diverse calibration, 2560 tokens >> 128
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
        self.assertLess(p1, p0)  # Conv1D actually pruned (no longer a no-op)
        with torch.no_grad():
            out = model(x)
        self.assertLess(((out - ref).norm() / ref.norm()).item(), 0.25)  # reconstruction preserves output

    def test_quant_aware_saliency_valid_and_distinct(self):
        import copy

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
        self.assertLess(
            sum(p.numel() for p in m_aware.parameters()), p_before
        )  # still genuinely pruned
        # 2-bit quantization distorts saliency -> different kept set -> output differs from quant-blind
        # (proves the option works)
        self.assertGreater((o_blind - o_aware).abs().max().item(), 1e-5)

    def test_cnn_channel_pruning(self):
        model, config = create_mini_cnn()
        model.eval()
        calib = [torch.randn(2, 3, 32, 32) for _ in range(8)]
        prune(model, CNN_VARIANCE_PRUNE_CFG, data=calib)
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(out.shape, (2, config.num_classes))

    def test_cnn_reconstruct_prunes_and_forwards(self):
        # In-house CNN reconstruct (no third-party deps): actually prunes, forward works, same structural
        # sparsity as variance. Accuracy benefit only matters on trained models (random weights can't
        # generalize), so here we assert structural correctness only; accuracy benefit is measured in the
        # examples MNIST end-to-end benchmark (trained: variance 0.343 vs reconstruct 0.574).
        import copy

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
            self.assertEqual(out.shape, (2, config.num_classes))  # forward works
            return sum(p.numel() for p in m.parameters())

        n_rc = pruned_params("reconstruct")
        n_var = pruned_params("variance_channel")
        self.assertLess(n_rc, p0)  # actually pruned (not a silent no-op)
        self.assertEqual(n_rc, n_var)  # same structural sparsity as variance

    def test_moe_expert_pruning(self):
        model, config = create_mini_moe()
        model.eval()
        calib = [torch.randint(0, 1000, (4, 20)) for _ in range(8)]
        prune(model, MOE_MASSVAR_PRUNE_CFG, data=calib)
        report = get_last_report()
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

        from amct_pytorch.pruning.methods.dense_reconstruct import (
            _fake_quant_amct,
        )

        w = torch.randn(16, 32)
        cfg = {"weights_cfg": {"strategy": "channel", "symmetric": True, "dtype": "int8"}}
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a fallback would warn -> treat as failure
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


if __name__ == "__main__":
    unittest.main()
