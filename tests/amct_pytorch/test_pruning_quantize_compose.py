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
"""End-to-end tests of pruning composed with quantization (prune -> quantize -> convert), verifying retention:
quantizing a pruned model should not significantly change its predictions. amct.quantize/convert runs on CPU
for small models; the whole group is skipped if unavailable.
"""

import copy
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp, MiniMoEConfig, MiniMoEModel
import amct_pytorch.pruning as P
from amct_pytorch.pruning.accuracy_based_auto_prune import (
    _accuracy_based_auto_prune as accuracy_based_auto_prune,
)
from amct_pytorch.pruning import (
    MOE_VARIANCE_MENU_CFG,
    DENSE_RECOVERY_MENU_CFG,
)

try:
    import amct_pytorch as amct

    _HAS_QUANT = hasattr(amct, "quantize") and hasattr(amct, "convert")
except Exception:
    _HAS_QUANT = False

try:
    import torch_npu

    _HAS_NPU = (torch_npu is not None) and (torch.npu.is_available() is True)
except Exception:
    _HAS_NPU = False


def _quant_module_count(model):
    """Count layers actually replaced by quant modules (type name contains 'Quant').

    Key fact: amct quantization only fires on fp16/bf16 weights and fake-quant needs torch_npu.
    On CPU+fp32 quantize/convert is a no-op (inserts no quant modules). This count distinguishes
    "quantization didn't actually happen" from real quantization, so a no-op can't make a test pass spuriously.
    """
    return sum(1 for m in model.modules() if "Quant" in type(m).__name__)


def _tok(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


def _params(m):
    return sum(p.numel() for p in m.parameters())


def _top1_agree(a, b):
    return (a.argmax(-1) == b.argmax(-1)).float().mean().item()


DENSE_CFG = {
    "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
    "skip_layers": ["self_attn"],
    "missing_data_policy": "warn_skip",
}


@unittest.skipUnless(_HAS_QUANT, "amct.quantize/convert unavailable")
class TestPruneThenQuantize(unittest.TestCase):
    def test_prune_then_quantize_pipeline_runs(self):
        x = torch.randint(0, 1000, (4, 20))
        model = self._fresh()
        with torch.no_grad():
            shape_before = model(x).shape

        p_before = _params(model)
        P.prune(model, DENSE_CFG, data=_tok())
        self.assertLess(_params(model), p_before)

        amct.quantize(model)
        with torch.no_grad():
            out_q = model(x)
        self.assertEqual(out_q.shape, shape_before)
        self.assertTrue(torch.isfinite(out_q).all())

        amct.convert(model)
        with torch.no_grad():
            out_c = model(x)
        self.assertEqual(out_c.shape, shape_before)
        self.assertTrue(torch.isfinite(out_c).all())

    def test_quantization_retention_on_pruned_model(self):
        if not _HAS_NPU:
            self.skipTest("real quantization needs fp16 weights + torch_npu (NPU)")
        dev = "npu:0"
        x = torch.randint(0, 1000, (8, 20)).to(dev)
        state = self._fresh().state_dict()
        calib = _tok()

        pruned = self._fresh(state).half().to(dev)
        P.prune(pruned, DENSE_CFG, data=calib)
        with torch.no_grad():
            out_pruned = pruned(x)

        pq = self._fresh(state).half().to(dev)
        P.prune(pq, DENSE_CFG, data=calib)
        amct.quantize(pq)
        self.assertGreater(
            _quant_module_count(pq),
            0,
            "quantize() did not insert any quant modules -- retention would be meaningless",
        )
        with torch.no_grad():
            out_pq = pq(x)

        retention = _top1_agree(out_pruned.float(), out_pq.float())
        self.assertGreaterEqual(retention, 0.9)
        self.assertTrue(torch.isfinite(out_pq).all())

    def test_cpu_quantize_is_noop_documented(self):
        m = self._fresh()
        amct.quantize(m)
        self.assertEqual(
            _quant_module_count(m),
            0,
            "CPU+fp32 quantize unexpectedly inserted quant modules -- assumption changed",
        )

    def test_quantization_actually_fires_and_perturbs(self):
        if not _HAS_NPU:
            self.skipTest("needs torch_npu (NPU)")
        dev = "npu:0"
        m = self._fresh().half().to(dev)
        x = torch.randint(0, 1000, (4, 20)).to(dev)
        with torch.no_grad():
            y0 = m(x).float().cpu()
        amct.quantize(m)
        self.assertGreater(_quant_module_count(m), 0)
        amct.convert(m)
        deploy_types = {
            type(mm).__name__ for mm in m.modules() if "Quant" in type(mm).__name__
        }
        self.assertTrue(deploy_types, "no deploy quant modules after convert")
        with torch.no_grad():
            y2 = m(x).float().cpu()
        self.assertGreater((y2 - y0).abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(y2).all())

    def test_tolerance_prune_then_quantize(self):
        x = torch.randint(0, 1000, (4, 20))
        model = self._fresh()
        p0 = _params(model)
        res = accuracy_based_auto_prune(
            model,
            DENSE_CFG,
            data=_tok(),
            tolerance=0.3,
            evaluator=lambda m: _params(m) / p0,
        )
        self.assertTrue(res.applied)

        amct.quantize(model)
        with torch.no_grad():
            out = model(x)
        self.assertTrue(torch.isfinite(out).all())

    def _fresh(self, state=None):
        m, _ = create_mini_mlp()
        if state is not None:
            m.load_state_dict(state)
        m.eval()
        return m


def _mini_moe():
    torch.manual_seed(0)
    cfg = MiniMoEConfig(
        hidden_size=32,
        intermediate_size=48,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        vocab_size=1000,
    )
    return MiniMoEModel(cfg).eval()


@unittest.skipUnless(_HAS_QUANT, "amct.quantize/convert unavailable")
class TestMenuModeThenQuantize(unittest.TestCase):
    def test_dense_recovery_then_quantize_pipeline_runs(self):
        x = torch.randint(0, 1000, (4, 20))
        for rec in ("none", "bias", "ls"):
            with self.subTest(recovery=rec):
                model, _ = create_mini_mlp()
                model.eval()
                with torch.no_grad():
                    shape_before = model(x).shape
                cfg = {
                    "methods": {
                        "dense": {
                            "name": "reconstruct",
                            "kwargs": {"prune_ratio": 0.5, "recovery": rec},
                        }
                    },
                    "skip_layers": ["self_attn"],
                    "missing_data_policy": "warn_skip",
                }
                P.prune(model, cfg, data=_tok())
                self._assert_pipeline_runs(model, x, shape_before)

    def test_recovery_menu_then_quantize_pipeline_runs(self):
        model, _ = create_mini_mlp()
        model.eval()
        x = torch.randint(0, 1000, (4, 20))
        with torch.no_grad():
            shape_before = model(x).shape
        rcfg = copy.deepcopy(DENSE_RECOVERY_MENU_CFG)
        rcfg["methods"]["dense"]["kwargs"] = {"prune_ratio": 0.5, "ridge": 1e-2}
        accuracy_based_auto_prune(model, rcfg, data=_tok(), eval_data=_tok(3))
        self._assert_pipeline_runs(model, x, shape_before)

    def test_variance_menu_then_quantize_pipeline_runs(self):
        model = _mini_moe()
        x = torch.randint(0, 1000, (4, 20))
        with torch.no_grad():
            shape_before = model(x).shape
        vcfg = copy.deepcopy(MOE_VARIANCE_MENU_CFG)
        vcfg["methods"]["moe"]["kwargs"] = {"prune_ratio": 0.5, "top_k": 2}
        accuracy_based_auto_prune(model, vcfg, data=_tok(), eval_data=_tok(3))
        self._assert_pipeline_runs(model, x, shape_before)

    def test_moe_variance_score_then_quantize_pipeline_runs(self):
        model = _mini_moe()
        x = torch.randint(0, 1000, (4, 20))
        with torch.no_grad():
            shape_before = model(x).shape
        cfg = {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {
                        "prune_ratio": 0.5,
                        "top_k": 2,
                        "boundary": -1,
                        "variance_score": "peak",
                    },
                }
            },
            "missing_data_policy": "warn_skip",
        }
        P.prune(model, cfg, data=_tok())
        self._assert_pipeline_runs(model, x, shape_before)

    def _assert_pipeline_runs(self, model, x, shape_before):
        amct.quantize(model)
        with torch.no_grad():
            out_q = model(x)
        self.assertEqual(out_q.shape, shape_before)
        self.assertTrue(torch.isfinite(out_q).all())
        amct.convert(model)
        with torch.no_grad():
            out_c = model(x)
        self.assertEqual(out_c.shape, shape_before)
        self.assertTrue(torch.isfinite(out_c).all())


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
