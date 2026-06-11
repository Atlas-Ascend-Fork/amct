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
"""Guarded recovery-HYBRID tests: recovery MENU picks best on val, never worse than fallback(none), params drop,
forward OK; also covers the dense/cnn reconstruct ``recovery`` kwarg (ls/bias/none) and backward compat
(default ls == old behavior).
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import (
    MiniMLPConfig,
    MiniMLPModel,
    MiniCNNConfig,
    MiniCNNModel,
)
from amct_pytorch.pruning import (
    guarded_recovery_hybrid_prune,
    prune,
)


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


def _mlp():
    torch.manual_seed(0)
    cfg = MiniMLPConfig(
        hidden_size=64, intermediate_size=256, num_hidden_layers=3, vocab_size=128
    )
    return MiniMLPModel(cfg).eval()


def _mlp_tokens(seed, n=6):
    torch.manual_seed(seed)
    return [torch.randint(0, 128, (2, 16)) for _ in range(n)]


def _cnn():
    torch.manual_seed(0)
    return MiniCNNModel(MiniCNNConfig()).eval()


def _images(seed, n=3, batch_size=8):
    torch.manual_seed(seed)
    return [torch.randn(batch_size, 3, 32, 32) for _ in range(n)]


def _trained_mlp(seed=0, steps=40):
    """Lightly trained mini-MLP: a randomly initialized toy model's output is nearly saturated
    (fidelity ~unchanged after any pruning) and cannot show a recovery quality ladder; only after
    training in nontrivial input-dependent structure does ls reconstruction truly beat none.
    """
    model = _mlp()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(123)
    data = [
        (torch.randint(0, 128, (4, 16)), torch.randint(0, 128, (4, 16)))
        for _ in range(8)
    ]
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        for x, y in data:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out.reshape(-1, out.shape[-1]), y.reshape(-1))
            loss.backward()
            optimizer.step()
    return model.eval()


class TestGuardedRecoveryHybrid(unittest.TestCase):
    def test_dense_never_worse_prunes_and_forwards(self):
        model = _mlp()
        before = _num_params(model)
        res = guarded_recovery_hybrid_prune(
            model,
            data=_mlp_tokens(7),
            prune_ratio=0.5,
            domain="dense",
            eval_data=_mlp_tokens(11, n=3),
            apply=True,
        )
        self.assertGreaterEqual(
            res.gain_over_fallback, 0.0
        )  # never worse than none on val
        self.assertEqual(res.fallback_recovery, "none")
        self.assertEqual(set(res.recovery_quality), {"none", "bias", "ls"})
        self.assertLess(_num_params(model), before)
        out = model(_mlp_tokens(11, n=1)[0])
        self.assertEqual(out.shape[-1], 128)

    def test_dense_recovery_quality_ladder_ls_best(self):
        # apply=False: only compare each MENU recovery's val quality, don't touch the original model
        res = guarded_recovery_hybrid_prune(
            _trained_mlp(0),
            data=_mlp_tokens(7, n=8),
            prune_ratio=0.5,
            domain="dense",
            eval_data=_mlp_tokens(11, n=4),
            apply=False,
        )
        q = res.recovery_quality
        self.assertEqual(set(q), {"none", "bias", "ls"})
        # ls is the best recovery: >= none and >= bias (so the guard picks ls over fallback).
        # Honest scope: only assert the partial order impl guarantees (ls>=none, ls>=bias); not bias>=none,
        # since under small calibration bias-folding overfits and can fall below none.
        # keep = round(256*(1-0.5)) = 128; calib rows = n*2*16 = 8*32 = 256 >= 128, so LS is determined.
        self.assertGreaterEqual(q["ls"], q["none"])
        self.assertGreaterEqual(q["ls"], q["bias"])
        # ladder really takes effect: ls is chosen and has strictly positive gain over fallback(none)
        self.assertEqual(res.chosen_recovery, "ls")
        self.assertGreater(res.gain_over_fallback, 0.0)

    def test_apply_false_leaves_model_unchanged(self):
        model = _mlp()
        before = _num_params(model)
        res = guarded_recovery_hybrid_prune(
            model,
            data=_mlp_tokens(7),
            prune_ratio=0.5,
            domain="dense",
            eval_data=_mlp_tokens(11, n=3),
            apply=False,
        )
        self.assertFalse(res.applied)
        self.assertEqual(_num_params(model), before)
        self.assertLess(res.params_after, before)

    def test_safety_menu_drops_bias(self):
        # small-calibration safety knob: recovery_menu=("none","ls") evaluates only these two
        model = _mlp()
        res = guarded_recovery_hybrid_prune(
            model,
            data=_mlp_tokens(7),
            prune_ratio=0.5,
            domain="dense",
            recovery_menu=("none", "ls"),
            eval_data=_mlp_tokens(11, n=3),
        )
        self.assertEqual(set(res.recovery_quality), {"none", "ls"})

    def test_cnn_domain_prunes_and_forwards(self):
        model = _cnn()
        before = _num_params(model)
        res = guarded_recovery_hybrid_prune(
            model,
            data=_images(3),
            prune_ratio=0.3,
            domain="cnn",
            eval_data=_images(5, n=2),
            apply=True,
        )
        self.assertGreaterEqual(res.gain_over_fallback, 0.0)
        self.assertLess(_num_params(model), before)
        out = model(_images(5, n=1)[0])
        self.assertEqual(out.shape[0], 8)

    def test_cnn_apply_false_leaves_model_unchanged(self):
        model = _cnn()
        before = _num_params(model)
        res = guarded_recovery_hybrid_prune(
            model,
            data=_images(3),
            prune_ratio=0.3,
            domain="cnn",
            eval_data=_images(5, n=2),
            apply=False,
        )
        self.assertFalse(res.applied)
        self.assertEqual(_num_params(model), before)  # original model untouched
        self.assertLess(res.params_after, before)

    def test_invalid_args_raise(self):
        with self.assertRaises(ValueError):
            guarded_recovery_hybrid_prune(_mlp(), data=_mlp_tokens(7), prune_ratio=1.0)
        with self.assertRaises(ValueError):
            guarded_recovery_hybrid_prune(_mlp(), data=_mlp_tokens(7), domain="bogus")


class TestRecoveryKwarg(unittest.TestCase):
    def test_dense_each_recovery_prunes(self):
        for rec in ("none", "bias", "ls"):
            with self.subTest(recovery=rec):
                model = _mlp()
                before = _num_params(model)
                prune(model, self._dense_cfg(rec), data=_mlp_tokens(7))
                self.assertLess(_num_params(model), before)
                out = model(_mlp_tokens(11, n=1)[0])
                self.assertEqual(out.shape[-1], 128)

    def test_dense_recovery_keeps_same_neurons(self):
        # recovery only changes weights/biases, not which neurons are kept -> none and ls have equal param counts
        m_none = _mlp()
        prune(m_none, self._dense_cfg("none"), data=_mlp_tokens(7))
        m_ls = _mlp()
        prune(m_ls, self._dense_cfg("ls"), data=_mlp_tokens(7))
        self.assertEqual(_num_params(m_none), _num_params(m_ls))

    def test_dense_default_is_ls_backward_compatible(self):
        m_default = _mlp()
        prune(
            m_default,
            {
                "methods": {
                    "dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.5}}
                },
                "missing_data_policy": "warn_skip",
            },
            data=_mlp_tokens(7),
        )
        m_ls = _mlp()
        prune(m_ls, self._dense_cfg("ls"), data=_mlp_tokens(7))
        self.assertEqual(_num_params(m_default), _num_params(m_ls))

    def test_recovery_is_not_a_no_op(self):

        def consumer_w_b(rec):
            m = _mlp()
            prune(m, self._dense_cfg(rec), data=_mlp_tokens(7))
            c = m.layers[0].dense2
            return (
                c.weight.detach().clone(),
                (c.bias.detach().clone() if c.bias is not None else None),
            )

        w_none, b_none = consumer_w_b("none")
        w_ls, _ = consumer_w_b("ls")
        _, b_bias = consumer_w_b("bias")
        self.assertFalse(torch.allclose(w_ls, w_none))  # ls reconstructed the consumer weights
        self.assertIsNotNone(b_bias)
        self.assertFalse(torch.allclose(b_bias, b_none))  # bias folding changed the consumer bias

    def test_dense_ls_underdetermined_warns_and_falls_back(self):
        import warnings as _w

        tiny = _mlp_tokens(7, n=1)  # 1 batch [2,16]=32 rows << keep(=128) -> underdetermined
        model = _mlp()
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            prune(model, self._dense_cfg("ls"), data=tiny)
        msgs = " ".join(str(c.message) for c in caught)
        self.assertIn("recovery=ls 未执行", msgs)
        self.assertIn("欠定", msgs)
        # falls back to naive slicing: consumer weights must exactly match recovery=none
        # (not overfit-rewritten by underdetermined ls)
        m_none = _mlp()
        prune(m_none, self._dense_cfg("none"), data=tiny)
        self.assertTrue(
            torch.allclose(
                model.layers[0].dense2.weight, m_none.layers[0].dense2.weight
            )
        )

    def test_invalid_recovery_raises(self):
        with self.assertRaises(ValueError):
            prune(_mlp(), self._dense_cfg("bogus"), data=_mlp_tokens(7))

    def test_cnn_each_recovery_prunes(self):
        for rec in ("none", "bias", "ls"):
            with self.subTest(recovery=rec):
                model = _cnn()
                before = _num_params(model)
                prune(
                    model,
                    {
                        "methods": {
                            "cnn": {
                                "name": "reconstruct",
                                "kwargs": {"prune_ratio": 0.3, "recovery": rec},
                            }
                        },
                        "missing_data_policy": "warn_skip",
                    },
                    data=_images(3),
                )
                self.assertLess(_num_params(model), before)
                out = model(_images(5, n=1)[0])
                self.assertEqual(out.shape[0], 8)

    def test_cnn_recovery_is_not_a_no_op(self):
        # batch_size=16, n=6 -> conv1's consumer conv2 reconstruction is determined
        # (measured: conv2 doesn't fall back at this scale)
        data = _images(3, n=6, batch_size=16)
        w_none = self._cnn_recovery("none", data).conv2.weight.detach().clone()
        w_ls = self._cnn_recovery("ls", data).conv2.weight.detach().clone()
        self.assertEqual(w_ls.shape, w_none.shape)  # same prune ratio -> same shape
        # ls really reconstructed conv2 weights (vs none's naive slicing) -> compensation runs, not a silent no-op
        self.assertFalse(torch.allclose(w_ls, w_none))

    def _dense_cfg(self, recovery):
        return {
            "methods": {
                "dense": {
                    "name": "reconstruct",
                    "kwargs": {"prune_ratio": 0.5, "recovery": recovery},
                }
            },
            "missing_data_policy": "warn_skip",
        }

    def _cnn_recovery(self, recovery, data):
        """Prune CNN with a single recovery (sequential targets producer conv1 -> consumer conv2 etc.); return model."""
        model = _cnn()
        prune(
            model,
            {
                "methods": {
                    "cnn": {
                        "name": "reconstruct",
                        "kwargs": {"prune_ratio": 0.3, "recovery": recovery},
                    }
                },
                "missing_data_policy": "warn_skip",
            },
            data=data,
        )
        return model


if __name__ == "__main__":
    unittest.main()
