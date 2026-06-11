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
"""守卫式 recovery-HYBRID 用例：recovery MENU 验证集择优、绝不劣于 fallback(none)、参数下降、前向正常。

并覆盖 dense_reconstruct / cnn_reconstruct 新增的 ``recovery`` kwarg（ls/bias/none）的直接调用与
向后兼容（默认 ls == 旧行为）。
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
from amct_pytorch.experimental.pruning import (
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
    """轻量训练的 mini-MLP：随机初始化的玩具模型输出几乎饱和（任何剪枝后保真度都≈不变），
    无法体现 recovery 的优劣阶梯；训练出非平凡的输入相关结构后，ls 重构才会真正优于 none。"""
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
        """recovery 阶梯（验证集质量）：在**训练过**的模型上、校准行数 >= keep（LS 定解），
        最小二乘 ls 不劣于不补偿 none，也不劣于均值折叠 bias —— 证明 ls 是头部 recovery，
        而非 ``gain_over_fallback >= 0`` 那条同义反复（任何 recovery 都满足）。

        诚实范围：只断言 impl 真正保证的偏序（ls>=none, ls>=bias）。不断言 bias>=none：
        impl 文档与实测均表明小校准下 bias 折叠最易过拟合，可能反低于 none（见 recovery_hybrid 注释）。
        keep = round(256*(1-0.5)) = 128；校准行数 = n*2*16 = 8*32 = 256 >= 128，故 LS 是定解。
        """
        # apply=False：只比较 MENU 各 recovery 的验证质量，不改原模型。
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
        # ls 是最优 recovery：不劣于不补偿、也不劣于 bias（守卫据此会选 ls 而非 fallback）。
        self.assertGreaterEqual(q["ls"], q["none"])
        self.assertGreaterEqual(q["ls"], q["bias"])
        # 阶梯真生效：ls 被守卫选中，且相对 fallback(none) 严格有正增益。
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
        """小校准安全旋钮：recovery_menu=("none","ls") 只评估这两个。"""
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
        self.assertEqual(_num_params(model), before)  # 原模型不动
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
        """recovery 只改权重/偏置，不改保留哪些神经元 → none 与 ls 参数量一致。"""
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
        """ls 改写 consumer 权重、bias 改写 consumer 偏置（相对 none 朴素切片）→ 补偿真的执行了，
        不是静默 no-op。（toy 随机模型上输出已饱和无法比较精度，故在结构层面验证补偿生效。）"""

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
        self.assertFalse(torch.allclose(w_ls, w_none))  # ls 重构了 consumer 权重
        self.assertIsNotNone(b_bias)
        self.assertFalse(torch.allclose(b_bias, b_none))  # bias 折叠改了 consumer 偏置

    def test_dense_ls_underdetermined_warns_and_falls_back(self):
        """[回归] dense recovery=ls 且校准行数 < 保留神经元 → 欠定：必须告警 + 退回朴素切片，
        不静默过拟合（README 声明此守卫；dense 路径曾缺失，仅 cnn 有）。"""
        import warnings as _w

        tiny = _mlp_tokens(7, n=1)  # 1 批 [2,16]=32 行 « keep(=128) → 欠定
        model = _mlp()
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            prune(model, self._dense_cfg("ls"), data=tiny)
        msgs = " ".join(str(c.message) for c in caught)
        self.assertIn("recovery=ls 未执行", msgs)
        self.assertIn("欠定", msgs)
        # 退回朴素切片：consumer 权重应与 recovery=none 完全一致（未被欠定 ls 过拟合改写）
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
        """镜像 dense 版：CNN 的 ls 重构真的改写了 **可重构 consumer**(conv1→conv2 的 conv2,
        Conv2d consumer)的权重 —— 不是静默退回朴素切片的 no-op。

        关键：用足够大的校准（行数 = N*L_out >= keep*kh*kw）让重构在 conv2 上**真正执行**。
        小校准会退回朴素切片(``退回朴素切片`` 告警)，那样 ls 与 none 权重相同、特性变 no-op。
        （注：conv3 的 consumer 是 fc(Linear)，结构上永远只能朴素切片，故此处针对 conv2 断言。）
        """
        # batch_size=16、n=6 → conv1 的 consumer conv2 重构定解（实测此规模下 conv2 不退回）。
        data = _images(3, n=6, batch_size=16)
        w_none = self._cnn_recovery("none", data).conv2.weight.detach().clone()
        w_ls = self._cnn_recovery("ls", data).conv2.weight.detach().clone()
        self.assertEqual(w_ls.shape, w_none.shape)  # 同一剪枝率 → 同形状
        # ls 真的重构了 conv2 权重（相对 none 朴素切片）→ 补偿确实执行，非静默 no-op。
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
        """用单一 recovery 直接剪 CNN（producer conv1 → consumer conv2 等顺序目标），返回剪后模型。"""
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
