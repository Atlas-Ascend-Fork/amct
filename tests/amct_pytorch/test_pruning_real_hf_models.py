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
"""AMCT 结构化剪枝 —— 真实 HuggingFace 网络的小尺寸 E2E 实验。

与 ``test_pruning_comprehensive.py`` 的手写 toy 模型不同，本用例直接实例化
*当前 transformers* 提供的真实 ``nn.Module`` 架构（随机初始化、迷你配置，
不联网下载权重），覆盖 dense / cnn / moe 三个剪枝域，验证剪枝器面对真实
模块结构时的行为，并把当前的支持边界固化为回归基线。

测量基线（transformers 5.6.x，CPU，随机权重）：

  dense (LLM FFN):
    - Qwen3       : MLP gate/up/down 被正确识别并剪枝；因层内带 q_norm/k_norm，
                    attention 投影不会被误配，**无需 skip_layers** 即可安全剪枝。
    - Llama/Qwen2 : MLP 可剪，但 dense 域的“相邻 Linear = producer→consumer”
                    启发式会把并列的 q_proj/k_proj 误判成两层 FFN。**必须**
                    设置 skip_layers=['self_attn'] 才能避免破坏注意力。
  cnn (vision):
    - RegNet      : 顺序卷积链被识别，通道剪枝后前向正常。
    - ResNet(basic): 每个 conv 被独立 block 包裹，producer/consumer 卷积不是
                    同一 parent 的兄弟节点 → 0 个目标（静默 no-op）。
  moe:
    - Mixtral 等  : 现代 transformers 把 experts 融合成单个 batched 模块
                    （非 nn.ModuleList）、router 为非 nn.Linear → MoE 域找不到
                    任何目标。当前实现仅支持“老式 ModuleList experts”结构。

这些边界用例（ResNet / Mixtral）以断言形式记录“今天的真实行为”，一旦
transformers 结构变化或剪枝器新增支持，测试会立刻提示，便于及时更新。
"""
import sys
import unittest

import torch

try:
    import transformers  # noqa: F401
    _HAS_TF = True
except Exception:  # pragma: no cover - 环境无 transformers 时整组跳过
    _HAS_TF = False

from amct_pytorch.experimental.pruning import prune, get_last_report


requires_tf = unittest.skipUnless(_HAS_TF, "transformers not installed")


def _has(*names):
    return all(hasattr(transformers, n) for n in names)


def _token_calib(vocab=128, batch=2, seq=16, n=6):
    return [torch.randint(0, vocab, (batch, seq)) for _ in range(n)]


def _image_calib(batch=2, ch=3, hw=64, n=6):
    return [torch.randn(batch, ch, hw, hw) for _ in range(n)]


def _params(model):
    return sum(p.numel() for p in model.parameters())


def _no_targets(report):
    """report 中是否存在“未找到可剪枝目标”的事件。"""
    return any(
        ev.module == "<none>" or "No compatible pruning targets" in ev.detail
        for ev in report.events
    )


# ============================================================================
# Dense —— 真实 LLM 的 FFN 剪枝
# ============================================================================
@requires_tf
class TestDenseRealLLM(unittest.TestCase):
    """真实解码器 LLM 的 SwiGLU/MLP 中间维剪枝。"""

    def _tiny_cfg(self, Cfg):
        return Cfg(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
        )

    @unittest.skipUnless(_HAS_TF and _has("Qwen3Config", "Qwen3ForCausalLM"),
                         "Qwen3 unavailable")
    def test_qwen3_mlp_prune_no_skip_needed(self):
        """Qwen3：层内 q_norm/k_norm 使注意力投影不被误配，无需 skip_layers。"""
        torch.manual_seed(0)
        model = transformers.Qwen3ForCausalLM(self._tiny_cfg(transformers.Qwen3Config)).eval()
        ids = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out_before = model(ids).logits

        inter_before = model.model.layers[0].mlp.gate_proj.out_features
        qproj_before = model.model.layers[0].self_attn.q_proj.out_features
        p_before = _params(model)

        cfg = {
            "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_token_calib())
        report = get_last_report()

        inter_after = model.model.layers[0].mlp.gate_proj.out_features
        qproj_after = model.model.layers[0].self_attn.q_proj.out_features

        # MLP 中间维被剪掉一半
        self.assertEqual(inter_after, inter_before // 2)
        # 注意力投影完好（未被误剪）
        self.assertEqual(qproj_after, qproj_before)
        self.assertLess(report.params_after, report.params_before)
        self.assertLess(_params(model), p_before)

        # 剪枝后仍可正常前向，且输出有限
        with torch.no_grad():
            out_after = model(ids).logits
        self.assertEqual(out_after.shape, out_before.shape)
        self.assertTrue(torch.isfinite(out_after).all())

    @unittest.skipUnless(_HAS_TF and _has("LlamaConfig", "LlamaForCausalLM"),
                         "Llama unavailable")
    def test_llama_mlp_prune_with_attention_skipped(self):
        """Llama：必须 skip_layers=['self_attn'] 才能只剪 MLP、保住注意力。"""
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(self._tiny_cfg(transformers.LlamaConfig)).eval()
        ids = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out_before = model(ids).logits

        inter_before = model.model.layers[0].mlp.gate_proj.out_features
        qproj_before = model.model.layers[0].self_attn.q_proj.out_features

        cfg = {
            "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
            "skip_layers": ["self_attn"],
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_token_calib())
        report = get_last_report()

        self.assertEqual(model.model.layers[0].mlp.gate_proj.out_features, inter_before // 2)
        # 注意力被 skip_layers 排除，保持原样
        self.assertEqual(model.model.layers[0].self_attn.q_proj.out_features, qproj_before)
        self.assertLess(report.params_after, report.params_before)

        with torch.no_grad():
            out_after = model(ids).logits
        self.assertTrue(torch.isfinite(out_after).all())
        # low_variance 删除的是低方差神经元 —— 输出应基本保持
        mse = torch.mean((out_before - out_after) ** 2).item()
        self.assertLess(mse, 1.0)

    @unittest.skipUnless(_HAS_TF and _has("LlamaConfig", "LlamaForCausalLM"),
                         "Llama unavailable")
    def test_llama_without_skip_corrupts_attention_known_limitation(self):
        """已知边界：不设 skip_layers 时，dense 域会把 q_proj/k_proj 误判为
        两层 FFN 并剪掉 q_proj 的输出维，导致注意力维度不一致、前向崩溃。

        固化“当前真实行为”：剪枝后 q_proj 被改尺寸（< 原值），前向抛错。
        若未来 dense 检测器学会跳过并列注意力投影，本用例会失败提示更新。
        """
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(self._tiny_cfg(transformers.LlamaConfig)).eval()
        qproj_before = model.model.layers[0].self_attn.q_proj.out_features
        ids = torch.randint(0, 128, (2, 16))

        cfg = {
            "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_token_calib())

        qproj_after = model.model.layers[0].self_attn.q_proj.out_features
        # 注意力投影被误剪
        self.assertLess(qproj_after, qproj_before)
        # 维度不一致 → 前向崩溃
        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                model(ids)


# ============================================================================
# CNN —— 真实视觉网络的通道剪枝
# ============================================================================
@requires_tf
class TestCNNRealVision(unittest.TestCase):

    @unittest.skipUnless(_HAS_TF and _has("RegNetConfig", "RegNetForImageClassification"),
                         "RegNet unavailable")
    def test_regnet_channel_prune(self):
        """RegNet：顺序卷积链被识别，通道剪枝后参数下降且前向正常。"""
        torch.manual_seed(0)
        model = transformers.RegNetForImageClassification(
            transformers.RegNetConfig(num_channels=3, num_labels=10)
        ).eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out_before = model(x).logits
        p_before = _params(model)

        cfg = {
            "methods": {"cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.3}}},
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_image_calib())
        report = get_last_report()

        self.assertLess(report.params_after, report.params_before)
        self.assertLess(_params(model), p_before)
        with torch.no_grad():
            out_after = model(x).logits
        self.assertEqual(out_after.shape, out_before.shape)
        self.assertTrue(torch.isfinite(out_after).all())

    @unittest.skipUnless(_HAS_TF and _has("ResNetConfig", "ResNetForImageClassification"),
                         "ResNet unavailable")
    def test_resnet_basic_block_no_targets_known_boundary(self):
        """已知边界：ResNet 把每个 conv 包进独立 block，producer/consumer 卷积
        不是同一 parent 的兄弟节点，cnn 域基于“同级兄弟”的检测找不到目标 →
        参数零变化（静默 no-op）。固化该行为以便未来扩展检测时提示。
        """
        torch.manual_seed(0)
        model = transformers.ResNetForImageClassification(
            transformers.ResNetConfig(
                num_channels=3, embedding_size=16, hidden_sizes=[32, 64],
                depths=[1, 1], layer_type="basic", num_labels=10,
            )
        ).eval()
        p_before = _params(model)

        cfg = {
            "methods": {"cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.3}}},
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_image_calib())
        report = get_last_report()

        self.assertEqual(_params(model), p_before)
        self.assertEqual(report.params_after, report.params_before)
        self.assertTrue(_no_targets(report))


# ============================================================================
# MoE —— 现代 HuggingFace MoE 的当前支持边界
# ============================================================================
@requires_tf
class TestMoERealHF(unittest.TestCase):

    @unittest.skipUnless(_HAS_TF and _has("MixtralConfig", "MixtralForCausalLM"),
                         "Mixtral unavailable")
    def test_modern_fused_moe_not_detected_known_boundary(self):
        """已知边界：现代 transformers 把 Mixtral experts 融合为单个 batched
        模块（非 nn.ModuleList）、router 为非 nn.Linear，moe 域（要求
        ModuleList experts + Linear gate）找不到目标 → 专家数与参数不变。

        固化该边界：experts 容器不是 ModuleList，且 moe stage 无目标。
        当 transformers 回退到 ModuleList 或剪枝器支持 fused experts 时提示更新。
        """
        torch.manual_seed(0)
        model = transformers.MixtralForCausalLM(
            transformers.MixtralConfig(
                vocab_size=128, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=64, num_local_experts=6, num_experts_per_tok=2,
            )
        ).eval()

        experts = model.model.layers[0].mlp.experts
        # 现代结构：experts 已被融合，不再是 ModuleList
        self.assertNotIsInstance(experts, torch.nn.ModuleList)

        ids = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out_before = model(ids).logits
        p_before = _params(model)

        cfg = {
            "methods": {"moe": {"name": "mass_variance",
                                "kwargs": {"prune_ratio": 0.5, "boundary": 10}}},
            "skip_layers": ["self_attn"],
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_token_calib())
        report = get_last_report()

        # MoE 域没有可剪目标
        moe_events = [ev for ev in report.events if ev.stage == "moe"]
        self.assertTrue(moe_events)
        self.assertTrue(all(ev.module == "<none>" for ev in moe_events))

        # 前向仍正常
        with torch.no_grad():
            out_after = model(ids).logits
        self.assertEqual(out_after.shape, out_before.shape)
        self.assertTrue(torch.isfinite(out_after).all())


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
