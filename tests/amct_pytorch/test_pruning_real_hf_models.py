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
模块结构时的行为。

测量基线（transformers 5.6.x，CPU，随机权重）：

  dense (LLM FFN):
    - Qwen3 / Llama / Qwen2 / Mistral : MLP gate/up/down 被识别并剪枝；dense 域
                    现已结构化排除并列的注意力投影（q/k/v/o 共享输入、且不满足
                    in→inter→in 残差维度复原），**无需 skip_layers** 即可安全剪枝。
  cnn (vision):
    - variance_channel / reconstruct: 均用 cnn 域（同级 conv→bn→consumer 检测），适合
                    平面/顺序卷积链；reconstruct 额外做最小二乘输出补偿（低破坏）。
                    ResNet 等把 conv 包进独立 block 的结构 0 目标（同级检测的已知边界；
                    原 torch_pruning 依赖图方案因 acc 破坏严重已移除）。
  moe:
    - Mixtral / Qwen3Moe 等 : 现代 transformers 把 experts 融合成单个 batched 模块
                    （非 nn.ModuleList）、router 为 *TopKRouter（非 Linear）。moe 域
                    现已支持该融合结构，沿专家维切片批张量完成剪枝。

ResNet 仍以断言记录当前边界；一旦其检测获得支持，测试会提示更新。
"""

import sys
import unittest

import torch

try:
    import transformers

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

    @unittest.skipUnless(
        _HAS_TF and _has("Qwen3Config", "Qwen3ForCausalLM"), "Qwen3 unavailable"
    )
    def test_qwen3_mlp_prune_no_skip_needed(self):
        """Qwen3：层内 q_norm/k_norm 使注意力投影不被误配，无需 skip_layers。"""
        torch.manual_seed(0)
        model = transformers.Qwen3ForCausalLM(
            self._tiny_cfg(transformers.Qwen3Config)
        ).eval()
        ids = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out_before = model(ids).logits

        inter_before = model.model.layers[0].mlp.gate_proj.out_features
        qproj_before = model.model.layers[0].self_attn.q_proj.out_features
        p_before = _params(model)

        cfg = {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
            },
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

    @unittest.skipUnless(
        _HAS_TF and _has("LlamaConfig", "LlamaForCausalLM"), "Llama unavailable"
    )
    def test_llama_mlp_prune_with_attention_skipped(self):
        """Llama：必须 skip_layers=['self_attn'] 才能只剪 MLP、保住注意力。"""
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(
            self._tiny_cfg(transformers.LlamaConfig)
        ).eval()
        ids = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out_before = model(ids).logits

        inter_before = model.model.layers[0].mlp.gate_proj.out_features
        qproj_before = model.model.layers[0].self_attn.q_proj.out_features

        cfg = {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
            },
            "skip_layers": ["self_attn"],
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_token_calib())
        report = get_last_report()

        self.assertEqual(
            model.model.layers[0].mlp.gate_proj.out_features, inter_before // 2
        )
        # 注意力被 skip_layers 排除，保持原样
        self.assertEqual(
            model.model.layers[0].self_attn.q_proj.out_features, qproj_before
        )
        self.assertLess(report.params_after, report.params_before)

        with torch.no_grad():
            out_after = model(ids).logits
        self.assertTrue(torch.isfinite(out_after).all())
        # low_variance 删除的是低方差神经元 —— 输出应基本保持
        mse = torch.mean((out_before - out_after) ** 2).item()
        self.assertLess(mse, 1.0)

    @unittest.skipUnless(
        _HAS_TF and _has("LlamaConfig", "LlamaForCausalLM"), "Llama unavailable"
    )
    def test_llama_without_skip_preserves_attention(self):
        """dense 域现在能结构化识别并列的注意力投影（q/k/v/o 共享输入、且
        down.out != gate.in），不会把它们误判为两层 FFN。因此即便 *不设*
        skip_layers，对 Llama 直接剪枝也只动 MLP、保住注意力、前向正常。
        """
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(
            self._tiny_cfg(transformers.LlamaConfig)
        ).eval()
        qproj_before = model.model.layers[0].self_attn.q_proj.out_features
        inter_before = model.model.layers[0].mlp.gate_proj.out_features
        ids = torch.randint(0, 128, (2, 16))

        cfg = {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_token_calib())

        # 注意力投影保持原样
        self.assertEqual(
            model.model.layers[0].self_attn.q_proj.out_features, qproj_before
        )
        # MLP 中间维被剪
        self.assertEqual(
            model.model.layers[0].mlp.gate_proj.out_features, inter_before // 2
        )
        # 前向正常
        with torch.no_grad():
            out = model(ids)
        self.assertTrue(torch.isfinite(out.logits).all())

    def _tiny_cfg(self, config_cls):
        return config_cls(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )


# ============================================================================
# CNN —— 真实视觉网络的通道剪枝
# ============================================================================
@requires_tf
class TestCNNRealVision(unittest.TestCase):
    @unittest.skipUnless(
        _HAS_TF and _has("RegNetConfig", "RegNetForImageClassification"),
        "RegNet unavailable",
    )
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
            "methods": {
                "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.3}}
            },
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

    @unittest.skipUnless(
        _HAS_TF and _has("ResNetConfig", "ResNetForImageClassification"),
        "ResNet unavailable",
    )
    def test_resnet_basic_block_no_targets_known_boundary(self):
        """`variance_channel`(启发式) 的边界：ResNet 把每个 conv 包进独立 block，
        producer/consumer 卷积不是同 parent 兄弟，基于“同级兄弟”的检测 0 目标 →
        参数零变化。这是 cnn 域同级检测的已知边界（原 torch_pruning 依赖图方案能剪
        ResNet，但因 acc 破坏严重已移除，改用同域的输出保留 reconstruct）。
        """
        torch.manual_seed(0)
        model = transformers.ResNetForImageClassification(
            transformers.ResNetConfig(
                num_channels=3,
                embedding_size=16,
                hidden_sizes=[32, 64],
                depths=[1, 1],
                layer_type="basic",
                num_labels=10,
            )
        ).eval()
        p_before = _params(model)

        cfg = {
            "methods": {
                "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.3}}
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_image_calib())
        report = get_last_report()

        self.assertEqual(_params(model), p_before)
        self.assertEqual(report.params_after, report.params_before)
        self.assertTrue(_no_targets(report))


# ============================================================================
# MoE —— 现代 HuggingFace 融合(批处理)专家剪枝
# ============================================================================
@requires_tf
class TestMoERealHF(unittest.TestCase):
    @unittest.skipUnless(
        _HAS_TF and _has("MixtralConfig", "MixtralForCausalLM"), "Mixtral unavailable"
    )
    def test_mixtral_fused_expert_pruning(self):
        """现代 transformers 把 Mixtral experts 融合为单个 batched 模块
        （非 nn.ModuleList）、router 为 *TopKRouter（非 Linear）。moe 域现已
        支持该融合结构：沿专家维切片批张量 → 专家数下降、参数减少、前向正常。
        """
        torch.manual_seed(0)
        model = transformers.MixtralForCausalLM(
            transformers.MixtralConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                num_local_experts=8,
                num_experts_per_tok=2,
            )
        ).eval()

        experts = model.model.layers[0].mlp.experts
        # 现代结构：experts 已被融合，不再是 ModuleList
        self.assertNotIsInstance(experts, torch.nn.ModuleList)
        experts_before = experts.num_experts

        ids = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out_before = model(ids).logits
        p_before = _params(model)

        cfg = {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {"prune_ratio": 0.5, "boundary": 10, "top_k": 2},
                }
            },
            "skip_layers": ["self_attn"],
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_token_calib())

        # 专家数与参数都下降
        self.assertLess(model.model.layers[0].mlp.experts.num_experts, experts_before)
        self.assertLess(_params(model), p_before)

        # 前向正常
        with torch.no_grad():
            out_after = model(ids).logits
        self.assertEqual(out_after.shape, out_before.shape)
        self.assertTrue(torch.isfinite(out_after).all())

    @unittest.skipUnless(
        _HAS_TF and _has("Qwen3MoeConfig", "Qwen3MoeForCausalLM"),
        "Qwen3Moe unavailable",
    )
    def test_qwen3moe_fused_expert_pruning(self):
        """另一融合专家族（Qwen3MoeExperts，专家维批张量形状不同）同样被支持。"""
        torch.manual_seed(0)
        model = transformers.Qwen3MoeForCausalLM(
            transformers.Qwen3MoeConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                moe_intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_experts=8,
                num_experts_per_tok=2,
                max_position_embeddings=64,
                decoder_sparse_step=1,
            )
        ).eval()
        experts_before = model.model.layers[0].mlp.experts.num_experts
        ids = torch.randint(0, 128, (2, 16))
        p_before = _params(model)

        cfg = {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {"prune_ratio": 0.5, "boundary": 10, "top_k": 2},
                }
            },
            "skip_layers": ["self_attn"],
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_token_calib())

        self.assertLess(model.model.layers[0].mlp.experts.num_experts, experts_before)
        self.assertLess(_params(model), p_before)
        with torch.no_grad():
            out = model(ids).logits
        self.assertTrue(torch.isfinite(out).all())

    def _build_or_skip(self, fn):
        """实例化迷你真实模型；transformers 版本字段漂移导致构建失败时跳过（而非误判失败）。"""
        try:
            return fn().eval()
        except (TypeError, ValueError, AttributeError, ImportError, KeyError) as exc:
            self.skipTest(
                f"config unsupported in this transformers: {type(exc).__name__}: {exc}"
            )

    def _assert_prune_forward(self, model, vocab=128):
        """剪 50% 专家（activation_count）→ 参数下降 + 前向有限。覆盖融合结构落地的健壮性。"""
        ids = torch.randint(0, vocab, (2, 16))
        with torch.no_grad():
            shape_before = model(ids).logits.shape
        p_before = _params(model)
        cfg = {
            "methods": {
                "moe": {
                    "name": "activation_count",
                    "kwargs": {"prune_ratio": 0.5, "top_k": 2},
                }
            },
            "skip_layers": ["self_attn"],
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_token_calib(vocab=vocab))
        self.assertLess(_params(model), p_before)
        with torch.no_grad():
            out = model(ids).logits
        self.assertEqual(out.shape, shape_before)
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(
        _HAS_TF and _has("GraniteMoeConfig", "GraniteMoeForCausalLM"),
        "GraniteMoe unavailable",
    )
    def test_granitemoe_fused_two_tensor_experts(self):
        """Granite：专家在 block_sparse_moe.input_linear/output_linear（两个 ParallelExperts，
        非 experts/* 容器）。moe 域现能识别该融合布局并切两路专家张量 + 嵌套门控 Linear → 可剪。"""
        torch.manual_seed(0)
        model = self._build_or_skip(
            lambda: transformers.GraniteMoeForCausalLM(
                transformers.GraniteMoeConfig(
                    vocab_size=128,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=64,
                    num_local_experts=8,
                    num_experts_per_tok=2,
                )
            )
        )
        self._assert_prune_forward(model)

    @unittest.skipUnless(
        _HAS_TF and _has("DeepseekV3Config", "DeepseekV3ForCausalLM"),
        "DeepseekV3 unavailable",
    )
    def test_deepseekv3_grouped_router_pruning(self):
        """DeepseekV3：分组路由（gate 读 self.n_routed_experts/n_group 建分组掩码）。moe 域现会
        同步 n_routed_experts 并把分组塌缩为 1 → 剪枝后前向不再维度不匹配。"""
        torch.manual_seed(0)
        model = self._build_or_skip(
            lambda: transformers.DeepseekV3ForCausalLM(
                transformers.DeepseekV3Config(
                    vocab_size=128,
                    hidden_size=64,
                    intermediate_size=128,
                    moe_intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=64,
                    n_routed_experts=8,
                    num_experts_per_tok=2,
                    n_group=2,
                    topk_group=1,
                    n_shared_experts=1,
                    first_k_dense_replace=0,
                    qk_rope_head_dim=16,
                    qk_nope_head_dim=16,
                    v_head_dim=16,
                    kv_lora_rank=16,
                    q_lora_rank=None,
                )
            )
        )
        self._assert_prune_forward(model)

    @unittest.skipUnless(
        _HAS_TF and _has("Ernie4_5_MoeConfig", "Ernie4_5_MoeForCausalLM"),
        "Ernie4_5_Moe unavailable",
    )
    def test_ernie45moe_nested_router_bias_pruning(self):
        """Ernie4.5-MoE：路由器内嵌 moe_statics.e_score_correction_bias 形如 (groups, experts)，
        专家维是最后一维。moe 域递归切片路由器子树（按唯一等于专家数的维）→ 剪枝后前向正常。"""
        torch.manual_seed(0)
        model = self._build_or_skip(
            lambda: transformers.Ernie4_5_MoeForCausalLM(
                transformers.Ernie4_5_MoeConfig(
                    vocab_size=128,
                    hidden_size=64,
                    intermediate_size=128,
                    moe_intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=64,
                    moe_num_experts=8,
                    moe_k=2,
                    moe_layer_start_index=0,
                )
            )
        )
        self._assert_prune_forward(model)


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
