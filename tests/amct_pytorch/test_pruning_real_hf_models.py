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
"""Small-scale E2E tests of AMCT structured pruning on real HuggingFace networks."""

import sys
import unittest

import torch

try:
    import transformers

    _HAS_TF = True
except Exception:  # pragma: no cover - skip the whole group when transformers is absent
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
    return any(
        ev.module == "<none>" or "No compatible pruning targets" in ev.detail
        for ev in report.events
    )


# ============================================================================
# Dense -- FFN pruning of real LLMs
# ============================================================================
@requires_tf
class TestDenseRealLLM(unittest.TestCase):

    @unittest.skipUnless(
        _HAS_TF and _has("Qwen3Config", "Qwen3ForCausalLM"), "Qwen3 unavailable"
    )
    def test_qwen3_mlp_prune_no_skip_needed(self):
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

        # MLP intermediate dim halved
        self.assertEqual(inter_after, inter_before // 2)
        # attention projection intact (not mispruned)
        self.assertEqual(qproj_after, qproj_before)
        self.assertLess(report.params_after, report.params_before)
        self.assertLess(_params(model), p_before)

        # still forwards correctly with finite output after pruning
        with torch.no_grad():
            out_after = model(ids).logits
        self.assertEqual(out_after.shape, out_before.shape)
        self.assertTrue(torch.isfinite(out_after).all())

    @unittest.skipUnless(
        _HAS_TF and _has("LlamaConfig", "LlamaForCausalLM"), "Llama unavailable"
    )
    def test_llama_mlp_prune_with_attention_skipped(self):
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
        # attention excluded by skip_layers, left unchanged
        self.assertEqual(
            model.model.layers[0].self_attn.q_proj.out_features, qproj_before
        )
        self.assertLess(report.params_after, report.params_before)

        with torch.no_grad():
            out_after = model(ids).logits
        self.assertTrue(torch.isfinite(out_after).all())
        # low_variance drops low-variance neurons -- output should be mostly preserved
        mse = torch.mean((out_before - out_after) ** 2).item()
        self.assertLess(mse, 1.0)

    @unittest.skipUnless(
        _HAS_TF and _has("LlamaConfig", "LlamaForCausalLM"), "Llama unavailable"
    )
    def test_llama_without_skip_preserves_attention(self):
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

        # attention projection unchanged
        self.assertEqual(
            model.model.layers[0].self_attn.q_proj.out_features, qproj_before
        )
        # MLP intermediate dim pruned
        self.assertEqual(
            model.model.layers[0].mlp.gate_proj.out_features, inter_before // 2
        )
        # forward works
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
# CNN -- channel pruning of real vision networks
# ============================================================================
@requires_tf
class TestCNNRealVision(unittest.TestCase):
    @unittest.skipUnless(
        _HAS_TF and _has("RegNetConfig", "RegNetForImageClassification"),
        "RegNet unavailable",
    )
    def test_regnet_channel_prune(self):
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
# MoE -- pruning of modern HuggingFace fused (batched) experts
# ============================================================================
@requires_tf
class TestMoERealHF(unittest.TestCase):
    @unittest.skipUnless(
        _HAS_TF and _has("MixtralConfig", "MixtralForCausalLM"), "Mixtral unavailable"
    )
    def test_mixtral_fused_expert_pruning(self):
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
        # modern structure: experts are fused, no longer a ModuleList
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

        # both expert count and params drop
        self.assertLess(model.model.layers[0].mlp.experts.num_experts, experts_before)
        self.assertLess(_params(model), p_before)

        # forward works
        with torch.no_grad():
            out_after = model(ids).logits
        self.assertEqual(out_after.shape, out_before.shape)
        self.assertTrue(torch.isfinite(out_after).all())

    @unittest.skipUnless(
        _HAS_TF and _has("Qwen3MoeConfig", "Qwen3MoeForCausalLM"),
        "Qwen3Moe unavailable",
    )
    def test_qwen3moe_fused_expert_pruning(self):
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

    @unittest.skipUnless(
        _HAS_TF and _has("GraniteMoeConfig", "GraniteMoeForCausalLM"),
        "GraniteMoe unavailable",
    )
    def test_granitemoe_fused_two_tensor_experts(self):
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

    def _build_or_skip(self, fn):
        try:
            return fn().eval()
        except (TypeError, ValueError, AttributeError, ImportError, KeyError) as exc:
            self.skipTest(
                f"config unsupported in this transformers: {type(exc).__name__}: {exc}"
            )
        return None

    def _assert_prune_forward(self, model, vocab=128):
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


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
