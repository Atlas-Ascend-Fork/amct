# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Experts-per-token resolution and post-prune `model.config` synchronization."""

import os
import sys
import unittest

import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import MiniMoEConfig, MiniMoEModel

from amct_pytorch.pruning.compat import patch_common_config
from amct_pytorch.pruning.domains.moe import DEFAULT_TOP_K, MoEPruningDomain
from amct_pytorch.pruning.config import PruneConfig


class _Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Model(nn.Module):
    """Bare carrier for a config + prune meta, for patch_common_config."""

    def __init__(self, config, meta):
        super().__init__()
        self.config = config
        self._amct_prune_meta = meta


class TopKResolutionTest(unittest.TestCase):
    def setUp(self):
        self.domain = MoEPruningDomain()
        self.model = MiniMoEModel(MiniMoEConfig(num_experts=8, num_experts_per_tok=2))
        self.targets = self.domain.find_targets(self.model, PruneConfig())
        self.assertTrue(self.targets, "mini MoE should expose routed targets")

    def test_reads_experts_per_token_from_the_model_not_the_default(self):
        for target in self.targets:
            self.assertEqual(self.domain.resolve_top_k(self.model, target), 2)

    def test_explicit_override_wins(self):
        target = self.targets[0]
        self.assertEqual(self.domain.resolve_top_k(self.model, target, 5), 5)

    def test_falls_back_to_model_config(self):
        target = self.targets[0]
        block = self.model.layers[0]
        del block.num_experts_per_tok
        self.model.config.num_experts_per_tok = 3
        self.assertEqual(self.domain.resolve_top_k(self.model, target), 3)

    def test_falls_back_to_default_when_nothing_exposes_it(self):
        target = self.targets[0]
        del self.model.layers[0].num_experts_per_tok
        del self.model.config.num_experts_per_tok
        self.assertEqual(self.domain.resolve_top_k(self.model, target), DEFAULT_TOP_K)


class ConfigSyncTest(unittest.TestCase):
    def test_writes_every_expert_count_spelling(self):
        config = _Config(
            num_local_experts=8, num_experts=8, n_routed_experts=8, n_experts=8
        )
        model = _Model(config, {"moe_num_experts": 4, "_moe_widths": {4}})
        patch_common_config(model)
        for name in (
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
            "n_experts",
        ):
            self.assertEqual(getattr(config, name), 4, name)

    def test_clamps_experts_per_token_below_surviving_experts(self):
        config = _Config(n_routed_experts=64, num_experts_per_tok=6)
        model = _Model(config, {"moe_num_experts": 4, "_moe_widths": {4}})
        patch_common_config(model)
        self.assertEqual(config.n_routed_experts, 4)
        self.assertEqual(config.num_experts_per_tok, 4)

    def test_leaves_experts_per_token_alone_when_it_already_fits(self):
        config = _Config(n_routed_experts=64, num_experts_per_tok=6)
        model = _Model(config, {"moe_num_experts": 32, "_moe_widths": {32}})
        patch_common_config(model)
        self.assertEqual(config.num_experts_per_tok, 6)

    def test_non_uniform_counts_skip_the_count_but_still_clamp_topk(self):
        config = _Config(n_routed_experts=64, num_experts_per_tok=6)
        model = _Model(config, {"moe_num_experts": 4, "_moe_widths": {2, 8}})
        patch_common_config(model)
        # A single scalar cannot describe per-layer counts, so it stays put...
        self.assertEqual(config.n_routed_experts, 64)
        # ...but top_k must fit the thinnest layer or that layer cannot route.
        self.assertEqual(config.num_experts_per_tok, 2)

    def test_dense_width_sync_is_unaffected(self):
        config = _Config(intermediate_size=512, ffn_hidden_size=512)
        model = _Model(config, {"dense_hidden_size": 128, "_dense_widths": {128}})
        patch_common_config(model)
        self.assertEqual(config.intermediate_size, 128)
        self.assertEqual(config.ffn_hidden_size, 128)


class NestedConfigSyncTest(unittest.TestCase):
    """Multimodal wrappers keep the language dims one level down, in text_config."""

    def test_expert_count_reaches_the_nested_text_config(self):
        text = _Config(num_experts=256, num_experts_per_tok=8)
        config = _Config(text_config=text, vision_config=_Config(depth=27))
        model = _Model(config, {"moe_num_experts": 230, "_moe_widths": {230}})
        patch_common_config(model)
        self.assertEqual(text.num_experts, 230)

    def test_experts_per_token_is_clamped_in_the_nested_config(self):
        text = _Config(n_routed_experts=64, num_experts_per_tok=6)
        config = _Config(llm_config=text)
        model = _Model(config, {"moe_num_experts": 4, "_moe_widths": {4}})
        patch_common_config(model)
        self.assertEqual(text.n_routed_experts, 4)
        self.assertEqual(text.num_experts_per_tok, 4)

    def test_the_vision_tower_width_is_left_alone(self):
        text = _Config(intermediate_size=512)
        vision = _Config(intermediate_size=1024)
        config = _Config(text_config=text, vision_config=vision)
        model = _Model(config, {"dense_hidden_size": 128, "_dense_widths": {128}})
        patch_common_config(model)
        self.assertEqual(text.intermediate_size, 128)
        # Pruning the language FFN says nothing about the vision tower's width.
        self.assertEqual(vision.intermediate_size, 1024)

    def test_a_flat_config_alongside_a_nested_one_gets_both(self):
        text = _Config(num_experts=256)
        config = _Config(num_experts=256, text_config=text)
        model = _Model(config, {"moe_num_experts": 230, "_moe_widths": {230}})
        patch_common_config(model)
        self.assertEqual(config.num_experts, 230)
        self.assertEqual(text.num_experts, 230)


if __name__ == "__main__":
    unittest.main()
