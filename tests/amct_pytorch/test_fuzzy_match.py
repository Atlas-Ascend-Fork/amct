# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import unittest
import torch
import torch.nn as nn

from amct_pytorch import HIFP8_MINMAX_FUZZY_CFG
from amct_pytorch.config import parse_config
from amct_pytorch.algorithm import AlgorithmRegistry


class TestFuzzyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.ModuleDict({
                    'q_proj': nn.Linear(64, 64),
                    'k_proj': nn.Linear(64, 64),
                    'v_proj': nn.Linear(64, 64),
                }),
                'mlp': nn.ModuleDict({
                    'up_proj': nn.Linear(64, 64),
                    'down_proj': nn.Linear(64, 64),
                })
            })
            for _ in range(2)
        ])
        self.lm_head = nn.Linear(64, 64)


class TestFuzzyMatch(unittest.TestCase):
    def setUp(self):
        self.model = TestFuzzyModel()
    
    def test_fuzzy_match_weights(self):
        detail_config = parse_config(self.model, HIFP8_MINMAX_FUZZY_CFG, AlgorithmRegistry)
        
        down_proj_layers = [name for name in detail_config.keys() if 'down_proj' in name]
        self.assertEqual(len(down_proj_layers), 2)
        
        for layer_name in down_proj_layers:
            self.assertEqual(detail_config[layer_name]['weights_cfg']['strategy'], 'tensor')
            self.assertEqual(detail_config[layer_name]['weights_cfg']['quant_type'], 'int8')
    
    def test_fuzzy_match_inputs(self):
        detail_config = parse_config(self.model, HIFP8_MINMAX_FUZZY_CFG, AlgorithmRegistry)
        
        q_proj_layers = [name for name in detail_config.keys() if 'q_proj' in name]
        self.assertEqual(len(q_proj_layers), 2)
        
        for layer_name in q_proj_layers:
            self.assertEqual(detail_config[layer_name]['inputs_cfg']['strategy'], 'channel')
            self.assertEqual(detail_config[layer_name]['inputs_cfg']['quant_type'], 'int8')
    
    def test_default_config_for_unmatched_layers(self):
        detail_config = parse_config(self.model, HIFP8_MINMAX_FUZZY_CFG, AlgorithmRegistry)
        
        up_proj_layers = [name for name in detail_config.keys() if 'up_proj' in name]
        self.assertEqual(len(up_proj_layers), 2)
        
        for layer_name in up_proj_layers:
            self.assertEqual(detail_config[layer_name]['weights_cfg']['strategy'], 'channel')
    
    def test_skip_layers_with_fuzzy_config(self):
        detail_config = parse_config(self.model, HIFP8_MINMAX_FUZZY_CFG, AlgorithmRegistry)
        
        self.assertNotIn('lm_head', detail_config)


if __name__ == '__main__':
    unittest.main()
