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

from amct_pytorch.config.utils import is_fuzzy_pattern, get_fuzzy_suffix, match_fuzzy_pattern, find_matching_pattern
from amct_pytorch import parse_config
from amct_pytorch.algorithm import AlgorithmRegistry


class TestModelForFuzzyMatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(64, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16)
        ])
        self.down_proj = nn.Linear(16, 8)
        self.self_attn = nn.ModuleDict({
            'q_proj': nn.Linear(8, 8),
            'k_proj': nn.Linear(8, 8),
            'v_proj': nn.Linear(8, 8)
        })
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.down_proj(x)
        return x


class TestFuzzyMatchingUtils(unittest.TestCase):
    def test_is_fuzzy_pattern(self):
        self.assertTrue(is_fuzzy_pattern('*self.atten.q_proj'))
        self.assertTrue(is_fuzzy_pattern('*down_proj.weight'))
        self.assertTrue(is_fuzzy_pattern('*self_attn.kvcache'))
        self.assertFalse(is_fuzzy_pattern('self.atten.q_proj'))
        self.assertFalse(is_fuzzy_pattern(''))
        self.assertFalse(is_fuzzy_pattern('layer'))
    
    def test_get_fuzzy_suffix(self):
        self.assertEqual(get_fuzzy_suffix('*self.atten.q_proj'), 'self.atten.q_proj')
        self.assertEqual(get_fuzzy_suffix('*down_proj.weight'), 'down_proj.weight')
        self.assertEqual(get_fuzzy_suffix('*self_attn.kvcache'), 'self_attn.kvcache')
        self.assertIsNone(get_fuzzy_suffix('self.atten.q_proj'))
        self.assertIsNone(get_fuzzy_suffix(''))
    
    def test_match_fuzzy_pattern(self):
        self.assertTrue(match_fuzzy_pattern('model.layers.0.self.atten.q_proj', '*self.atten.q_proj'))
        self.assertTrue(match_fuzzy_pattern('transformer.layers.1.down_proj.weight', '*down_proj.weight'))
        self.assertTrue(match_fuzzy_pattern('encoder.self_attn.kvcache', '*self_attn.kvcache'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.self.atten.q_proj', '*down_proj.weight'))
        self.assertFalse(match_fuzzy_pattern('transformer.layers.1.down_proj.weight', '*self.atten.q_proj'))
        self.assertTrue(match_fuzzy_pattern('exact.match', 'exact.match'))
    
    def test_find_matching_pattern(self):
        patterns = ['*self.atten.q_proj', '*down_proj.weight', '*self_attn.kvcache']
        
        self.assertEqual(find_matching_pattern('model.layers.0.self.atten.q_proj', patterns), '*self.atten.q_proj')
        self.assertEqual(find_matching_pattern('transformer.layers.1.down_proj.weight', patterns), '*down_proj.weight')
        self.assertEqual(find_matching_pattern('encoder.self_attn.kvcache', patterns), '*self_attn.kvcache')
        self.assertIsNone(find_matching_pattern('no.match', patterns))
        
        patterns_with_exact = ['*self.atten.q_proj', 'exact.match']
        self.assertEqual(find_matching_pattern('exact.match', patterns_with_exact), 'exact.match')


class TestFuzzyMatchingConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModelForFuzzyMatch()
        print('TestFuzzyMatchingConfig START!')
    
    @classmethod
    def tearDownClass(cls):
        print('TestFuzzyMatchingConfig END!')
    
    def test_fuzzy_matching_override_weights_config(self):
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'minmax'},
            'override_layer_configs': {
                '*down_proj': {
                    'quant_cfg': {
                        'weights': {
                            'type': 'int4',
                            'symmetric': True,
                            'strategy': 'channel',
                        }
                    }
                }
            }
        }
        
        detail_config = parse_config(self.test_model, config, AlgorithmRegistry)
        
        self.assertIn('down_proj', detail_config)
        self.assertEqual(detail_config['down_proj']['weights_cfg']['quant_type'], 'int4')
        
        for i in range(3):
            layer_name = f'layers.{i}'
            self.assertIn(layer_name, detail_config)
            self.assertEqual(detail_config[layer_name]['weights_cfg']['quant_type'], 'int8')
    
    def test_fuzzy_matching_override_inputs_config(self):
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'minmax'},
            'override_layer_configs': {
                '*q_proj': {
                    'quant_cfg': {
                        'inputs': {
                            'type': 'int8',
                            'symmetric': True,
                            'strategy': 'tensor',
                            'enable_quant': True
                        }
                    }
                }
            }
        }
        
        detail_config = parse_config(self.test_model, config, AlgorithmRegistry)
        
        self.assertIn('self_attn.q_proj', detail_config)
        self.assertTrue(detail_config['self_attn.q_proj']['inputs_cfg']['enable_quant'])
        
        self.assertIn('self_attn.k_proj', detail_config)
        self.assertFalse(detail_config['self_attn.k_proj']['inputs_cfg']['enable_quant'])
    
    def test_fuzzy_matching_multiple_patterns(self):
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'minmax'},
            'override_layer_configs': {
                '*down_proj': {
                    'quant_cfg': {
                        'weights': {
                            'type': 'int4',
                            'symmetric': True,
                            'strategy': 'channel',
                        }
                    }
                },
                '*q_proj': {
                    'quant_cfg': {
                        'inputs': {
                            'type': 'int8',
                            'symmetric': True,
                            'strategy': 'tensor',
                            'enable_quant': True
                        }
                    }
                }
            }
        }
        
        detail_config = parse_config(self.test_model, config, AlgorithmRegistry)
        
        self.assertEqual(detail_config['down_proj']['weights_cfg']['quant_type'], 'int4')
        self.assertTrue(detail_config['self_attn.q_proj']['inputs_cfg']['enable_quant'])
        self.assertEqual(detail_config['layers.0']['weights_cfg']['quant_type'], 'int8')


if __name__ == '__main__':
    unittest.main()
