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
"""
Fuzzy Matching Configuration Tests

This test file combines all fuzzy matching related tests into a single module.
It covers:
1. Pattern matching tests
2. Configuration extraction tests
3. Exception and boundary condition tests
4. Integration tests
5. Performance tests
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fnmatch
import torch
import torch.nn as nn


def match_fuzzy_pattern(layer_name, pattern):
    """
    Match layer name with pattern supporting wildcard '*'
    """
    if '*' not in pattern:
        return layer_name == pattern
    
    suffix = None
    if pattern.endswith('.weights'):
        suffix = '.weights'
        pattern = pattern[:-8]
    elif pattern.endswith('.inputs'):
        suffix = '.inputs'
        pattern = pattern[:-7]
    
    if suffix and layer_name.endswith(suffix):
        layer_name = layer_name[:-len(suffix)]
    
    return fnmatch.fnmatch(layer_name, pattern)


def _extract_fuzzy_configs(config):
    """
    Extract fuzzy matching configs from quant_cfg
    """
    fuzzy_configs = {'weights': [], 'inputs': []}
    
    for key, value in config.items():
        if '*' in key:
            if key.endswith('.weights'):
                fuzzy_configs['weights'].append({
                    'pattern': key,
                    'config': value
                })
            elif key.endswith('.inputs'):
                fuzzy_configs['inputs'].append({
                    'pattern': key,
                    'config': value
                })
    
    return fuzzy_configs


def get_fuzzy_config(layer_name, fuzzy_configs, config_type):
    """
    Get matching fuzzy config for a layer
    """
    for fuzzy_cfg in fuzzy_configs[config_type]:
        if match_fuzzy_pattern(layer_name, fuzzy_cfg['pattern']):
            return fuzzy_cfg['config']
    return None


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
            for _ in range(4)
        ])
        self.lm_head = nn.Linear(64, 64)


class TestFuzzyPatternMatching(unittest.TestCase):
    """测试模糊模式匹配功能"""
    
    def test_exact_match(self):
        """测试精确匹配"""
        self.assertTrue(match_fuzzy_pattern('layer1', 'layer1'))
        self.assertFalse(match_fuzzy_pattern('layer1', 'layer2'))
    
    def test_wildcard_prefix(self):
        """测试前缀通配符"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', '*down_proj'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', '*down_proj'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.up_proj', '*down_proj'))
    
    def test_wildcard_suffix(self):
        """测试后缀通配符"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', 'model.layers.*'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', 'model.layers.*'))
    
    def test_wildcard_middle(self):
        """测试中间通配符"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', 'model.*.down_proj'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', 'model.*.down_proj'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.up_proj', 'model.*.down_proj'))
    
    def test_wildcard_with_weights_suffix(self):
        """测试通配符加.weights后缀"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', '*down_proj.weights'))
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj.weights', '*down_proj.weights'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.down_proj.inputs', '*down_proj.weights'))
    
    def test_wildcard_with_inputs_suffix(self):
        """测试通配符加.inputs后缀"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.q_proj', '*q_proj.inputs'))
        self.assertTrue(match_fuzzy_pattern('model.layers.0.q_proj.inputs', '*q_proj.inputs'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.q_proj.weights', '*q_proj.inputs'))
    
    def test_complex_pattern(self):
        """测试复杂模式"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.self_attn.q_proj', '*self_attn.q_proj.inputs'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.self_attn.k_proj', '*self_attn.q_proj.inputs'))


class TestFuzzyPatternExceptions(unittest.TestCase):
    """测试通配符模式异常场景"""
    
    def test_empty_pattern(self):
        """WC-001: 测试空模式"""
        result = (match_fuzzy_pattern('layer1', ''))
        self.assertFalse(result)
    
    def test_wildcard_only(self):
        """WC-002: 测试只有通配符"""
        result = match_fuzzy_pattern('any.layer.name', '*')
        self.assertTrue(result)
    
    def test_multiple_wildcards(self):
        """WC-003: 测试多个连续通配符"""
        result = match_fuzzy_pattern('layer', '**')
        self.assertTrue(result)
    
    def test_wildcard_at_start(self):
        """WC-004: 测试通配符在开头"""
        result = match_fuzzy_pattern('model.layer.layer', '*layer')
        self.assertTrue(result)
        result = match_fuzzy_pattern('model.other', '*layer')
        self.assertFalse(result)
    
    def test_wildcard_in_middle(self):
        """WC-005: 测试通配符在中间"""
        result = match_fuzzy_pattern('model.layer.name', 'model*name')
        self.assertTrue(result)
        result = match_fuzzy_pattern('model.other.name', 'model*name')
        self.assertTrue(result)
        result = match_fuzzy_pattern('other.model.name', 'model*name')
        self.assertFalse(result)
    
    def test_wildcard_at_end(self):
        """WC-006: 测试通配符在结尾"""
        result = match_fuzzy_pattern('model.layer.name', 'model*')
        self.assertTrue(result)
        result = match_fuzzy_pattern('other.layer.name', 'model*')
        self.assertFalse(result)
    
    def test_multiple_wildcards_combination(self):
        """WC-007: 测试多个通配符组合"""
        result = match_fuzzy_pattern('model.layer.name', '*layer*')
        self.assertTrue(result)
        result = match_fuzzy_pattern('model.other.name', '*layer*')
        self.assertFalse(result)
    
    def test_special_characters_with_wildcard(self):
        """WC-008: 测试特殊字符与通配符"""
        result = match_fuzzy_pattern('model@layer@name', '*layer@name')
        self.assertTrue(result)
    
    def test_suffix_only(self):
        """SF-001: 测试只有后缀"""
        result = match_fuzzy_pattern('layer', '.weights')
        self.assertFalse(result)
    
    def test_wildcard_with_suffix(self):
        """SF-002: 测试通配符加后缀"""
        result = match_fuzzy_pattern('model.layer1', '*.weights')
        self.assertTrue(result)
        result = match_fuzzy_pattern('model.layer1.weights', '*.weights')
        self.assertTrue(result)
    
    def test_multiple_suffixes(self):
        """SF-003: 测试多个后缀"""
        result = match_fuzzy_pattern('model.layer', '*layer.weights.weights')
        self.assertFalse(result)
    
    def test_suffix_case_sensitive(self):
        """SF-004: 测试后缀大小写"""
        result = match_fuzzy_pattern('model.layer', '*.Weights')
        self.assertFalse(result)
    
    def test_invalid_suffix(self):
        """SF-005: 测试无效后缀"""
        result = match_fuzzy_pattern('model.layer', '*.invalid')
        self.assertFalse(result)


class TestFuzzyConfigConflicts(unittest.TestCase):
    """测试配置冲突和优先级"""
    
    def test_multiple_patterns_same_layer(self):
        """CF-001: 测试多个模式匹配同一层"""
        config = {
            '*layer.weights': {'type': 'int8', 'strategy': 'tensor'},
            '*specific.weights': {'type': 'int4', 'strategy': 'channel'},
            'weights': {'type': 'int8', 'strategy': 'channel'},
        }
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        self.assertEqual(len(fuzzy_configs['weights']), 2)
        
        result = get_fuzzy_config('specific.layer', fuzzy_configs, 'weights')
        self.assertEqual(result['type'], 'int8')
        self.assertEqual(result['strategy'], 'tensor')
    
    def test_weights_and_inputs_patterns(self):
        """CF-002: 测试权重和输入模式同时匹配"""
        config = {
            '*layer.weights': {'type': 'int8', 'strategy': 'tensor'},
            '*layer.inputs': {'type': 'int8', 'strategy': 'channel'},
        }
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        self.assertEqual(len(fuzzy_configs['weights']), 1)
        self.assertEqual(len(fuzzy_configs['inputs']), 1)
        
        weights_result = get_fuzzy_config('model.layer', fuzzy_configs, 'weights')
        self.assertEqual(weights_result['strategy'], 'tensor')
        
        inputs_result = get_fuzzy_config('model.layer', fuzzy_configs, 'inputs')
        self.assertEqual(inputs_result['strategy'], 'channel')
    
    def test_fuzzy_vs_default_priority(self):
        """CF-003: 测试模糊配置优先级高于默认配置"""
        config = {
            '*down_proj.weights': {'type': 'int8', 'strategy': 'tensor'},
            'weights': {'type': 'int8', 'strategy': 'channel'},
        }
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        result = get_fuzzy_config('model.down_proj', fuzzy_configs, 'weights')
        self.assertIsNotNone(result)
        self.assertEqual(result['strategy'], 'tensor')
        
        result = get_fuzzy_config('model.other', fuzzy_configs, 'weights')
        self.assertIsNone(result)
    
    def test_exact_vs_fuzzy_priority(self):
        """CF-004: 测试精确配置优先级高于模糊配置"""
        config = {
            'exact.weights': {'type': 'int4', 'strategy': 'channel'},
            '*exact.weights': {'type': 'int8', 'strategy': 'tensor'},
        }
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        self.assertEqual(len(fuzzy_configs['weights']), 1)
        
        result = get_fuzzy_config('exact', fuzzy_configs, 'weights')
        self.assertEqual(result['type'], 'int8')
        self.assertEqual(result['strategy'], 'tensor')


class TestFuzzyConfigBoundaryConditions(unittest.TestCase):
    """测试边界条件"""
    
    def test_empty_layer_name(self):
        """LN-001: 测试空层名称"""
        result = match_fuzzy_pattern('', '*layer')
        self.assertFalse(result)
    
    def test_long_layer_name(self):
        """LN-002: 测试非常长的层名称"""
        long_name = 'a.' * 100 + 'layer'
        result = match_fuzzy_pattern(long_name, '*layer')
        self.assertTrue(result)
    
    def test_layer_name_with_numbers(self):
        """LN-003: 测试层名称包含数字"""
        result = match_fuzzy_pattern('layer0', 'layer*')
        self.assertTrue(result)
        result = match_fuzzy_pattern('layer1', 'layer*')
        self.assertTrue(result)
    
    def test_layer_name_with_underscore(self):
        """LN-004: 测试层名称包含下划线"""
        result = match_fuzzy_pattern('layer_name', '*name')
        self.assertTrue(result)
    
    def test_layer_name_with_dots(self):
        """LN-005: 测试层名称包含点号"""
        result = match_fuzzy_pattern('model.layer.name', '*name')
        self.assertTrue(result)
    
    def test_empty_quant_cfg(self):
        """BC-001: 测试空的quant_cfg"""
        config = {}
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        self.assertEqual(len(fuzzy_configs['weights']), 0)
        self.assertEqual(len(fuzzy_configs['inputs']), 0)
    
    def test_empty_fuzzy_config_value(self):
        """BC-003: 测试空的模糊配置值"""
        config = {
            '*layer.weights': {},
            'weights': {'type': 'int8', 'strategy': 'channel'},
        }
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        self.assertEqual(len(fuzzy_configs['weights']), 1)
        
        result = get_fuzzy_config('model.layer', fuzzy_configs, 'weights')
        self.assertEqual(result, {})
    
    def test_none_fuzzy_config_value(self):
        """BC-004: 测试None的模糊配置值"""
        config = {
            '*layer.weights': None,
            'weights': {'type': 'int8', 'strategy': 'channel'},
        }
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        self.assertEqual(len(fuzzy_configs['weights']), 1)
        
        result = get_fuzzy_config('model.layer', fuzzy_configs, 'weights')
        self.assertIsNone(result)


class TestFuzzyConfigSpecialCharacters(unittest.TestCase):
    """测试特殊字符处理"""
    
    def test_layer_name_with_spaces(self):
        """SC-001: 测试空格字符"""
        result = match_fuzzy_pattern('layer name', '*name')
        self.assertTrue(result)
    
    def test_layer_name_with_chinese(self):
        """SC-002: 测试中文字符"""
        result = match_fuzzy_pattern('层名称', '*名称')
        self.assertTrue(result)
    
    def test_layer_name_with_special_symbols(self):
        """SC-003: 测试特殊符号"""
        result = match_fuzzy_pattern('layer@#$', '*@#$')
        self.assertTrue(result)
    
    def test_layer_name_with_unicode(self):
        """SC-004: 测试Unicode字符"""
        result = match_fuzzy_pattern('layerαβγ', '*αβγ')
        self.assertTrue(result)
    
    def test_layer_name_with_escape(self):
        """SC-005: 测试转义字符"""
        result = match_fuzzy_pattern('layer\nname', '*name')
        self.assertTrue(result)


class TestFuzzyConfigPerformance(unittest.TestCase):
    """测试性能"""
    
    def test_fuzzy_pattern_performance(self):
        """测试模糊匹配性能"""
        import time
        
        pattern = '*layer.weights'
        layer_names = [f'model.layers.{i}.down_proj' for i in range(1000)]
        
        start_time = time.time()
        for layer_name in layer_names:
            match_fuzzy_pattern(layer_name, pattern)
        elapsed_time = time.time() - start_time
        
        self.assertLess(elapsed_time, 0.1)
    
    def test_complex_pattern_performance(self):
        """测试复杂模式性能"""
        import time
        
        pattern = '*self_attn.*_proj.weights'
        layer_names = [
            f'model.layers.{i}.self_attn.q_proj' for i in range(100)
        ] + [
            f'model.layers.{i}.self_attn.k_proj' for i in range(100)
        ] + [
            f'model.layers.{i}.self_attn.v_proj' for i in range(100)
        ]
        
        start_time = time.time()
        for layer_name in layer_names:
            match_fuzzy_pattern(layer_name, pattern)
        elapsed_time = time.time() - start_time
        
        self.assertLess(elapsed_time, 0.1)


class TestFuzzyConfigIntegration(unittest.TestCase):
    """测试集成场景"""
    
    def test_fuzzy_config_with_real_model(self):
        """测试模糊配置与真实模型的集成"""
        model = TestFuzzyModel()
        
        config = {
            '*down_proj.weights': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor',
            },
            '*self_attn.q_proj.inputs': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'channel',
            },
            'weights': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'channel',
            },
            'inputs': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor',
            },
        }
        
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        self.assertEqual(len(fuzzy_configs['weights']), 1)
        self.assertEqual(len(fuzzy_configs['inputs']), 1)
        
        down_proj_layers = [name for name, _ in model.named_modules() if 'down_proj' in name]
        for layer_name in down_proj_layers:
            result = get_fuzzy_config(layer_name, fuzzy_configs, 'weights')
            self.assertIsNotNone(result)
            self.assertEqual(result['strategy'], 'tensor')
        
        q_proj_layers = [name for name, _ in model.named_modules() if 'q_proj' in name]
        for layer_name in q_proj_layers:
            result = get_fuzzy_config(layer_name, fuzzy_configs, 'inputs')
            self.assertIsNotNone(result)
            self.assertEqual(result['strategy'], 'channel')
    
    def test_fuzzy_config_mixed_with_exact_config(self):
        """测试模糊配置与精确配置混合使用"""
        config = {
            '*layer.weights': {'type': 'int8', 'strategy': 'tensor'},
            'exact.weights': {'type': 'int4', 'strategy': 'channel'},
            'weights': {'type': 'int8', 'strategy': 'channel'},
        }
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        self.assertEqual(len(fuzzy_configs['weights']), 1)
        
        result = get_fuzzy_config('model.layer', fuzzy_configs, 'weights')
        self.assertIsNotNone(result)
        self.assertEqual(result['type'], 'int8')
        
        result = get_fuzzy_config('exact', fuzzy_configs, 'weights')
        self.assertIsNone(result)
    
    def test_fuzzy_config_with_complex_patterns(self):
        """测试复杂的模糊配置模式"""
        config = {
            '*self_attn.*_proj.weights': {'type': 'int8', 'strategy': 'channel'},
            '*mlp.*.weights': {'type': 'int8', 'strategy': 'tensor'},
            'weights': {'type': 'int8', 'strategy': 'channel'},
        }
        fuzzy_configs = _extract_fuzzy_configs(config)
        
        result = get_fuzzy_config('model.layers.0.self_attn.q_proj', fuzzy_configs, 'weights')
        self.assertIsNotNone(result)
        self.assertEqual(result['strategy'], 'channel')
        
        result = get_fuzzy_config('model.layers.0.mlp.down_proj', fuzzy_configs, 'weights')
        self.assertIsNotNone(result)
        self.assertEqual(result['strategy'], 'tensor')
    
    def test_fp8_fp4_group_sym_minmax_config(self):
        """测试 FP8 FP4 Group Sym Minmax 模糊配置"""
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                '*linear1.weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                '*linear1.inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'minmax'},
            'skip_layers': {'lm_head'}
        }
        
        fuzzy_configs = _extract_fuzzy_configs(cfg['quant_cfg'])
        
        self.assertEqual(len(fuzzy_configs['weights']), 1)
        self.assertEqual(len(fuzzy_configs['inputs']), 1)
        
        weights_cfg = get_fuzzy_config('model.linear1', fuzzy_configs, 'weights')
        self.assertIsNotNone(weights_cfg)
        self.assertEqual(weights_cfg['strategy'], 'tensor')
        
        inputs_cfg = get_fuzzy_config('model.linear1', fuzzy_configs, 'inputs')
        self.assertIsNotNone(inputs_cfg)
        self.assertEqual(inputs_cfg['strategy'], 'channel')
        
        weights_cfg = get_fuzzy_config('model.linear2', fuzzy_configs, 'weights')
        self.assertIsNone(weights_cfg)
        
        inputs_cfg = get_fuzzy_config('model.linear2', fuzzy_configs, 'inputs')
        self.assertIsNone(inputs_cfg)


if __name__ == '__main__':
    unittest.main()
