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
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class TestFuzzyPatternExceptions(unittest.TestCase):
    """测试通配符模式异常场景"""
    
    def test_empty_pattern(self):
        """WC-001: 测试空模式"""
        result = match_fuzzy_pattern('layer1', '')
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
        result = match_fuzzy_pattern('model.layer', '*layer')
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
        
        self.assertLess(elapsed_time, 0.1)  # 100ms
    
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
        
        self.assertLess(elapsed_time, 0.1)  # 100ms


if __name__ == '__main__':
    unittest.main()
