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


def match_fuzzy_pattern(layer_name, pattern):
    """
    Match layer name with pattern supporting wildcard '*'
    Params:
        layer_name: str, the full layer name (e.g., 'model.layers.0.self_attn.q_proj')
        pattern: str, the pattern to match (e.g., '*self_attn.q_proj.weights')
    Return: bool, True if pattern matches layer_name
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


class TestFuzzyPattern(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(match_fuzzy_pattern('layer1', 'layer1'))
        self.assertFalse(match_fuzzy_pattern('layer1', 'layer2'))
    
    def test_wildcard_prefix(self):
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', '*down_proj'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', '*down_proj'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.up_proj', '*down_proj'))
    
    def test_wildcard_suffix(self):
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', 'model.layers.*'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', 'model.layers.*'))
        self.assertTrue(match_fuzzy_pattern('model.layers.0', 'model.layers.*'))
        self.assertFalse(match_fuzzy_pattern('model.layers', 'model.layers.*'))
    
    def test_wildcard_middle(self):
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', 'model.*.down_proj'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', 'model.*.down_proj'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.up_proj', 'model.*.down_proj'))
    
    def test_wildcard_with_weights_suffix(self):
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', '*down_proj.weights'))
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj.weights', '*down_proj.weights'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.down_proj.inputs', '*down_proj.weights'))
    
    def test_wildcard_with_inputs_suffix(self):
        self.assertTrue(match_fuzzy_pattern('model.layers.0.q_proj', '*q_proj.inputs'))
        self.assertTrue(match_fuzzy_pattern('model.layers.0.q_proj.inputs', '*q_proj.inputs'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.q_proj.weights', '*q_proj.inputs'))
    
    def test_multiple_wildcards(self):
        self.assertTrue(match_fuzzy_pattern('a.b.c.d', '*.*.c.*'))
        self.assertTrue(match_fuzzy_pattern('x.y.c.z', '*.*.c.*'))
        self.assertFalse(match_fuzzy_pattern('a.b.x.d', '*.*.c.*'))
    
    def test_complex_pattern(self):
        pattern = '*self_attn.q_proj.weights'
        self.assertTrue(match_fuzzy_pattern('model.layers.0.self_attn.q_proj', pattern))
        self.assertTrue(match_fuzzy_pattern('model.layers.0.self_attn.q_proj.weights', pattern))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.self_attn.k_proj', pattern))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.mlp.q_proj', pattern))


if __name__ == '__main__':
    unittest.main()
