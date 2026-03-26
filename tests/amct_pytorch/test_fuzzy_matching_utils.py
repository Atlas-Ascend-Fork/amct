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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from amct_pytorch.config.utils import is_fuzzy_pattern, get_fuzzy_suffix, match_fuzzy_pattern, find_matching_pattern


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


if __name__ == '__main__':
    unittest.main()
