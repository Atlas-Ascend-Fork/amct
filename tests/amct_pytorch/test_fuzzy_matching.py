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
import copy
import unittest
from unittest.mock import Mock
from unittest.mock import MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fnmatch
import torch
import torch.nn as nn
from mock_torch_npu import mock_npu
from utils import TestModel
from amct_pytorch.config.utils import match_fuzzy_pattern
from amct_pytorch import quantize, convert


class TestCast(unittest.TestCase):
    '''
    ST FOR CAST ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        print('TestCast START!')

    @classmethod
    def tearDownClass(cls):
        print('TestCast END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu
 
    def tearDown(self):
        del sys.modules['torch_npu']


class TestFuzzyPatternMatching(TestCast):
    """Test fuzzy pattern matching functionality"""
    
    def test_exact_match(self):
        """Test exact matching"""
        self.assertTrue(match_fuzzy_pattern('layer1', 'layer1'))
        self.assertFalse(match_fuzzy_pattern('layer1', 'layer2'))
    
    def test_wildcard_prefix(self):
        """Test prefix wildcard"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', '*down_proj'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', '*down_proj'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.up_proj', '*down_proj'))
    
    def test_wildcard_suffix(self):
        """Test suffix wildcard"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', 'model.layers.*'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', 'model.layers.*'))
    
    def test_wildcard_middle(self):
        """Test middle wildcard"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', 'model.*.down_proj'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', 'model.*.down_proj'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.up_proj', 'model.*.down_proj'))
    
    def test_wildcard_with_weights_suffix(self):
        """Test wildcard with .weights suffix"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', '*down_proj.weights'))
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj.weights', '*down_proj.weights'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.down_proj.inputs', '*down_proj.weights'))
    
    def test_wildcard_with_inputs_suffix(self):
        """Test wildcard with .inputs suffix"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.q_proj', '*q_proj.inputs'))
        self.assertTrue(match_fuzzy_pattern('model.layers.0.q_proj.inputs', '*q_proj.inputs'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.q_proj.weights', '*q_proj.inputs'))
    
    def test_complex_pattern(self):
        """Test complex pattern"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.self_attn.q_proj', '*self_attn.q_proj.inputs'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.self_attn.k_proj', '*self_attn.q_proj.inputs'))


class TestFuzzyPatternExceptions(TestCast):
    """Test wildcard pattern exception scenarios"""
    
    def test_empty_pattern(self):
        """WC-001: Test empty pattern"""
        result = (match_fuzzy_pattern('layer1', ''))
        self.assertFalse(result)
    
    def test_wildcard_only(self):
        """WC-002: Test wildcard only"""
        result = match_fuzzy_pattern('any.layer.name', '*')
        self.assertTrue(result)
    
    def test_multiple_wildcards(self):
        """WC-003: Test multiple consecutive wildcards"""
        result = match_fuzzy_pattern('layer', '**')
        self.assertTrue(result)
    
    def test_wildcard_at_start(self):
        """WC-004: Test wildcard at start"""
        result = match_fuzzy_pattern('model.layer.layer', '*layer')
        self.assertTrue(result)
        result = match_fuzzy_pattern('model.other', '*layer')
        self.assertFalse(result)
    
    def test_wildcard_in_middle(self):
        """WC-005: Test wildcard in middle"""
        result = match_fuzzy_pattern('model.layer.name', 'model*name')
        self.assertTrue(result)
        result = match_fuzzy_pattern('model.other.name', 'model*name')
        self.assertTrue(result)
        result = match_fuzzy_pattern('other.model.name', 'model*name')
        self.assertFalse(result)
    
    def test_wildcard_at_end(self):
        """WC-006: Test wildcard at end"""
        result = match_fuzzy_pattern('model.layer.name', 'model*')
        self.assertTrue(result)
        result = match_fuzzy_pattern('other.layer.name', 'model*')
        self.assertFalse(result)
    
    def test_multiple_wildcards_combination(self):
        """WC-007: Test multiple wildcards combination"""
        result = match_fuzzy_pattern('model.layer.name', '*layer*')
        self.assertTrue(result)
        result = match_fuzzy_pattern('model.other.name', '*layer*')
        self.assertFalse(result)
    
    def test_special_characters_with_wildcard(self):
        """WC-008: Test special characters with wildcard"""
        result = match_fuzzy_pattern('model@layer@name', '*layer@name')
        self.assertTrue(result)
    
    def test_suffix_only(self):
        """SF-001: Test suffix only"""
        result = match_fuzzy_pattern('layer', '.weights')
        self.assertFalse(result)
    
    def test_wildcard_with_suffix(self):
        """SF-002: Test wildcard with suffix"""
        result = match_fuzzy_pattern('model.layer1', '*.weights')
        self.assertTrue(result)
        result = match_fuzzy_pattern('model.layer1.weights', '*.weights')
        self.assertTrue(result)
    
    def test_multiple_suffixes(self):
        """SF-003: Test multiple suffixes"""
        result = match_fuzzy_pattern('model.layer', '*layer.weights.weights')
        self.assertFalse(result)
    
    def test_suffix_case_sensitive(self):
        """SF-004: Test suffix case sensitivity"""
        result = match_fuzzy_pattern('model.layer', '*.Weights')
        self.assertFalse(result)
    
    def test_invalid_suffix(self):
        """SF-005: Test invalid suffix"""
        result = match_fuzzy_pattern('model.layer', '*.invalid')
        self.assertFalse(result)


class TestFuzzyConfigBoundaryConditions(TestCast):
    """Test boundary conditions"""
    
    def test_empty_layer_name(self):
        """LN-001: Test empty layer name"""
        result = match_fuzzy_pattern('', '*layer')
        self.assertFalse(result)
    
    def test_long_layer_name(self):
        """LN-002: Test very long layer name"""
        long_name = 'a.' * 100 + 'layer'
        result = match_fuzzy_pattern(long_name, '*layer')
        self.assertTrue(result)
    
    def test_layer_name_with_numbers(self):
        """LN-003: Test layer name with numbers"""
        result = match_fuzzy_pattern('layer0', 'layer*')
        self.assertTrue(result)
        result = match_fuzzy_pattern('layer1', 'layer*')
        self.assertTrue(result)
    
    def test_layer_name_with_underscore(self):
        """LN-004: Test layer name with underscore"""
        result = match_fuzzy_pattern('layer_name', '*name')
        self.assertTrue(result)
    
    def test_layer_name_with_dots(self):
        """LN-005: Test layer name with dots"""
        result = match_fuzzy_pattern('model.layer.name', '*name')
        self.assertTrue(result)


class TestFuzzyConfigSpecialCharacters(TestCast):
    """Test special character handling"""
    
    def test_layer_name_with_spaces(self):
        """SC-001: Test space characters"""
        result = match_fuzzy_pattern('layer name', '*name')
        self.assertTrue(result)
    
    def test_layer_name_with_chinese(self):
        """SC-002: Test Chinese characters"""
        result = match_fuzzy_pattern('层名称', '*名称')
        self.assertTrue(result)
    
    def test_layer_name_with_special_symbols(self):
        """SC-003: Test special symbols"""
        result = match_fuzzy_pattern('layer@#$', '*@#$')
        self.assertTrue(result)
    
    def test_layer_name_with_unicode(self):
        """SC-004: Test Unicode characters"""
        result = match_fuzzy_pattern('layerαβγ', '*αβγ')
        self.assertTrue(result)
    
    def test_layer_name_with_escape(self):
        """SC-005: Test escape characters"""
        result = match_fuzzy_pattern('layer\nname', '*name')
        self.assertTrue(result)


class TestFuzzyConfigPerformance(TestCast):
    """Test performance"""
    
    def test_fuzzy_pattern_performance(self):
        """Test fuzzy matching performance"""
        import time
        
        pattern = '*layer.weights'
        layer_names = [f'model.layers.{i}.down_proj' for i in range(1000)]
        
        start_time = time.time()
        for layer_name in layer_names:
            match_fuzzy_pattern(layer_name, pattern)
        elapsed_time = time.time() - start_time
        
        self.assertLess(elapsed_time, 0.1)
    
    def test_complex_pattern_performance(self):
        """Test complex pattern performance"""
        import time
        
        pattern = '*self_attn.*_proj.weights'
        layer_names = [
            f'model.layers.{i}.self_attnn.q_proj' for i in range(100)
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


class TestWeightsCfgFieldAdditional(TestCast):
    """Test WeightsCfgField class new code"""
    
    def test_weights_cfg_with_none_quant_type(self):
        """Test quant_type is None"""
        from amct_pytorch.config.fields import WeightsCfgField
        
        config = {}
        field = WeightsCfgField(config)
        self.assertIsNone(field.get_value())
    
    def test_weights_cfg_with_group_size(self):
        """Test config with group_size"""
        from amct_pytorch.config.fields import WeightsCfgField
        
        config = {
            'type': 'int4',
            'symmetric': True,
            'strategy': 'group',
            'group_size': 32
        }
        field = WeightsCfgField(config)
        value = field.get_value()
        self.assertEqual(value['quant_type'], 'int4')
        self.assertEqual(value['group_size'], 32)
    
    def test_weights_cfg_without_group_size(self):
        """Test config without group_size"""
        from amct_pytorch.config.fields import WeightsCfgField
        
        config = {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'channel'
        }
        field = WeightsCfgField(config)
        value = field.get_value()
        self.assertEqual(value['quant_type'], 'int8')
        self.assertIsNone(value.get('group_size'))


class TestInputsCfgFieldAdditional(TestCast):
    """Test InputsCfgField class new code"""
    
    def test_inputs_cfg_with_none_quant_type(self):
        """Test quant_type is None"""
        from amct_pytorch.config.fields import InputsCfgField
        
        config = {}
        field = InputsCfgField(config)
        self.assertIsNone(field.get_value())
    
    def test_inputs_cfg_enable_quant_false(self):
        """Test enable_quant is False"""
        from amct_pytorch.config.fields import InputsCfgField
        
        config = {'enable_quant': False}
        field = InputsCfgField(config)
        value = field.get_value()
        self.assertFalse(value['enable_quant'])
    
    def test_inputs_cfg_enable_quant_true(self):
        """Test enable_quant is True"""
        from amct_pytorch.config.fields import InputsCfgField
        
        config = {
            'enable': True,
            'type': 'int8',
            'symmetric': True,
            'strategy': 'tensor'
        }
        field = InputsCfgField(config)
        value = field.get_value()
        self.assertEqual(value['quant_type'], 'int8')
        self.assertEqual(value['strategy'], 'tensor')
    
    def test_get_fuzzy_config_inputs(self):
        """Test getting inputs fuzzy config"""
        config = {
            '*q_proj.inputs': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor'
            },
            'inputs': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor'
            }
        }
        torch.Tensor.npu = mock_npu
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        try:
            quantize(model, config)
        except ValueError as e:
            assert 'Configuration must include at least one weights configuration' in str(e)
    
    def test_validate_weights_config_with_weights(self):
        """Test validation with weights config"""
        
        config = {
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                }
            },
            'algorithm': {'minmax'},
        }
        torch.Tensor.npu = mock_npu
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, config)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear2.scale_w)
    
    def test_validate_weights_config_with_fuzzy_weights(self):
        """Test validation with fuzzy weights config"""
        config = {
            'quant_cfg': {
                '*linear1.weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor'
                }
            },
            'algorithm': {'minmax'},
        }
        torch.Tensor.npu = mock_npu
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, config)
        self.assertIsNotNone(model.linear1.scale_w)
    
    def test_validate_weights_config_without_weights(self):
        """Test validation without weights config"""
        config = {
            'quant_cfg': {
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor'
                }
            },
            'algorithm': {'minmax'},
        }
        torch.Tensor.npu = mock_npu
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        try:
            quantize(model, config)
        except ValueError as e:
            assert 'weights configuration' in str(e)


class TestAlgorithmFieldAdditional(TestCast):
    """Test AlgorithmField class new code"""
    
    def test_algorithm_field_get_value(self):
        """Test AlgorithmField.get_value() method"""
        from amct_pytorch.config.fields import AlgorithmField
        
        registed_alg = Mock()
        registed_alg.algo = {'minmax': {'Linear': None}}
        
        config = {'minmax': {}}
        field = AlgorithmField(config, registed_alg)
        
        value = field.get_value()
        self.assertIsNotNone(value)
        self.assertEqual(value['algorithm'], {'minmax': {}})


class TestParserFunctionsAdditional(TestCast):
    """Test parser module new helper functions"""
    
    def test_check_fuzzy_config_warnings_no_fuzzy(self):
        """Test warning check without fuzzy config"""
        from amct_pytorch.config.parser import _check_fuzzy_config_warnings
        from amct_pytorch.config.fields import QuantConfig
        
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {'enable_quant': False}
            },
            'algorithm': {'minmax'}
        }
        quant_config = QuantConfig(config, Mock())
        all_layer_names = ['layer1', 'layer2']
        
        _check_fuzzy_config_warnings(all_layer_names, quant_config)
    
    def test_check_fuzzy_config_warnings_with_fuzzy(self):
        """Test warning check with fuzzy config"""
        from amct_pytorch.config.parser import _check_fuzzy_config_warnings
        from amct_pytorch.config.fields import QuantConfig
        
        config = {
            'batch_num': 1,
            'quant_cfg': {
                '*down_proj.weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor'
                },
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {'enable_quant': False}
            },
            'algorithm': {'minmax'},
            'skip_layers': ['model.layers.0.down_proj']
        }
        quant_config = QuantConfig(config, Mock())
        all_layer_names = ['model.layers.0.down_proj', 'model.layers.1.down_proj']
        
        _check_fuzzy_config_warnings(all_layer_names, quant_config)
    
    def test_check_fuzzy_config_warnings_no_match(self):
        """Test warning when fuzzy config matches no layer"""
        from amct_pytorch.config.parser import _check_fuzzy_config_warnings
        from amct_pytorch.config.fields import QuantConfig
        
        config = {
            'batch_num': 1,
            'quant_cfg': {
                '*nonexistent.weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor'
                },
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {'enable_quant': False}
            },
            'algorithm': {'minmax'}
        }
        quant_config = QuantConfig(config, Mock())
        all_layer_names = ['layer1', 'layer2']
        
        _check_fuzzy_config_warnings(all_layer_names, quant_config)
    
    def test_build_layer_types_and_quant_type(self):
        """Test building layer types and quant type"""
        from amct_pytorch.config.parser import _build_layer_types_and_quant_type
        from amct_pytorch.config.fields import QuantConfig
        
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor'
                }
            },
            'algorithm': {'minmax'}
        }
        quant_config = QuantConfig(config, Mock())
        registed_alg = Mock()
        registed_alg.algo = {'minmax': {'Linear': None}}
        
        layer_types, quant_type_comb = _build_layer_types_and_quant_type(quant_config, registed_alg)
        
        self.assertIn('Linear', layer_types)
        self.assertEqual(quant_type_comb, 'int8 int8')
    
    def test_is_layer_supported_conv2d_padding_mode(self):
        """Test Conv2d padding_mode check"""
        from amct_pytorch.config.parser import _is_layer_supported
        from amct_pytorch.config.fields import QuantConfig
        
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {'enable_quant': False}
            },
            'algorithm': {'minmax'}
        }
        quant_config = QuantConfig(config, Mock())
        layer_types = {'Conv2d': 'minmax'}
        quant_type_comb = 'int8 int8'
        
        mod = nn.Conv2d(3, 64, 3, padding_mode='reflect')
        is_supported = _is_layer_supported(mod, 'conv1', layer_types, quant_type_comb, quant_config)
        self.assertFalse(is_supported)
    
    def test_get_layer_quant_config_with_fuzzy_weights(self):
        """Test getting layer config with fuzzy weights"""
        from amct_pytorch.config.fields import QuantConfig
        
        config = {
            'batch_num': 1,
            'quant_cfg': {
                '*down_proj.weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor'
                },
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {'enable_quant': False}
            },
            'algorithm': {'minmax'}
        }
        quant_config = QuantConfig(config, Mock())
        
        layer_config = quant_config.get_layer_config('model.layers.0.down_proj')
        self.assertIsNotNone(layer_config)
        self.assertEqual(layer_config['weights_cfg']['strategy'], 'tensor')
    
    def test_get_layer_quant_config_with_fuzzy_inputs(self):
        """Test getting layer config with fuzzy inputs"""
        from amct_pytorch.config.fields import QuantConfig
        
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                '*q_proj.inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor'
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor'
                }
            },
            'algorithm': {'minmax'}
        }
        quant_config = QuantConfig(config, Mock())
        
        layer_config = quant_config.get_layer_config('model.layers.0.self_attn.q_proj')
        self.assertIsNotNone(layer_config)
        self.assertEqual(layer_config['inputs_cfg']['strategy'], 'tensor')
    
    def test_get_layer_quant_config_without_fuzzy_match(self):
        """Test layer config without fuzzy match"""
        from amct_pytorch.config.fields import QuantConfig
        
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {'enable_quant': False}
            },
            'algorithm': {'minmax'}
        }
        quant_config = QuantConfig(config, Mock())
        
        layer_config = quant_config.get_layer_config('layer1')
        self.assertIsNotNone(layer_config)
        self.assertEqual(layer_config['weights_cfg']['strategy'], 'channel')
    
    def test_get_layer_quant_config_no_weights_config(self):
        """Test layer config without weights config"""
        from amct_pytorch.config.fields import QuantConfig
        
        config = {
            'batch_num': 1,
            'quant_cfg': {
                '*down_proj.weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor'
                },
                'algorithm': {'minmax'}
            }
        }
        quant_config = QuantConfig(config, Mock())
        
        layer_config = quant_config.get_layer_config('model.layers.0.up_proj')
        self.assertIsNone(layer_config)


if __name__ == '__main__':
    unittest.main()
