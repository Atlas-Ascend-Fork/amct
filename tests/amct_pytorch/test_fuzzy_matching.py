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
import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fnmatch

import torch
import torch.nn as nn
from mock_torch_npu import mock_npu
from utils import TestModel

from amct_pytorch import convert, quantize
from amct_pytorch.common.config.utils import match_fuzzy_pattern

ALGORITHM_MINMAX = 'minmax'
KEY_BATCH_NUM = 'batch_num'
KEY_ENABLE_QUANT = 'enable_quant'
KEY_STRATEGY = 'strategy'
KEY_SYMMETRIC = 'symmetric'
KEY_TYPE = 'type'
QUANT_TYPE_INT8 = 'int8'
PATTERN_MODEL_STAR_NAME = 'model*name'
PATTERN_STAR_DOWN_PROJ = '*down_proj'
LAYER1_NAME = 'layer1'
PATTERN_MODEL_STAR_DOT_DOWN_PROJ = 'model.*.down_proj'
PATTERN_STAR_Q_PROJ_INPUTS = '*q_proj.inputs'
PATTERN_STAR_DOWN_PROJ_WEIGHTS = '*down_proj.weights'

STRATEGY_CHANNEL = 'channel'
STRATEGY_TENSOR = 'tensor'
GROUP_SIZE = 'group_size'
QUANT_TYPE = 'quant_type'

logger = logging.getLogger(__name__)

LAYER1 = 'layer1'


class TestCast(unittest.TestCase):
    '''
    ST FOR CAST ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        logger.info('TestCast START!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestCast END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu
 
    def tearDown(self):
        del sys.modules['torch_npu']


class TestFuzzyPatternMatching(TestCast):
    """Test fuzzy pattern matching functionality"""
    
    def test_exact_match(self):
        """Test exact matching"""
        self.assertTrue(match_fuzzy_pattern(LAYER1, LAYER1))
        self.assertFalse(match_fuzzy_pattern(LAYER1, 'layer2'))
    
    def test_wildcard_prefix(self):
        """Test prefix wildcard"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', PATTERN_STAR_DOWN_PROJ))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', PATTERN_STAR_DOWN_PROJ))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.up_proj', PATTERN_STAR_DOWN_PROJ))
    
    def test_wildcard_suffix(self):
        """Test suffix wildcard"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', 'model.layers.*'))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', 'model.layers.*'))
    
    def test_wildcard_middle(self):
        """Test middle wildcard"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', PATTERN_MODEL_STAR_DOT_DOWN_PROJ))
        self.assertTrue(match_fuzzy_pattern('model.layers.1.down_proj', PATTERN_MODEL_STAR_DOT_DOWN_PROJ))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.up_proj', PATTERN_MODEL_STAR_DOT_DOWN_PROJ))
    
    def test_wildcard_with_weights_suffix(self):
        """Test wildcard with .weights suffix"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj', PATTERN_STAR_DOWN_PROJ_WEIGHTS))
        self.assertTrue(match_fuzzy_pattern('model.layers.0.down_proj.weights', PATTERN_STAR_DOWN_PROJ_WEIGHTS))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.down_proj.inputs', PATTERN_STAR_DOWN_PROJ_WEIGHTS))
    
    def test_wildcard_with_inputs_suffix(self):
        """Test wildcard with .inputs suffix"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.q_proj', PATTERN_STAR_Q_PROJ_INPUTS))
        self.assertTrue(match_fuzzy_pattern('model.layers.0.q_proj.inputs', PATTERN_STAR_Q_PROJ_INPUTS))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.q_proj.weights', PATTERN_STAR_Q_PROJ_INPUTS))
    
    def test_complex_pattern(self):
        """Test complex pattern"""
        self.assertTrue(match_fuzzy_pattern('model.layers.0.self_attn.q_proj', '*self_attn.q_proj.inputs'))
        self.assertFalse(match_fuzzy_pattern('model.layers.0.self_attn.k_proj', '*self_attn.q_proj.inputs'))


class TestFuzzyPatternExceptions(TestCast):
    """Test wildcard pattern exception scenarios"""
    
    def test_empty_pattern(self):
        """WC-001: Test empty pattern"""
        result = (match_fuzzy_pattern(LAYER1, ''))
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
        result = match_fuzzy_pattern('model.layer.name', PATTERN_MODEL_STAR_NAME)
        self.assertTrue(result)
        result = match_fuzzy_pattern('model.other.name', PATTERN_MODEL_STAR_NAME)
        self.assertTrue(result)
        result = match_fuzzy_pattern('other.model.name', PATTERN_MODEL_STAR_NAME)
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
        result = match_fuzzy_pattern(LAYER1, 'layer*')
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
        from amct_pytorch.common.config.fields import WeightsCfgField
        
        config = {}
        field = WeightsCfgField(config)
        self.assertIsNone(field.get_value())
    
    def test_weights_cfg_with_group_size(self):
        """Test config with group_size"""
        from amct_pytorch.common.config.fields import WeightsCfgField
        
        config = {
            KEY_TYPE: 'int4',
            KEY_SYMMETRIC: True,
            KEY_STRATEGY: 'group',
            GROUP_SIZE: 32
        }
        field = WeightsCfgField(config)
        value = field.get_value()
        self.assertEqual(value[QUANT_TYPE], 'int4')
        self.assertEqual(value[GROUP_SIZE], 32)
    
    def test_weights_cfg_without_group_size(self):
        """Test config without group_size"""
        from amct_pytorch.common.config.fields import WeightsCfgField
        
        config = {
            KEY_TYPE: QUANT_TYPE_INT8,
            KEY_SYMMETRIC: True,
            KEY_STRATEGY: STRATEGY_CHANNEL
        }
        field = WeightsCfgField(config)
        value = field.get_value()
        self.assertEqual(value[QUANT_TYPE], QUANT_TYPE_INT8)
        self.assertIsNone(value.get(GROUP_SIZE))


class TestInputsCfgFieldAdditional(TestCast):
    """Test InputsCfgField class new code"""
    
    def test_inputs_cfg_with_none_quant_type(self):
        """Test quant_type is None"""
        from amct_pytorch.common.config.fields import InputsCfgField
        
        config = {}
        field = InputsCfgField(config)
        self.assertIsNone(field.get_value())
    
    def test_inputs_cfg_enable_quant_false(self):
        """Test enable_quant is False"""
        from amct_pytorch.common.config.fields import InputsCfgField
        
        config = {KEY_ENABLE_QUANT: False}
        field = InputsCfgField(config)
        value = field.get_value()
        self.assertFalse(value[KEY_ENABLE_QUANT])
    
    def test_inputs_cfg_enable_quant_true(self):
        """Test enable_quant is True"""
        from amct_pytorch.common.config.fields import InputsCfgField
        
        config = {
            'enable': True,
            KEY_TYPE: QUANT_TYPE_INT8,
            KEY_SYMMETRIC: True,
            KEY_STRATEGY: STRATEGY_TENSOR
        }
        field = InputsCfgField(config)
        value = field.get_value()
        self.assertEqual(value[QUANT_TYPE], QUANT_TYPE_INT8)
        self.assertEqual(value[KEY_STRATEGY], STRATEGY_TENSOR)
    
    def test_get_fuzzy_config_inputs(self):
        """Test getting inputs fuzzy config"""
        config = {
            PATTERN_STAR_Q_PROJ_INPUTS: {
                KEY_TYPE: QUANT_TYPE_INT8,
                KEY_SYMMETRIC: True,
                KEY_STRATEGY: STRATEGY_TENSOR
            },
            'inputs': {
                KEY_TYPE: QUANT_TYPE_INT8,
                KEY_SYMMETRIC: True,
                KEY_STRATEGY: STRATEGY_TENSOR
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
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_CHANNEL
                }
            },
            'algorithm': {ALGORITHM_MINMAX},
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
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                }
            },
            'algorithm': {ALGORITHM_MINMAX},
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
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                }
            },
            'algorithm': {ALGORITHM_MINMAX},
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
        from amct_pytorch.common.config.fields import AlgorithmField
        
        registed_alg = Mock()
        registed_alg.algo = {ALGORITHM_MINMAX: {'Linear': None}}
        
        config = {ALGORITHM_MINMAX: {}}
        field = AlgorithmField(config, registed_alg)
        
        value = field.get_value()
        self.assertIsNotNone(value)
        self.assertEqual(value['algorithm'], {ALGORITHM_MINMAX: {}})


class TestParserFunctionsAdditional(TestCast):
    """Test parser module new helper functions"""
    
    def test_check_fuzzy_config_warnings_no_fuzzy(self):
        """Test warning check without fuzzy config"""
        from amct_pytorch.common.config.fields import QuantConfig
        from amct_pytorch.common.config.parser import _check_fuzzy_config_warnings
        
        config = {
            KEY_BATCH_NUM: 1,
            'quant_cfg': {
                'weights': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_CHANNEL
                },
                'inputs': {KEY_ENABLE_QUANT: False}
            },
            'algorithm': {ALGORITHM_MINMAX}
        }
        quant_config = QuantConfig(config, Mock())
        all_layer_names = [LAYER1, 'layer2']
        
        _check_fuzzy_config_warnings(all_layer_names, quant_config)
    
    def test_check_fuzzy_config_warnings_with_fuzzy(self):
        """Test warning check with fuzzy config"""
        from amct_pytorch.common.config.fields import QuantConfig
        from amct_pytorch.common.config.parser import _check_fuzzy_config_warnings
        
        config = {
            KEY_BATCH_NUM: 1,
            'quant_cfg': {
                PATTERN_STAR_DOWN_PROJ_WEIGHTS: {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
                'weights': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_CHANNEL
                },
                'inputs': {KEY_ENABLE_QUANT: False}
            },
            'algorithm': {ALGORITHM_MINMAX},
            'skip_layers': ['model.layers.0.down_proj']
        }
        quant_config = QuantConfig(config, Mock())
        all_layer_names = ['model.layers.0.down_proj', 'model.layers.1.down_proj']
        
        _check_fuzzy_config_warnings(all_layer_names, quant_config)
    
    def test_check_fuzzy_config_warnings_no_match(self):
        """Test warning when fuzzy config matches no layer"""
        from amct_pytorch.common.config.fields import QuantConfig
        from amct_pytorch.common.config.parser import _check_fuzzy_config_warnings
        
        config = {
            KEY_BATCH_NUM: 1,
            'quant_cfg': {
                '*nonexistent.weights': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
                'weights': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_CHANNEL
                },
                'inputs': {KEY_ENABLE_QUANT: False}
            },
            'algorithm': {ALGORITHM_MINMAX}
        }
        quant_config = QuantConfig(config, Mock())
        all_layer_names = [LAYER1, 'layer2']
        
        _check_fuzzy_config_warnings(all_layer_names, quant_config)
    
    def test_build_layer_types_and_quant_type(self):
        """Test building layer types and quant type"""
        from amct_pytorch.common.config.fields import QuantConfig
        from amct_pytorch.common.config.parser import _build_layer_types_and_quant_type
        
        config = {
            KEY_BATCH_NUM: 1,
            'quant_cfg': {
                'weights': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_CHANNEL
                },
                'inputs': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                }
            },
            'algorithm': {ALGORITHM_MINMAX}
        }
        quant_config = QuantConfig(config, Mock())
        registed_alg = Mock()
        registed_alg.algo = {ALGORITHM_MINMAX: {'Linear': None}}
        
        layer_types, quant_type_comb = _build_layer_types_and_quant_type(quant_config, registed_alg)
        
        self.assertIn('Linear', layer_types)
        self.assertEqual(quant_type_comb, 'int8 int8')
    
    def test_is_layer_supported_conv2d_padding_mode(self):
        """Test Conv2d padding_mode check"""
        from amct_pytorch.common.config.fields import QuantConfig
        from amct_pytorch.common.config.parser import _is_layer_supported
        
        config = {
            KEY_BATCH_NUM: 1,
            'quant_cfg': {
                'weights': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_CHANNEL
                },
                'inputs': {KEY_ENABLE_QUANT: False}
            },
            'algorithm': {ALGORITHM_MINMAX}
        }
        quant_config = QuantConfig(config, Mock())
        layer_types = {'Conv2d': ALGORITHM_MINMAX}
        quant_type_comb = 'int8 int8'
        
        mod = nn.Conv2d(3, 64, 3, padding_mode='reflect')
        is_supported = _is_layer_supported(mod, 'conv1', layer_types, quant_type_comb, quant_config)
        self.assertFalse(is_supported)
    
    def test_get_layer_quant_config_with_fuzzy_weights(self):
        """Test getting layer config with fuzzy weights"""
        from amct_pytorch.common.config.fields import QuantConfig
        
        config = {
            KEY_BATCH_NUM: 1,
            'quant_cfg': {
                PATTERN_STAR_DOWN_PROJ_WEIGHTS: {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
                'weights': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_CHANNEL
                },
                'inputs': {KEY_ENABLE_QUANT: False}
            },
            'algorithm': {ALGORITHM_MINMAX}
        }
        quant_config = QuantConfig(config, Mock())
        
        layer_config = quant_config.get_layer_config('model.layers.0.down_proj')
        self.assertIsNotNone(layer_config)
        self.assertEqual(layer_config['weights_cfg'][KEY_STRATEGY], STRATEGY_TENSOR)
    
    def test_get_layer_quant_config_with_fuzzy_inputs(self):
        """Test getting layer config with fuzzy inputs"""
        from amct_pytorch.common.config.fields import QuantConfig
        
        config = {
            KEY_BATCH_NUM: 1,
            'quant_cfg': {
                'weights': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_CHANNEL
                },
                PATTERN_STAR_Q_PROJ_INPUTS: {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
                'inputs': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                }
            },
            'algorithm': {ALGORITHM_MINMAX}
        }
        quant_config = QuantConfig(config, Mock())
        
        layer_config = quant_config.get_layer_config('model.layers.0.self_attn.q_proj')
        self.assertIsNotNone(layer_config)
        self.assertEqual(layer_config['inputs_cfg'][KEY_STRATEGY], STRATEGY_TENSOR)
    
    def test_get_layer_quant_config_without_fuzzy_match(self):
        """Test layer config without fuzzy match"""
        from amct_pytorch.common.config.fields import QuantConfig
        
        config = {
            KEY_BATCH_NUM: 1,
            'quant_cfg': {
                'weights': {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_CHANNEL
                },
                'inputs': {KEY_ENABLE_QUANT: False}
            },
            'algorithm': {ALGORITHM_MINMAX}
        }
        quant_config = QuantConfig(config, Mock())
        
        layer_config = quant_config.get_layer_config(LAYER1)
        self.assertIsNotNone(layer_config)
        self.assertEqual(layer_config['weights_cfg'][KEY_STRATEGY], STRATEGY_CHANNEL)
    
    def test_get_layer_quant_config_no_weights_config(self):
        """Test layer config without weights config"""
        from amct_pytorch.common.config.fields import QuantConfig
        
        config = {
            KEY_BATCH_NUM: 1,
            'quant_cfg': {
                PATTERN_STAR_DOWN_PROJ_WEIGHTS: {
                    KEY_TYPE: QUANT_TYPE_INT8,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
                'algorithm': {ALGORITHM_MINMAX}
            }
        }
        quant_config = QuantConfig(config, Mock())
        
        layer_config = quant_config.get_layer_config('model.layers.0.up_proj')
        self.assertIsNone(layer_config)

if __name__ == '__main__':
    unittest.main()

