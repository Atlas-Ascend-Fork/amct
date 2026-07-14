# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import copy
import logging
import unittest

import torch
import torch.nn as nn
from utils import TestModel, TestModelBias, TestModelConv2d

from amct_pytorch import (
    INT4_AWQ_WEIGHT_QUANT_CFG,
    INT4_GPTQ_WEIGHT_QUANT_CFG,
    INT8_MINMAX_WEIGHT_QUANT_CFG,
    INT8_SMOOTHQUANT_CFG,
)
from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.common.config import parse_config

KEY_BATCH_NUM = 'batch_num'
KEY_SYMMETRIC = 'symmetric'
KEY_TYPE = 'type'
LAYER_LINEAR1 = 'linear1'
STRATEGY_TENSOR = 'tensor'
HIFLOAT8 = 'hifloat8'
QUANT_CFG = 'quant_cfg'
KEY_STRATEGY = 'strategy'
WEIGHTS_CFG = 'weights_cfg'
INPUTS_CFG = 'inputs_cfg'
ALGORITHM_KEY = 'algorithm'
WEIGHTS_KEY = 'weights'
ALGO_NAME_A = 'AlgorithmA'

CONV2D1 = 'conv2d1'
FLOAT4_E2M1 = 'float4_e2m1'
LINEAR2 = 'linear2'

GROUP_SIZE = 'group_size'
INPUTS = 'inputs'

FLOAT8_E4M3FN = 'float8_e4m3fn'

SMOOTH_QUANT = 'smoothquant'

SMOOTH_STRENGTH = 'smooth_strength'


class CustomQuant(BaseQuantizeModule):
    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__(ori_module, layer_name, quant_config)

    def forward(self, inputs):
        return inputs


class CustomDeployQuant(nn.Module):
    def __init__(self, ori_module):
        super().__init__()

    def forward(self, inputs):
        return inputs

logger = logging.getLogger(__name__)


class TestConfigParse(unittest.TestCase):
    '''
    ST FOR CONFIG PARSER
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel()
        cls.inputs = torch.randn(64, 64)
        logger.info('TestConfigParse START!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestConfigParse END!')

    def test_int4_awq_cfg(self):
        detail_config = parse_config(self.test_model, INT4_AWQ_WEIGHT_QUANT_CFG, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 2)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(KEY_BATCH_NUM), 1)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get('quant_type'), 'int4')
        self.assertTrue(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_SYMMETRIC))
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_STRATEGY), 'channel')
        self.assertFalse(detail_config.get(LAYER_LINEAR1).get(INPUTS_CFG).get('enable_quant'))
        self.assertEqual(len(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY)), 1)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY).get('awq').get('grids_num'), 20)


    def test_int4_gptq_cfg(self):
        detail_config = parse_config(self.test_model, INT4_GPTQ_WEIGHT_QUANT_CFG, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 2)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(KEY_BATCH_NUM), 1)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get('quant_type'), 'int4')
        self.assertTrue(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_SYMMETRIC))
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_STRATEGY), 'channel')
        self.assertFalse(detail_config.get(LAYER_LINEAR1).get(INPUTS_CFG).get('enable_quant'))
        self.assertEqual(len(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY)), 1)
        self.assertEqual(list(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY))[0], 'gptq')


    def test_int8_minmax_cfg(self):
        model_bfloat16 = copy.deepcopy(self.test_model).to(torch.bfloat16)
        detail_config = parse_config(model_bfloat16, INT8_MINMAX_WEIGHT_QUANT_CFG, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 3)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(KEY_BATCH_NUM), 1)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get('quant_type'), 'int8')
        self.assertTrue(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_SYMMETRIC))
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_STRATEGY), 'channel')
        self.assertFalse(detail_config.get(LAYER_LINEAR1).get(INPUTS_CFG).get('enable_quant'))
        self.assertEqual(len(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY)), 1)
        self.assertEqual(list(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY))[0], 'minmax')

    
    def test_int8_smooth_cfg(self):
        model_bfloat16 = copy.deepcopy(self.test_model).to(torch.bfloat16)
        detail_config = parse_config(model_bfloat16, INT8_SMOOTHQUANT_CFG, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 3)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(KEY_BATCH_NUM), 1)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get('quant_type'), 'int8')
        self.assertTrue(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_SYMMETRIC))
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_STRATEGY), 'channel')
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(INPUTS_CFG).get('quant_type'), 'int8')
        self.assertTrue(detail_config.get(LAYER_LINEAR1).get(INPUTS_CFG).get(KEY_SYMMETRIC))
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(INPUTS_CFG).get(KEY_STRATEGY), STRATEGY_TENSOR)
        self.assertEqual(len(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY)), 1)
        self.assertEqual(
            detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY)
            .get(SMOOTH_QUANT).get(SMOOTH_STRENGTH), 0.5)


    def test_int8_int4_minmax_cfg(self):
        model_fp16 = copy.deepcopy(self.test_model).to(torch.float16)
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
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
        detail_config = parse_config(model_fp16, cfg, AlgorithmRegistry)

        self.assertEqual(len(detail_config.keys()), 2)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('quant_type'), 'int4')
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('strategy'), 'channel')
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('quant_type'), 'int8')
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('strategy'), 'tensor')
        self.assertEqual(list(detail_config.get('linear1').get('algorithm'))[0], 'minmax')

    def test_int8_int4_smooth_cfg(self):
        model_fp16 = copy.deepcopy(self.test_model).to(torch.float16)
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
            'skip_layers': {'lm_head'}
        }
        detail_config = parse_config(model_fp16, cfg, AlgorithmRegistry)

        self.assertEqual(len(detail_config.keys()), 2)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('quant_type'), 'int4')
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('strategy'), 'channel')
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('quant_type'), 'int8')
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('strategy'), 'tensor')
        self.assertEqual(detail_config.get('linear1').get('algorithm').get('smoothquant').get('smooth_strength'), 0.5)

    def test_invalid_batch_num(self):
        invalid_batch_num_cfg = {
            KEY_BATCH_NUM: 1.2,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int4',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
            },
            ALGORITHM_KEY: {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }
        try:
            parse_config(self.test_model, invalid_batch_num_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Batch num only support positive int, but got', str(e))

    def test_invalid_wts_quant_type(self):
        invalid_quant_type_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int6',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
            },
            ALGORITHM_KEY: {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }

        try:
            parse_config(self.test_model, invalid_quant_type_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights quant_dtype only support', str(e))
    
    def test_invalid_wts_symmetric(self):
        invalid_symmetric_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: 'true',
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
            },
            ALGORITHM_KEY: {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }

        try:
            parse_config(self.test_model, invalid_symmetric_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights symmetric only support bool [True, False], but got', str(e))

    def test_invalid_wts_strategy(self):
        invalid_strategy_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'token'
                },
            },
            ALGORITHM_KEY: {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }

        try:
            parse_config(self.test_model, invalid_strategy_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights strategy only support', str(e))

    def test_invalid_inputs_strategy(self):
        invalid_strategy_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
                INPUTS: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
            },
            ALGORITHM_KEY: {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }

        try:
            parse_config(self.test_model, invalid_strategy_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Inputs strategy only support', str(e))

    def test_invalid_inputs_quant_type(self):
        invalid_quant_type_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
                INPUTS: {
                    KEY_TYPE: 'uint4',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
            },
            ALGORITHM_KEY: {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }
        try:
            parse_config(self.test_model, invalid_quant_type_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Inputs quant_dtype only support', str(e))

    def test_unregisted_algo(self):
        invalid_regist_algo_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
            },
            ALGORITHM_KEY: {'cluster'},
        }

        try:
            parse_config(self.test_model, invalid_regist_algo_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Not support algorithm', str(e))

    def test_awq_no_param(self):
        invalid_awq_param_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
            },
            ALGORITHM_KEY: {'awq'},
        }

        try:
            parse_config(self.test_model, invalid_awq_param_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Awq grids_num is necessary param, pls check and set', str(e))

    def test_awq_invalid_grids_num(self):
        invalid_awq_param_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
            },
            ALGORITHM_KEY: {'awq': {'grids_num': -20}},
        }
        try:
            parse_config(self.test_model, invalid_awq_param_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Awq grids_num only support positive int, but got', str(e))

    def test_smooth_invalid_grids_num(self):
        invalid_smooth_param_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0}},
        }
        try:
            parse_config(self.test_model, invalid_smooth_param_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Smoothquant smooth_strength only support float (0, 1), but got', str(e))

    def test_unsupported_quant_type_comb(self):
        unsupported_quant_type_comb_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int4',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
                INPUTS: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
            },
            ALGORITHM_KEY: {'minmax'},
        }
        try:
            parse_config(self.test_model, unsupported_quant_type_comb_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Do not support combination int8 int4 of act and weight quant dtype', str(e))

    def test_unsupported_algo_for_quant_type_comb(self):
        unsupported_algo_quant_type_comb_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
                INPUTS: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
            },
            ALGORITHM_KEY: {'gptq'},
        }
        try:
            parse_config(self.test_model, unsupported_algo_quant_type_comb_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Algorithm gptq do not support act and weight quant dtype int8 int8', str(e))

    def test_skip_layers(self):
        int4_awq_skip_layer_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int4',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel'
                },
            },
            ALGORITHM_KEY: {'awq': {'grids_num': 20}},
            'skip_layers': {'linear'}

        }
        detail_config = parse_config(self.test_model, int4_awq_skip_layer_cfg, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 0)

    def test_group_size(self):
        int4_group_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int4',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
            },
            ALGORITHM_KEY: {'minmax'},

        }
        parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        

    def test_group_size_none_strategy_group(self):
        int4_group_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int4',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group'
                },
            },
            ALGORITHM_KEY: {'minmax'},

        }
        try:
            parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights group_size is necessary, when weights strategy is group', str(e))
    
    def test_group_size_32_strategy_other(self):
        int4_group_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int4',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel',
                    GROUP_SIZE: 32
                },
            },
            ALGORITHM_KEY: {'minmax'},

        }
        try:
            parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights group_size only support strategy group, but got ', str(e))

    def test_group_size_2_strategy_group(self):
        int4_group_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int4',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 2
                },
            },
            ALGORITHM_KEY: {'minmax'},

        }
        try:
            parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('only support group_size larger than 32 and', str(e))

    def test_group_size_invalid(self):
        int4_group_cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int4',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 2.2
                },
            },
            ALGORITHM_KEY: {'minmax'},

        }
        try:
            parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('group_size only support positive int, but got ', str(e))       
    
    def test_int8_int8_group_invalid_cfg(self):
        cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: False,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        try:
            parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('int8 int8 only support symmetric weight quantization', str(e))

    def test_unregist_algo_cfg(self):
        cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: False,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {'AA': {'BB': 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        try:
            parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn('Not support algorithm AA, pls regiter it first', str(e))  

    def test_customize_algo_cfg(self):
        cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: False,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {ALGO_NAME_A: {'BB': 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        AlgorithmRegistry.register(ALGO_NAME_A, 'Linear', CustomQuant, CustomDeployQuant)
        detail_config = parse_config(model, cfg, AlgorithmRegistry)
        self.assertEqual(len(detail_config.keys()), 3)
        self.assertEqual(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY).get(ALGO_NAME_A).get('BB'), 0.8)


    def test_int8_int8_token_asymetric_invalid_cfg(self):
        cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: False,
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
                INPUTS: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: False,
                    KEY_STRATEGY: 'token',
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        try:
            parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
    
            self.assertIn('Inputs strategy token do not support asymmetric quantization', str(e))

    def test_int8_int8_token_asymmetric_invalid_cfg(self):
        cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: False,
                    KEY_STRATEGY: STRATEGY_TENSOR
                },
                INPUTS: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: False,
                    KEY_STRATEGY: 'token',
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        try:
            parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn('Inputs strategy token do not support asymmetric quantization', str(e))

    
    def test_int8_int8_weight_asymmetric_invalid_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'channel'
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        try:
            parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn('int8 int8 only support symmetric weight quantization', str(e))

    def test_int8_int4_weight_asymmetric_invalid_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': False,
                    'strategy': 'channel'
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'minmax'},
        }
        model = self.test_model.to(torch.float16)
        try:
            parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn('int8 int4 only support symmetric weight quantization', str(e))

    def test_int8_int4_group_invalid_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        model = self.test_model.to(torch.float16)
        try:
            parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn('int8 int4 only support weight quant strategy tensor or channel', str(e))

    def test_int8_int4_token_invalid_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'token',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        model = self.test_model.to(torch.float16)
        try:
            parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn('int8 int4 only support activation quant strategy tensor', str(e))

    def test_int8_int4_activation_asymmetric_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'tensor'
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'minmax'},
        }
        detail_config = parse_config(copy.deepcopy(self.test_model).to(torch.float16), cfg, AlgorithmRegistry)
        self.assertEqual(len(detail_config.keys()), 2)
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('symmetric'), False)

    def test_int8_int4_dtype_skip_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        detail_config = parse_config(copy.deepcopy(self.test_model).to(torch.bfloat16), cfg, AlgorithmRegistry)
        self.assertEqual(len(detail_config.keys()), 0)

    def test_float8_float4_smooth_cfg(self):
        cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.5}}
        }
        model_bfloat16 = copy.deepcopy(TestModelBias()).to(torch.bfloat16)
        detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)

        self.assertEqual(len(detail_config.keys()), 1)
        self.assertEqual(detail_config.get(LINEAR2).get(KEY_BATCH_NUM), 1)
        self.assertEqual(detail_config.get(LINEAR2).get(WEIGHTS_CFG).get('quant_type'), FLOAT4_E2M1)
        self.assertTrue(detail_config.get(LINEAR2).get(WEIGHTS_CFG).get(KEY_SYMMETRIC))
        self.assertEqual(detail_config.get(LINEAR2).get(WEIGHTS_CFG).get(KEY_STRATEGY), 'group')
        self.assertEqual(detail_config.get(LINEAR2).get(INPUTS_CFG).get('quant_type'), FLOAT8_E4M3FN)
        self.assertTrue(detail_config.get(LINEAR2).get(INPUTS_CFG).get(KEY_SYMMETRIC))
        self.assertEqual(detail_config.get(LINEAR2).get(INPUTS_CFG).get(KEY_STRATEGY), STRATEGY_TENSOR)
        self.assertEqual(len(detail_config.get(LINEAR2).get(ALGORITHM_KEY)), 1)
        self.assertEqual(detail_config.get(LINEAR2).get(ALGORITHM_KEY).get(SMOOTH_QUANT).get(SMOOTH_STRENGTH), 0.5)

    def test_float8_float4_minmax_cfg(self):
        cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {'minmax'}
        }
        model_bfloat16 = copy.deepcopy(TestModelBias()).to(torch.bfloat16)
        detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)

        self.assertEqual(len(detail_config.keys()), 1)
        self.assertEqual(detail_config.get(LINEAR2).get(KEY_BATCH_NUM), 1)
        self.assertEqual(detail_config.get(LINEAR2).get(WEIGHTS_CFG).get('quant_type'), FLOAT4_E2M1)
        self.assertTrue(detail_config.get(LINEAR2).get(WEIGHTS_CFG).get(KEY_SYMMETRIC))
        self.assertEqual(detail_config.get(LINEAR2).get(WEIGHTS_CFG).get(KEY_STRATEGY), 'group')
        self.assertEqual(detail_config.get(LINEAR2).get(INPUTS_CFG).get('quant_type'), FLOAT8_E4M3FN)
        self.assertTrue(detail_config.get(LINEAR2).get(INPUTS_CFG).get(KEY_SYMMETRIC))
        self.assertEqual(detail_config.get(LINEAR2).get(INPUTS_CFG).get(KEY_STRATEGY), STRATEGY_TENSOR)
        self.assertEqual(len(detail_config.get(LINEAR2).get(ALGORITHM_KEY)), 1)
        self.assertEqual(list(detail_config.get(LINEAR2).get(ALGORITHM_KEY))[0], 'minmax')

    def test_float8_float4_conv2d_invalid_cfg(self):
        cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {'minmax'}
        }
        model_bfloat16 = copy.deepcopy(TestModelConv2d()).to(torch.bfloat16)

        detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)
        self.assertEqual(len(detail_config.keys()), 0)

    def test_float8_float4_symmetric_invalid_cfg(self):
        cfg_weights_asymmetric = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: False,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.5}}
        }
        cfg_inputs_asymmetric = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: False,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.5}}
        }
        model_bfloat16 = copy.deepcopy(TestModelBias()).to(torch.bfloat16)
        try:
            detail_config = parse_config(model_bfloat16, cfg_weights_asymmetric, AlgorithmRegistry)
            self.assertEqual(len(detail_config.keys()), 0)
        except Exception as e:
            self.assertIn('Weights symmetric only support to be True when weight quant_type is float4_e2m1', str(e))

        try:
            detail_config = parse_config(model_bfloat16, cfg_inputs_asymmetric, AlgorithmRegistry)
            self.assertEqual(len(detail_config.keys()), 0)
        except Exception as e:
            self.assertIn('Inputs symmetric is unsupported to be False when Inputs quant_type is float8_e4m3fn', str(e))

    def test_float8_float4_strategy_invalid_cfg(self):
        cfg_weights_tensor = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.5}}
        }
        cfg_weights_channel = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'channel',
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.5}}
        }
        cfg_inputs_token = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group',
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'token',
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.5}}
        }

        model_bfloat16 = copy.deepcopy(TestModelBias()).to(torch.bfloat16)
        try:
            detail_config = parse_config(model_bfloat16, cfg_weights_tensor, AlgorithmRegistry)
            self.assertEqual(len(detail_config.keys()), 0)
        except Exception as e:
            self.assertIn('act_dtype and wts_dtype float8_e4m3fn float4_e2m1 do not support '
                'weight quant strategy tensor', str(e))

        try:
            detail_config = parse_config(model_bfloat16, cfg_weights_channel, AlgorithmRegistry)
            self.assertEqual(len(detail_config.keys()), 0)
        except Exception as e:
            self.assertIn('act_dtype and wts_dtype float8_e4m3fn float4_e2m1 do not support '
                'weight quant strategy channel', str(e))

        try:
            detail_config = parse_config(model_bfloat16, cfg_inputs_token, AlgorithmRegistry)
            self.assertEqual(len(detail_config.keys()), 0)
        except Exception as e:
            self.assertIn('act_dtype and wts_dtype float8_e4m3fn float4_e2m1 do not support '
                'activation quant strategy token', str(e))

    def test_float8_float4_groupsize_invalid_cfg(self):
        cfg_weights_tensor_groupsize = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                    GROUP_SIZE: 32
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.5}}
        }
        cfg_weights_without_groupsize = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: FLOAT4_E2M1,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: 'group',
                },
                INPUTS: {
                    KEY_TYPE: FLOAT8_E4M3FN,
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                },
            },
            ALGORITHM_KEY: {SMOOTH_QUANT: {SMOOTH_STRENGTH: 0.5}}
        }

        model_bfloat16 = copy.deepcopy(TestModelBias()).to(torch.bfloat16)
        try:
            detail_config = parse_config(model_bfloat16, cfg_weights_tensor_groupsize, AlgorithmRegistry)
            self.assertEqual(len(detail_config.keys()), 0)
        except Exception as e:
            self.assertIn('Weights group_size only support strategy group, but got tensor', str(e))

        try:
            detail_config = parse_config(model_bfloat16, cfg_weights_without_groupsize, AlgorithmRegistry)
            self.assertEqual(len(detail_config.keys()), 0)
        except Exception as e:
            self.assertIn('Weights group_size is necessary, when weights strategy is group', str(e))

    def test_float8_float4_algorithm_invalid_cfg(self):
        algorithm_not_support = {'gptq', 'ofmr'}
        for algorithm in algorithm_not_support:
            cfg_algorithm = {
                KEY_BATCH_NUM: 1,
                QUANT_CFG: {
                    WEIGHTS_KEY: {
                        KEY_TYPE: FLOAT4_E2M1,
                        KEY_SYMMETRIC: True,
                        KEY_STRATEGY: 'group',
                        GROUP_SIZE: 32
                    },
                    INPUTS: {
                        KEY_TYPE: FLOAT8_E4M3FN,
                        KEY_SYMMETRIC: True,
                        KEY_STRATEGY: STRATEGY_TENSOR,
                    },
                },
                ALGORITHM_KEY: f"{algorithm}"
            }
            model_bfloat16 = copy.deepcopy(TestModelBias()).to(torch.bfloat16)
            try:
                detail_config = parse_config(model_bfloat16, cfg_algorithm, AlgorithmRegistry)
                self.assertEqual(len(detail_config.keys()), 0)
            except Exception as e:
                self.assertIn(f'Algorithm {algorithm} do not support act and weight quant dtype '
                    'float8_e4m3fn float4_e2m1', str(e))

            cfg_algorithm_awq = {
                KEY_BATCH_NUM: 1,
                QUANT_CFG: {
                    WEIGHTS_KEY: {
                        KEY_TYPE: FLOAT4_E2M1,
                        KEY_SYMMETRIC: True,
                        KEY_STRATEGY: 'group',
                        GROUP_SIZE: 32
                    },
                    INPUTS: {
                        KEY_TYPE: FLOAT8_E4M3FN,
                        KEY_SYMMETRIC: True,
                        KEY_STRATEGY: STRATEGY_TENSOR,
                    },
                },
                ALGORITHM_KEY: {'awq': {'grids_num': 20}},
            }
            try:
                detail_config = parse_config(model_bfloat16, cfg_algorithm_awq, AlgorithmRegistry)
                self.assertEqual(len(detail_config.keys()), 0)
            except Exception as e:
                self.assertIn(f'Algorithm awq do not support act and weight quant dtype '
                    'float8_e4m3fn float4_e2m1', str(e))

    def test_hifloat8_hifloat8_cfg(self):
        weights_strategy_support = {STRATEGY_TENSOR, 'channel'}
        for weights_strategy in weights_strategy_support:
            cfg = {
                KEY_BATCH_NUM: 1,
                QUANT_CFG: {
                    WEIGHTS_KEY: {
                        KEY_TYPE: HIFLOAT8,
                        KEY_SYMMETRIC: True,
                        KEY_STRATEGY: f"{weights_strategy}",
                    },
                    INPUTS: {
                        KEY_TYPE: HIFLOAT8,
                        KEY_SYMMETRIC: True,
                        KEY_STRATEGY: STRATEGY_TENSOR,
                    },
                },
                ALGORITHM_KEY: {'ofmr'}
            }
            model_bfloat16 = copy.deepcopy(TestModel()).to(torch.bfloat16)
            detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)

            self.assertEqual(len(detail_config.keys()), 3)
            self.assertEqual(detail_config.get(LAYER_LINEAR1).get(KEY_BATCH_NUM), 1)
            self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get('quant_type'), HIFLOAT8)
            self.assertTrue(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_SYMMETRIC))
            self.assertEqual(detail_config.get(LAYER_LINEAR1).get(WEIGHTS_CFG).get(KEY_STRATEGY), f"{weights_strategy}")
            self.assertEqual(detail_config.get(LAYER_LINEAR1).get(INPUTS_CFG).get('quant_type'), HIFLOAT8)
            self.assertTrue(detail_config.get(LAYER_LINEAR1).get(INPUTS_CFG).get(KEY_SYMMETRIC))
            self.assertEqual(detail_config.get(LAYER_LINEAR1).get(INPUTS_CFG).get(KEY_STRATEGY), STRATEGY_TENSOR)
            self.assertEqual(len(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY)), 1)
            self.assertEqual(list(detail_config.get(LAYER_LINEAR1).get(ALGORITHM_KEY))[0], 'ofmr')

    def test_hifloat8_hifloat8_conv2d_cfg(self):
        weights_strategy_support = {STRATEGY_TENSOR, 'channel'}
        for weights_strategy in weights_strategy_support:
            cfg = {
                KEY_BATCH_NUM: 1,
                QUANT_CFG: {
                    WEIGHTS_KEY: {
                        KEY_TYPE: HIFLOAT8,
                        KEY_SYMMETRIC: True,
                        KEY_STRATEGY: f"{weights_strategy}",
                    },
                    INPUTS: {
                        KEY_TYPE: HIFLOAT8,
                        KEY_SYMMETRIC: True,
                        KEY_STRATEGY: STRATEGY_TENSOR,
                    },
                },
                ALGORITHM_KEY: {'ofmr'}
            }
            model_bfloat16 = copy.deepcopy(TestModelConv2d()).to(torch.bfloat16)
            detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)

            self.assertEqual(len(detail_config.keys()), 3)
            self.assertEqual(detail_config.get(CONV2D1).get(KEY_BATCH_NUM), 1)
            self.assertEqual(detail_config.get(CONV2D1).get(WEIGHTS_CFG).get('quant_type'), HIFLOAT8)
            self.assertTrue(detail_config.get(CONV2D1).get(WEIGHTS_CFG).get(KEY_SYMMETRIC))
            self.assertEqual(detail_config.get(CONV2D1).get(WEIGHTS_CFG).get(KEY_STRATEGY), f"{weights_strategy}")
            self.assertEqual(detail_config.get(CONV2D1).get(INPUTS_CFG).get('quant_type'), HIFLOAT8)
            self.assertTrue(detail_config.get(CONV2D1).get(INPUTS_CFG).get(KEY_SYMMETRIC))
            self.assertEqual(detail_config.get(CONV2D1).get(INPUTS_CFG).get(KEY_STRATEGY), STRATEGY_TENSOR)
            self.assertEqual(len(detail_config.get(CONV2D1).get(ALGORITHM_KEY)), 1)
            self.assertEqual(list(detail_config.get(CONV2D1).get(ALGORITHM_KEY))[0], 'ofmr')

    def test_multi_algo_fail_cfg(self):
        cfg = {
            KEY_BATCH_NUM: 1,
            QUANT_CFG: {
                WEIGHTS_KEY: {
                    KEY_TYPE: 'int8',
                    KEY_SYMMETRIC: True,
                    KEY_STRATEGY: STRATEGY_TENSOR,
                }
            },
            ALGORITHM_KEY: {
                'minmax',
                'gptq'
            }
        }
        model_bfloat16 = copy.deepcopy(TestModel()).to(torch.bfloat16)
        try:
            parse_config(model_bfloat16, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn(f'One src_op only support one algorithm, current algo', str(e))

    def test_int8_minmax_skip_lm_head_with_hidden_layers(self):
        """INT8_MINMAX_WEIGHT_QUANT_CFG should skip lm_head but quantize all hidden layers.

        Simulates a realistic LLM structure: multiple hidden Linear layers + lm_head.
        Verifies that skip_layers={'lm_head'} in the built-in config correctly excludes
        only lm_head while all hidden layers remain quantizable.
        """
        class RealisticLLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_q = nn.Linear(4096, 4096)
                self.attn_k = nn.Linear(4096, 4096)
                self.attn_v = nn.Linear(4096, 4096)
                self.attn_out = nn.Linear(4096, 4096)
                self.mlp_gate = nn.Linear(4096, 11008)
                self.mlp_up = nn.Linear(4096, 11008)
                self.mlp_down = nn.Linear(11008, 4096)
                self.lm_head = nn.Linear(4096, 151936, bias=False)  # Qwen2 vocab_size
            def forward(self, x):
                return self.lm_head(self.mlp_down(torch.relu(self.mlp_gate(x) * self.mlp_up(x))))
        model = RealisticLLM().to(torch.bfloat16)
        detail_config = parse_config(model, INT8_MINMAX_WEIGHT_QUANT_CFG, AlgorithmRegistry)
        # All 7 hidden layers should be quantized
        for layer_name in ['attn_q', 'attn_k', 'attn_v', 'attn_out',
                           'mlp_gate', 'mlp_up', 'mlp_down']:
            self.assertIn(layer_name, detail_config,
                          f"Hidden layer '{layer_name}' should be quantized")
        # lm_head must be skipped
        self.assertNotIn('lm_head', detail_config)
        self.assertEqual(len(detail_config), 7)

    def test_int8_minmax_lm_head_auto_skipped_by_dim_limit(self):
        """Even without explicit skip_layers, check_quant_op_constraint should
        auto-skip layers whose weight dims exceed NPU limit (65535).

        This validates the defensive check works as a safety net when users
        provide a custom config that forgets to skip lm_head.
        """
        from amct_pytorch.common.config.parser import check_quant_op_constraint
        from amct_pytorch.common.config.fields import QuantConfig
        # Custom config WITHOUT skip_layers — simulates user oversight
        cfg_no_skip = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
            },
            'algorithm': {'minmax'},
        }
        quant_config = QuantConfig(cfg_no_skip, AlgorithmRegistry)
        # Qwen2-style lm_head: 4096 -> 151936 (exceeds 65535)
        lm_head = nn.Linear(4096, 151936, bias=False)
        self.assertFalse(
            check_quant_op_constraint(lm_head, 'lm_head', 'NOT_QUANTIZE int8', quant_config),
            "lm_head with vocab_size=151936 should be auto-skipped by NPU dim limit check"
        )
        # LLaMA-style lm_head: 4096 -> 32000 (within limit, should pass)
        llama_lm_head = nn.Linear(4096, 32000, bias=False)
        self.assertTrue(
            check_quant_op_constraint(llama_lm_head, 'lm_head', 'NOT_QUANTIZE int8', quant_config),
            "lm_head with vocab_size=32000 should pass (within NPU dim limit)"
        )


class TestNpuWeightQuantDimLimit(unittest.TestCase):
    """Verify NPU weight-quant operator dimension limit (65535) in check_quant_op_constraint.

    The aclnnWeightQuantBatchMatmulV2 operator has a hard limit of 65535 for both
    k (input features) and n (output features). This class tests the defensive
    check that auto-skips layers exceeding this limit.
    """

    def setUp(self):
        from amct_pytorch.common.config.parser import check_quant_op_constraint
        from amct_pytorch.common.config.fields import QuantConfig
        self.check_fn = check_quant_op_constraint
        base_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
            },
            'algorithm': {'minmax'},
        }
        self.quant_config = QuantConfig(base_cfg, AlgorithmRegistry)

    def test_boundary_and_exceeds(self):
        """Boundary value 65535 passes; 65536 is rejected.
        Tests both output_features (n) and input_features (k) dimensions.
        """
        # Boundary: exactly 65535 should pass
        self.assertTrue(self.check_fn(
            nn.Linear(4096, 65535, bias=False), 'layer_n_boundary',
            'NOT_QUANTIZE int8', self.quant_config))
        self.assertTrue(self.check_fn(
            nn.Linear(65535, 4096, bias=False), 'layer_k_boundary',
            'NOT_QUANTIZE int8', self.quant_config))
        # Just over: 65536 should fail
        self.assertFalse(self.check_fn(
            nn.Linear(4096, 65536, bias=False), 'layer_n_exceed',
            'NOT_QUANTIZE int8', self.quant_config))
        self.assertFalse(self.check_fn(
            nn.Linear(65536, 4096, bias=False), 'layer_k_exceed',
            'NOT_QUANTIZE int8', self.quant_config))

    def test_real_model_vocab_sizes(self):
        """Test with real-world model vocab sizes to validate the fix for issue #157."""
        # Qwen2-7B: vocab_size=151936 >> 65535 → must skip
        self.assertFalse(self.check_fn(
            nn.Linear(4096, 151936, bias=False), 'qwen2_lm_head',
            'NOT_QUANTIZE int8', self.quant_config))
        # Qwen3-8B: vocab_size=151936 >> 65535 → must skip
        self.assertFalse(self.check_fn(
            nn.Linear(4096, 151936, bias=False), 'qwen3_lm_head',
            'NOT_QUANTIZE int8', self.quant_config))
        # LLaMA2-7B: vocab_size=32000 < 65535 → should pass
        self.assertTrue(self.check_fn(
            nn.Linear(4096, 32000, bias=False), 'llama2_lm_head',
            'NOT_QUANTIZE int8', self.quant_config))

    def test_dim_limit_only_for_weight_only_combos(self):
        """The NPU dim limit check should ONLY apply to weight-only quantization.

        Weight-only combos (NOT_QUANTIZE *) use aclnnWeightQuantBatchMatmulV2
        which has the 65535 limit. Full quantization combos use different operators
        and should NOT be affected by this constraint.
        """
        large_layer = nn.Linear(4096, 65536, bias=False)
        # Weight-only combos should be rejected by dim limit
        for comb in ['NOT_QUANTIZE int8', 'NOT_QUANTIZE int4',
                     'NOT_QUANTIZE float8_e4m3fn', 'NOT_QUANTIZE float4_e2m1']:
            self.assertFalse(
                self.check_fn(large_layer, 'oversized', comb, self.quant_config),
                f"Dim limit check should reject oversized layer for weight-only comb={comb}"
            )

    def test_full_quant_combos_not_affected_by_dim_limit(self):
        """Full quantization combos should NOT be rejected by the NPU dim limit.

        Combos like int8 int8, float8_e4m3fn float8_e4m3fn use different NPU
        operators that do not have the 65535 dimension constraint.
        """
        large_layer = nn.Linear(4096, 65536, bias=False)
        full_quant_combs = ['int8 int8', 'float8_e4m3fn float8_e4m3fn']
        for comb in full_quant_combs:
            # These should NOT return False due to dim limit
            # (they may return True or be caught by other checks, but not dim limit)
            result = self.check_fn(large_layer, 'oversized_full_quant', comb, self.quant_config)
            # int8 int8 with large dims should pass (no dim limit for full quant)
            self.assertTrue(
                result,
                f"Full quant comb={comb} should NOT be rejected by NPU dim limit check"
            )

    def test_dim_limit_does_not_mask_other_checks(self):
        """A layer within dim limit should still be checked by subsequent constraints.

        For float8_e4m3fn float4_e2m1, cin must be multiple of 64.
        A layer with cin=63 (< 65535) should pass the dim check but fail the cin%64 check.
        """
        # cin=63 is not multiple of 64, but within dim limit
        bad_cin_layer = nn.Linear(63, 128, bias=False)
        self.assertFalse(
            self.check_fn(bad_cin_layer, 'bad_cin', 'float8_e4m3fn float4_e2m1',
                          self.quant_config),
            "Layer with cin=63 should fail cin%64 check for float8_e4m3fn float4_e2m1"
        )
        # cin=64 passes both checks
        good_layer = nn.Linear(64, 128, bias=False)
        self.assertTrue(
            self.check_fn(good_layer, 'good_layer', 'float8_e4m3fn float4_e2m1',
                          self.quant_config),
            "Layer with cin=64 should pass all checks"
        )
