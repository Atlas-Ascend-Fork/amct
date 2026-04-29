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
import unittest
import torch
import torch.nn as nn

from utils import TestModel, TestModelBias, TestModelConv2d

from amct_pytorch import INT4_AWQ_WEIGHT_QUANT_CFG, INT4_GPTQ_WEIGHT_QUANT_CFG
from amct_pytorch import INT8_MINMAX_WEIGHT_QUANT_CFG, INT8_SMOOTHQUANT_CFG
from amct_pytorch.config import parse_config
from amct_pytorch.algorithm import AlgorithmRegistry
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule

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

class TestConfigParse(unittest.TestCase):
    '''
    ST FOR CONFIG PARSER
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel()
        cls.inputs = torch.randn(64, 64)
        print('TestConfigParse START!')

    @classmethod
    def tearDownClass(cls):
        print('TestConfigParse END!')

    def test_int4_awq_cfg(self):
        detail_config = parse_config(self.test_model, INT4_AWQ_WEIGHT_QUANT_CFG, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 2)
        self.assertEqual(detail_config.get('linear1').get('batch_num'), 1)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('quant_type'), 'int4')
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('strategy'), 'channel')
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('enable_quant'), False)
        self.assertEqual(len(detail_config.get('linear1').get('algorithm')), 1)
        self.assertEqual(detail_config.get('linear1').get('algorithm').get('awq').get('grids_num'), 20)


    def test_int4_gptq_cfg(self):
        detail_config = parse_config(self.test_model, INT4_GPTQ_WEIGHT_QUANT_CFG, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 2)
        self.assertEqual(detail_config.get('linear1').get('batch_num'), 1)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('quant_type'), 'int4')
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('strategy'), 'channel')
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('enable_quant'), False)
        self.assertEqual(len(detail_config.get('linear1').get('algorithm')), 1)
        self.assertEqual(list(detail_config.get('linear1').get('algorithm'))[0], 'gptq')


    def test_int8_minmax_cfg(self):
        model_bfloat16 = copy.deepcopy(self.test_model).to(torch.bfloat16)
        detail_config = parse_config(model_bfloat16, INT8_MINMAX_WEIGHT_QUANT_CFG, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 3)
        self.assertEqual(detail_config.get('linear1').get('batch_num'), 1)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('quant_type'), 'int8')
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('strategy'), 'channel')
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('enable_quant'), False)
        self.assertEqual(len(detail_config.get('linear1').get('algorithm')), 1)
        self.assertEqual(list(detail_config.get('linear1').get('algorithm'))[0], 'minmax')

    
    def test_int8_smooth_cfg(self):
        model_bfloat16 = copy.deepcopy(self.test_model).to(torch.bfloat16)
        detail_config = parse_config(model_bfloat16, INT8_SMOOTHQUANT_CFG, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 3)
        self.assertEqual(detail_config.get('linear1').get('batch_num'), 1)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('quant_type'), 'int8')
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('strategy'), 'channel')
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('quant_type'), 'int8')
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('strategy'), 'tensor')
        self.assertEqual(len(detail_config.get('linear1').get('algorithm')), 1)
        self.assertEqual(detail_config.get('linear1').get('algorithm').get('smoothquant').get('smooth_strength'), 0.5)

    def test_invalid_batch_num(self):
        invalid_batch_num_cfg = {
            'batch_num': 1.2,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel'
                },
            },
            'algorithm': {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }
        try:
            detail_config = parse_config(self.test_model, invalid_batch_num_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Batch num only support positive int, but got', str(e))

    def test_invalid_wts_quant_type(self):
        invalid_quant_type_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int6',
                    'symmetric': True,
                    'strategy': 'channel'
                },
            },
            'algorithm': {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }

        try:
            detail_config = parse_config(self.test_model, invalid_quant_type_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights quant_dtype only support', str(e))
    
    def test_invalid_wts_symmetric(self):
        invalid_symmetric_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': 'true',
                    'strategy': 'tensor'
                },
            },
            'algorithm': {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }

        try:
            detail_config = parse_config(self.test_model, invalid_symmetric_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights symmetric only support bool [True, False], but got', str(e))

    def test_invalid_wts_strategy(self):
        invalid_strategy_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'token'
                },
            },
            'algorithm': {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }

        try:
            detail_config = parse_config(self.test_model, invalid_strategy_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights strategy only support', str(e))

    def test_invalid_inputs_strategy(self):
        invalid_strategy_cfg = {
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
                    'strategy': 'channel'
                },
            },
            'algorithm': {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }

        try:
            detail_config = parse_config(self.test_model, invalid_strategy_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Inputs strategy only support', str(e))

    def test_invalid_inputs_quant_type(self):
        invalid_quant_type_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
                'inputs': {
                    'type': 'uint4',
                    'symmetric': True,
                    'strategy': 'channel'
                },
            },
            'algorithm': {'awq': {'grids_num': 20}},
            'skip_layers': {'lm_head'}

        }
        try:
            detail_config = parse_config(self.test_model, invalid_quant_type_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Inputs quant_dtype only support', str(e))

    def test_unregisted_algo(self):
        invalid_regist_algo_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
            },
            'algorithm': {'cluster'},
        }

        try:
            detail_config = parse_config(self.test_model, invalid_regist_algo_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Not support algorithm', str(e))

    def test_awq_no_param(self):
        invalid_awq_param_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
            },
            'algorithm': {'awq'},
        }

        try:
            detail_config = parse_config(self.test_model, invalid_awq_param_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Awq grids_num is necessary param, pls check and set', str(e))

    def test_awq_invalid_grids_num(self):
        invalid_awq_param_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
            },
            'algorithm': {'awq': {'grids_num': -20}},
        }
        try:
            detail_config = parse_config(self.test_model, invalid_awq_param_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Awq grids_num only support positive int, but got', str(e))

    def test_smooth_invalid_grids_num(self):
        invalid_smooth_param_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel'
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0}},
        }
        try:
            detail_config = parse_config(self.test_model, invalid_smooth_param_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Smoothquant smooth_strength only support float (0, 1), but got', str(e))

    def test_unsupported_quant_type_comb(self):
        unsupported_quant_type_comb_cfg = {
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
                    'strategy': 'tensor'
                },
            },
            'algorithm': {'minmax'},
        }
        try:
            detail_config = parse_config(self.test_model, unsupported_quant_type_comb_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Do not support combination int8 int4 of act and weight quant dtype', str(e))

    def test_unsupported_algo_for_quant_type_comb(self):
        unsupported_algo_quant_type_comb_cfg = {
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
                },
            },
            'algorithm': {'gptq'},
        }
        try:
            detail_config = parse_config(self.test_model, unsupported_algo_quant_type_comb_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Algorithm gptq do not support act and weight quant dtype int8 int8', str(e))

    def test_skip_layers(self):
        int4_awq_skip_layer_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel'
                },
            },
            'algorithm': {'awq': {'grids_num': 20}},
            'skip_layers': {'linear'}

        }
        detail_config = parse_config(self.test_model, int4_awq_skip_layer_cfg, AlgorithmRegistry)
        
        self.assertEqual(len(detail_config.keys()), 0)

    def test_group_size(self):
        int4_group_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
            },
            'algorithm': {'minmax'},

        }
        detail_config = parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        

    def test_group_size_none_strategy_group(self):
        int4_group_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'group'
                },
            },
            'algorithm': {'minmax'},

        }
        try:
            detail_config = parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights group_size is necessary, when weights strategy is group', str(e))
    
    def test_group_size_32_strategy_other(self):
        int4_group_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                    'group_size': 32
                },
            },
            'algorithm': {'minmax'},

        }
        try:
            detail_config = parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('Weights group_size only support strategy group, but got ', str(e))

    def test_group_size_2_strategy_group(self):
        int4_group_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 2
                },
            },
            'algorithm': {'minmax'},

        }
        try:
            detail_config = parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('only support group_size larger than 32 and', str(e))

    def test_group_size_invalid(self):
        int4_group_cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 2.2
                },
            },
            'algorithm': {'minmax'},

        }
        try:
            detail_config = parse_config(self.test_model, int4_group_cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('group_size only support positive int, but got ', str(e))       
    
    def test_int8_int8_group_invalid_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'group',
                    'group_size': 32
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
            detail_config = parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            
            self.assertIn('do not support weight quant strategy group', str(e))    

    def test_unregist_algo_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'AA': {'BB': 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        try:
            detail_config = parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn('Not support algorithm AA, pls regiter it first', str(e))  

    def test_customize_algo_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'AlgorithmA': {'BB': 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        AlgorithmRegistry.register('AlgorithmA', 'Linear', CustomQuant, CustomDeployQuant)
        detail_config = parse_config(model, cfg, AlgorithmRegistry)
        self.assertEqual(len(detail_config.keys()), 3)
        self.assertEqual(detail_config.get('linear1').get('algorithm').get('AlgorithmA').get('BB'), 0.8)


    def test_int8_int8_token_asymetric_invalid_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'tensor'
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'token',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        try:
            detail_config = parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
    
            self.assertIn('Inputs strategy token do not support asymmetric quantization', str(e))

    def test_int8_int8_token_asymmetric_invalid_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'tensor'
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'token',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.8}}
        }
        model = self.test_model.to(torch.bfloat16)
        try:
            detail_config = parse_config(model, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn('Inputs strategy token do not support asymmetric quantization', str(e))

    def test_float8_float4_smooth_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        model_bfloat16 = copy.deepcopy(TestModelBias()).to(torch.bfloat16)
        detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)

        self.assertEqual(len(detail_config.keys()), 1)
        self.assertEqual(detail_config.get('linear2').get('batch_num'), 1)
        self.assertEqual(detail_config.get('linear2').get('weights_cfg').get('quant_type'), 'float4_e2m1')
        self.assertEqual(detail_config.get('linear2').get('weights_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear2').get('weights_cfg').get('strategy'), 'group')
        self.assertEqual(detail_config.get('linear2').get('inputs_cfg').get('quant_type'), 'float8_e4m3fn')
        self.assertEqual(detail_config.get('linear2').get('inputs_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear2').get('inputs_cfg').get('strategy'), 'tensor')
        self.assertEqual(len(detail_config.get('linear2').get('algorithm')), 1)
        self.assertEqual(detail_config.get('linear2').get('algorithm').get('smoothquant').get('smooth_strength'), 0.5)

    def test_float8_float4_minmax_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'minmax'}
        }
        model_bfloat16 = copy.deepcopy(TestModelBias()).to(torch.bfloat16)
        detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)

        self.assertEqual(len(detail_config.keys()), 1)
        self.assertEqual(detail_config.get('linear2').get('batch_num'), 1)
        self.assertEqual(detail_config.get('linear2').get('weights_cfg').get('quant_type'), 'float4_e2m1')
        self.assertEqual(detail_config.get('linear2').get('weights_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear2').get('weights_cfg').get('strategy'), 'group')
        self.assertEqual(detail_config.get('linear2').get('inputs_cfg').get('quant_type'), 'float8_e4m3fn')
        self.assertEqual(detail_config.get('linear2').get('inputs_cfg').get('symmetric'), True)
        self.assertEqual(detail_config.get('linear2').get('inputs_cfg').get('strategy'), 'tensor')
        self.assertEqual(len(detail_config.get('linear2').get('algorithm')), 1)
        self.assertEqual(list(detail_config.get('linear2').get('algorithm'))[0], 'minmax')

    def test_float8_float4_conv2d_invalid_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'minmax'}
        }
        model_bfloat16 = copy.deepcopy(TestModelConv2d()).to(torch.bfloat16)

        detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)
        self.assertEqual(len(detail_config.keys()), 0)

    def test_float8_float4_symmetric_invalid_cfg(self):
        cfg_weights_asymmetric = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': False,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        cfg_inputs_asymmetric = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': False,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
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
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        cfg_weights_channel = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        cfg_inputs_token = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'token',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
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
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'tensor',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        cfg_weights_without_groupsize = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        cfg_inputs_groupsize = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                    'group_size': 32
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
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
                'batch_num': 1,
                'quant_cfg': {
                    'weights': {
                        'type': 'float4_e2m1',
                        'symmetric': True,
                        'strategy': 'group',
                        'group_size': 32
                    },
                    'inputs': {
                        'type': 'float8_e4m3fn',
                        'symmetric': True,
                        'strategy': 'tensor',
                    },
                },
                'algorithm': f"{algorithm}"
            }
            model_bfloat16 = copy.deepcopy(TestModelBias()).to(torch.bfloat16)
            try:
                detail_config = parse_config(model_bfloat16, cfg_algorithm, AlgorithmRegistry)
                self.assertEqual(len(detail_config.keys()), 0)
            except Exception as e:
                self.assertIn(f'Algorithm {algorithm} do not support act and weight quant dtype '
                    'float8_e4m3fn float4_e2m1', str(e))

            cfg_algorithm_awq = {
                'batch_num': 1,
                'quant_cfg': {
                    'weights': {
                        'type': 'float4_e2m1',
                        'symmetric': True,
                        'strategy': 'group',
                        'group_size': 32
                    },
                    'inputs': {
                        'type': 'float8_e4m3fn',
                        'symmetric': True,
                        'strategy': 'tensor',
                    },
                },
                'algorithm': {'awq': {'grids_num': 20}},
            }
            try:
                detail_config = parse_config(model_bfloat16, cfg_algorithm_awq, AlgorithmRegistry)
                self.assertEqual(len(detail_config.keys()), 0)
            except Exception as e:
                self.assertIn(f'Algorithm awq do not support act and weight quant dtype '
                    'float8_e4m3fn float4_e2m1', str(e))

    def test_hifloat8_hifloat8_cfg(self):
        weights_strategy_support = {'tensor', 'channel'}
        for weights_strategy in weights_strategy_support:
            cfg = {
                'batch_num': 1,
                'quant_cfg': {
                    'weights': {
                        'type': 'hifloat8',
                        'symmetric': True,
                        'strategy': f"{weights_strategy}",
                    },
                    'inputs': {
                        'type': 'hifloat8',
                        'symmetric': True,
                        'strategy': 'tensor',
                    },
                },
                'algorithm': {'ofmr'}
            }
            model_bfloat16 = copy.deepcopy(TestModel()).to(torch.bfloat16)
            detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)

            self.assertEqual(len(detail_config.keys()), 3)
            self.assertEqual(detail_config.get('linear1').get('batch_num'), 1)
            self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('quant_type'), 'hifloat8')
            self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('symmetric'), True)
            self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('strategy'), f"{weights_strategy}")
            self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('quant_type'), 'hifloat8')
            self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('symmetric'), True)
            self.assertEqual(detail_config.get('linear1').get('inputs_cfg').get('strategy'), 'tensor')
            self.assertEqual(len(detail_config.get('linear1').get('algorithm')), 1)
            self.assertEqual(list(detail_config.get('linear1').get('algorithm'))[0], 'ofmr')

    def test_hifloat8_hifloat8_conv2d_cfg(self):
        weights_strategy_support = {'tensor', 'channel'}
        for weights_strategy in weights_strategy_support:
            cfg = {
                'batch_num': 1,
                'quant_cfg': {
                    'weights': {
                        'type': 'hifloat8',
                        'symmetric': True,
                        'strategy': f"{weights_strategy}",
                    },
                    'inputs': {
                        'type': 'hifloat8',
                        'symmetric': True,
                        'strategy': 'tensor',
                    },
                },
                'algorithm': {'ofmr'}
            }
            model_bfloat16 = copy.deepcopy(TestModelConv2d()).to(torch.bfloat16)
            detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)

            self.assertEqual(len(detail_config.keys()), 3)
            self.assertEqual(detail_config.get('conv2d1').get('batch_num'), 1)
            self.assertEqual(detail_config.get('conv2d1').get('weights_cfg').get('quant_type'), 'hifloat8')
            self.assertEqual(detail_config.get('conv2d1').get('weights_cfg').get('symmetric'), True)
            self.assertEqual(detail_config.get('conv2d1').get('weights_cfg').get('strategy'), f"{weights_strategy}")
            self.assertEqual(detail_config.get('conv2d1').get('inputs_cfg').get('quant_type'), 'hifloat8')
            self.assertEqual(detail_config.get('conv2d1').get('inputs_cfg').get('symmetric'), True)
            self.assertEqual(detail_config.get('conv2d1').get('inputs_cfg').get('strategy'), 'tensor')
            self.assertEqual(len(detail_config.get('conv2d1').get('algorithm')), 1)
            self.assertEqual(list(detail_config.get('conv2d1').get('algorithm'))[0], 'ofmr')

    def test_multi_algo_fail_cfg(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                }
            },
            'algorithm': {
                'minmax',
                'gptq'
            }
        }
        model_bfloat16 = copy.deepcopy(TestModel()).to(torch.bfloat16)
        try:
            detail_config = parse_config(model_bfloat16, cfg, AlgorithmRegistry)
        except Exception as e:
            self.assertIn(f'One src_op only support one algorithm, current algo', str(e))