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
import sys
import torch
import torch.nn as nn

from amct_pytorch import quantize, convert
from utils import TestModel
from mock_torch_npu import *
from unittest.mock import MagicMock
from unittest.mock import patch

torch.manual_seed(0)

class TestGptq(unittest.TestCase):
    '''
    ST FOR GPTQ ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        print('TestGptq START!')

    @classmethod
    def tearDownClass(cls):
        print('TestGptq END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu
 
    def tearDown(self):
        del sys.modules['torch_npu']

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int8_tensor_sym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'GPTQuant')
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuWeightQuantizedLinear')
    

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int8_tensor_asym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'GPTQuant')
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear1.offset_w)
        self.assertIsNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuWeightQuantizedLinear')
    
    
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int8_channel_sym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'GPTQuant')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuWeightQuantizedLinear')
    

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int8_channel_asym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'GPTQuant')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuWeightQuantizedLinear')
    
    
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int8_group_sym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
    

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int8_group_asym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'group',
                    'group_size': 32
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
    
    # Not Quant - int4
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int4_tensor_sym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int4_tensor_asym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': False,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int4_channel_sym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'gptq'}
            }

        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int4_channel_asym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': False,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
    
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int4_group_sym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int4_group_asym_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': False,
                    'strategy': 'group',
                    'group_size': 32
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant)
    def test_fp4_group_sym_gptq_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'Linear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        self.assertIsNotNone(model.linear1.scale_w)
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'Linear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant)
    def test_hifp8_tensor_sym_gptq_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'GPTQuant')
        self.assertIsNotNone(model.linear1.scale_w)
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuWeightQuantizedLinear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant)
    def test_fp8_tensor_sym_gptq_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'gptq'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'GPTQuant')
        self.assertIsNotNone(model.linear1.scale_w)
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuWeightQuantizedLinear')

