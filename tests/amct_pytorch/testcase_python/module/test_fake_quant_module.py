#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import os
import shutil
import torch
from torch import nn
import unittest
import copy
import numpy as np

from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.fake_quant.fake_quant_module import \
    FakeQuantConv2d, FakeQuantLinear
from amct_pytorch.graph_based_compression.amct_pytorch.utils.model_util import ModuleHelper

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class CustomizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, 3)
        self.conv_2 = torch.nn.Conv2d(8, 8, 3)
        self.matmul_1 = torch.nn.Linear(8, 8)
        self.matmul_2 = torch.nn.Linear(8, 4)
        self.matmul_3 = torch.nn.Linear(4, 4)
        self.relu_0 = torch.nn.ReLU()

    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.conv_2(y)
        y = y.transpose(1,3)
        y = self.matmul_1(y)
        y = self.matmul_2(y)
        y = self.matmul_3(y)
        y = self.relu_0(y)
        return y

class CustomizedModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, 3)
        self.conv_2 = torch.nn.Conv2d(8, 8, 3)
        self.convtranspose_1 = torch.nn.ConvTranspose2d(8,8,3)
        self.matmul_1 = torch.nn.Linear(8, 8)
        self.matmul_2 = torch.nn.Linear(8, 4)
        self.matmul_3 = torch.nn.Linear(4, 4)
        self.relu_0 = torch.nn.ReLU()

    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.conv_2(y)
        y = self.convtranspose_1(y)
        y = y.transpose(1,3)
        y = self.matmul_1(y)
        y = self.matmul_2(y)
        y = self.matmul_3(y)
        y = self.relu_0(y)
        return y

def replace_to_fakequant(model, quant_params):
    for name, mod in model.named_modules():
        if type(mod).__name__ == 'Conv2d':
            quant_mod = FakeQuantConv2d(mod, quant_params, name)
        elif type(mod).__name__ == 'Linear':
            quant_mod = FakeQuantLinear(mod, quant_params, name)
        else:
            return
        ModuleHelper.replace_module_by_name(model, name, quant_mod)


class TestFakeQuantModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join(CUR_DIR, 'temp')
        if not os.path.exists(cls.temp_dir):
            os.mkdir(cls.temp_dir)
        print('start TestFakeQuantModule')
        cls.model = CustomizedModel()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
        print('end TestFakeQuantModule')

    def setUp(self):
        pass

    def testDown(self):
        pass

    def test_fake_quant_module_weight_quantize_float32_hif8_round(self):
        model = copy.deepcopy(self.model)
        quant_params = {'data_scale': np.array([2.], dtype=np.float32),
                        'data_offset': np.array([], dtype=np.int32),
                        'weight_scale': np.array([32.], dtype=np.float32),
                        'deq_scale':  np.array([64.], dtype=np.float32),
                        'act_type': 'HIFLOAT8',
                        'wts_type': 'HIFLOAT8',
                        'round_mode': 'ROUND'}
        replace_to_fakequant(model, quant_params)
        out = model(torch.randn(8, 3, 32, 32))
        self.assertEqual(out.dtype, torch.float32)

    def test_fake_quant_module_weight_quantize_float32_hif8_hybrid(self):
        model = copy.deepcopy(self.model)
        quant_params = {'data_scale': np.array([2.], dtype=np.float32),
                        'data_offset': np.array([], dtype=np.int32),
                        'weight_scale': np.array([32.], dtype=np.float32),
                        'deq_scale':  np.array([64.], dtype=np.float32),
                        'act_type': 'HIFLOAT8',
                        'wts_type': 'HIFLOAT8',
                        'round_mode': 'HYBRID'}
        replace_to_fakequant(model, quant_params)
        out = model(torch.randn(8, 3, 32, 32))
        self.assertEqual(out.dtype, torch.float32)

    def test_fake_quant_module_weight_quantize_float32_float8_rint(self):
        model = copy.deepcopy(self.model)
        quant_params = {'data_scale': np.array([2.], dtype=np.float32),
                        'data_offset': np.array([], dtype=np.int32),
                        'weight_scale': np.array([32.], dtype=np.float32),
                        'deq_scale':  np.array([64.], dtype=np.float32),
                        'act_type': 'FLOAT8_E4M3FN',
                        'wts_type': 'FLOAT8_E4M3FN',
                        'round_mode': 'RINT'}
        replace_to_fakequant(model, quant_params)
        out = model(torch.randn(8, 3, 32, 32))
        self.assertEqual(out.dtype, torch.float32)

    def test_fake_quant_module_weight_quantize_float32_a8w8(self):
        model = copy.deepcopy(self.model)
        quant_params = {'data_scale': np.array([2.], dtype=np.float32),
                        'data_offset': np.array([2], dtype=np.int32),
                        'weight_scale': np.array([32.], dtype=np.float32),
                        'deq_scale':  np.array([64.], dtype=np.float32),
                        'act_type': 'INT8',
                        'wts_type': 'INT8'}
        replace_to_fakequant(model, quant_params)
        out = model(torch.randn(8, 3, 32, 32))
        self.assertEqual(out.dtype, torch.float32)

    def test_fakequant_linear_success(self):
        mod = torch.nn.Linear(4, 3)
        pt_file = os.path.join(self.temp_dir, 'quant_result.pt')
        quant_info = {'quant_result_path': pt_file,
                        'act_type': 'INT8',
                        'wts_type': 'INT8'}
        quant_factors = {'matmul1': 
                            {'quant_factors':
                                {'scale_w': torch.tensor([32.]),
                                    'scale_d': torch.tensor([2.,]),
                                    'offset_d': torch.tensor([2.,])
                                }
                            }
                        }
        torch.save(quant_factors, pt_file)
        fakequant_mod = FakeQuantLinear(mod, quant_info, 'matmul1')
        out = fakequant_mod(torch.randn(8, 3, 4, 4))
        self.assertEqual(out.dtype, torch.float32)

        mod = torch.nn.Linear(4, 3)
        pt_file = os.path.join(self.temp_dir, 'quant_result.pt')
        quant_info = {'quant_result_path': pt_file,
                        'act_type': 'INT8',
                        'wts_type': 'INT8'}
        quant_factors = {'matmul1': 
                            {'quant_factors':
                                {'scale_w': torch.tensor([32., 32., 32.]),
                                    'scale_d': torch.tensor([2.,]),
                                    'offset_d': torch.tensor([2.,])
                                }
                            }
                        }
        torch.save(quant_factors, pt_file)
        fakequant_mod = FakeQuantLinear(mod, quant_info, 'matmul1')
        out = fakequant_mod(torch.randn(8, 3, 4, 4))
        self.assertEqual(out.dtype, torch.float32)

    def test_fakequant_linear_fail_dim_wrong(self):
        mod = torch.nn.Linear(4, 3)
        pt_file = os.path.join(self.temp_dir, 'quant_result.pt')
        quant_info = {'quant_result_path': pt_file,
                        'act_type': 'INT8',
                        'wts_type': 'INT8'}
        quant_factors = {'matmul1': 
                            {'quant_factors':
                                {'scale_w': torch.tensor([32., 32., 32.]),
                                    'scale_d': torch.tensor([2.,]),
                                    'offset_d': torch.tensor([2.,])
                                }
                            }
                        }
        torch.save(quant_factors, pt_file)
        fakequant_mod = FakeQuantLinear(mod, quant_info, 'matmul1')
        self.assertRaises(RuntimeError, fakequant_mod, torch.randn(8, 3, 4, 3, 3, 3, 3))
        self.assertRaises(RuntimeError, fakequant_mod, torch.randn(3))
