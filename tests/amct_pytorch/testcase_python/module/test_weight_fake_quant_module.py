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
from unittest import mock
from unittest.mock import patch
import copy
import numpy as np

from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.fake_quant.weight_fake_quant_module import \
    FakeWeightQuantizedConv2d, FakeWeightQuantizedLinear
from amct_pytorch.graph_based_compression.amct_pytorch.utils.model_util import ModuleHelper

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

def fake_clamp(tensor, min=None, max=None):
    return torch.ones_like(tensor).to(tensor.dtype)

class TestWeightFakeQuantModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join(CUR_DIR, 'tmp')
        os.makedirs(cls.temp_dir, exist_ok=True)
        print('start TestWeightFakeQuantModule')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
        print('end TestWeightFakeQuantModule')

    def setUp(self):
        pass

    def testDown(self):
        pass
        
    @patch.object(torch, 'clamp', fake_clamp)
    def test_weight_fakequant_linear_success(self):
        mod = torch.nn.Linear(3,3)
        pt_file = os.path.join(self.temp_dir, 'quant_result.pt')
        quant_factors = {'name': 
                            {'quant_factors':
                                {'scale_w': torch.tensor([32.]),
                                }
                            }
                        }
        torch.save(quant_factors, pt_file)
        quant_params = {'quant_result_path': pt_file,
                        'weight_compress_only': True,
                        'wts_type': 'INT8'}
        fakequant_mod = FakeWeightQuantizedLinear(mod, quant_params, 'name')
        out = fakequant_mod(torch.randn(8, 3, 4, 3))
        self.assertEqual(out.dtype, torch.float32)

        mod = torch.nn.Linear(3,3)
        pt_file = os.path.join(self.temp_dir, 'quant_result.pt')
        quant_factors = {'name': 
                            {'quant_factors':
                                {'scale_w': torch.tensor([32., 32., 32.]),
                                }
                            }
                        }
        torch.save(quant_factors, pt_file)
        quant_params = {'quant_result_path': pt_file,
                        'weight_compress_only': True,
                        'wts_type': 'INT8'}
        fakequant_mod = FakeWeightQuantizedLinear(mod, quant_params, 'name')
        out = fakequant_mod(torch.randn(8, 3, 4, 3))
        self.assertEqual(out.dtype, torch.float32)

    def test_weight_fakequant_linear_success_HIF8_FP8(self):
        mod = torch.nn.Linear(3,3)
        pt_file = os.path.join(self.temp_dir, 'quant_result.pt')
        quant_factors = {'name': 
                            {'quant_factors':
                                {'scale_w': torch.tensor([32.]),
                                }
                            }
                        }
        torch.save(quant_factors, pt_file)
        quant_params = {'quant_result_path': pt_file,
                        'weight_compress_only': True,
                        'wts_type': 'FLOAT8_E4M3FN'}
        fakequant_mod = FakeWeightQuantizedLinear(mod, quant_params, 'name')
        out = fakequant_mod(torch.randn(8, 3, 4, 3))
        self.assertEqual(out.dtype, torch.float32)

        mod = torch.nn.Linear(3,3)
        pt_file = os.path.join(self.temp_dir, 'quant_result.pt')
        quant_factors = {'name': 
                            {'quant_factors':
                                {'scale_w': torch.tensor([32., 32., 32.]),
                                }
                            }
                        }
        torch.save(quant_factors, pt_file)
        quant_params = {'quant_result_path': pt_file,
                        'weight_compress_only': True,
                        'wts_type': 'FLOAT8_E4M3FN'}
        fakequant_mod = FakeWeightQuantizedLinear(mod, quant_params, 'name')
        out = fakequant_mod(torch.randn(8, 3, 4, 3))
        self.assertEqual(out.dtype, torch.float32)