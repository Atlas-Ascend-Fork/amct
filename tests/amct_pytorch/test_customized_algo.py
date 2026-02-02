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
import numpy as np

from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.algorithm import AlgorithmRegistry
from amct_pytorch import quantize, convert, algorithm_register
from utils import TestModel

torch.manual_seed(0)

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

class TestCustomizedAlgo(unittest.TestCase):
    '''
    ST FOR CUSTOMIZED ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs).to(torch.float32).detach().to('cpu').numpy().astype(np.float32)
        print('TestCustomizedAlgo START!')

    @classmethod
    def tearDownClass(cls):
        print('TestCustomizedAlgo END!')


    def test_customize_algo_quantize_success(self):
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
        algorithm_register('AA', 'Linear', CustomQuant, CustomDeployQuant)
        self.assertEqual(AlgorithmRegistry.algo['AA']['Linear'], CustomQuant)
        self.assertEqual(AlgorithmRegistry.quant_to_deploy[CustomQuant], [CustomDeployQuant])
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        self.assertEqual(type(model.linear1).__name__, 'CustomQuant')
        self.assertEqual(type(model.linear2).__name__, 'CustomQuant')
        self.assertEqual(type(model.linear3).__name__, 'CustomQuant')
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'CustomDeployQuant')
        self.assertEqual(type(model.linear2).__name__, 'CustomDeployQuant')
        self.assertEqual(type(model.linear3).__name__, 'CustomDeployQuant')

    def test_customize_algo_repeated_register_fail(self):
        try:
            algorithm_register('AA', 'BB', CustomQuant, None)
        except Exception as e:
            self.assertIn('AA is already registered', str(e))
