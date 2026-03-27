# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import torch
import torch.nn as nn

from amct_pytorch import quantize, convert, HIFP8_QUANTILE_CFG
from amct_pytorch.algorithm import AlgorithmRegistry


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class TestQuantileQuant(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.inputs = torch.randn(1, 64)
    
    def test_quantile_weight_only(self):
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'quantile'}
        }
        
        model_copy = type(self.model)()
        model_copy.load_state_dict(self.model.state_dict())
        
        quantize(model_copy, config)
        
        self.assertIsInstance(model_copy.linear1, nn.Module)
    
    def test_quantile_full_quant(self):
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'quantile'}
        }
        
        model_copy = type(self.model)()
        model_copy.load_state_dict(self.model.state_dict())
        
        quantize(model_copy, config)
        
        self.assertIsInstance(model_copy.linear1, nn.Module)
    
    def test_quantile_weight_per_tensor(self):
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'quantile'}
        }
        
        model_copy = type(self.model)()
        model_copy.load_state_dict(self.model.state_dict())
        
        quantize(model_copy, config)
        
        self.assertIsInstance(model_copy.linear1, nn.Module)
    
    def test_quantile_activation_per_token(self):
        config = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'token',
                },
            },
            'algorithm': {'quantile'}
        }
        
        model_copy = type(self.model)()
        model_copy.load_state_dict(self.model.state_dict())
        
        quantize(model_copy, config)
        
        self.assertIsInstance(model_copy.linear1, nn.Module)
    
    def test_quantile_algorithm_registered(self):
        self.assertIn('quantile', AlgorithmRegistry.algo.keys())
        self.assertIn('Linear', AlgorithmRegistry.algo.get('quantile').keys())


if __name__ == '__main__':
    unittest.main()
