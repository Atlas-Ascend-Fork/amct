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
import copy
import unittest
import sys
from unittest.mock import MagicMock
from unittest.mock import patch
import torch
import torch.nn as nn
from utils import TestModelDeepseekV3Attention
from mock_torch_npu import *
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.cache_utils import DynamicCache
from amct_pytorch import quantize, convert

torch.manual_seed(0)


class TestDeepseekV3Attention(unittest.TestCase):
    '''
    ST FOR KVCACHE 
    '''
    @classmethod
    def setUpClass(cls):
        config = DeepseekV3Config()
        cls.test_model = TestModelDeepseekV3Attention(config).to(torch.bfloat16)
        cls.hidden_states = torch.randn(1, 16, config.hidden_size).to(torch.bfloat16)
        cls.kvcache_ori = DynamicCache()
        cls.kvcache_quant = DynamicCache()
        cls.kvcache = DynamicCache()
        for i in range(5):
            cls.ori_out = cls.test_model(cls.hidden_states, past_key_values=cls.kvcache_ori)
        print('TestDeepseekV3Attention START!')

    @classmethod
    def tearDownClass(cls):
        print('TestDeepseekV3Attention END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu

    def tearDown(self):
        del sys.modules['torch_npu']
        
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_anti_quant', wraps=mock_npu_anti_quant)
    def test_quantize_deepseekv3_success(self, mock_1, mock_2):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'kvcache': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'quantile'},
        }
        model = copy.deepcopy(self.test_model)
        quantize(model, cfg)
        model(self.hidden_states, past_key_values=self.kvcache_quant)
        self.assertEqual(type(model.attn).__name__, 'DeepseekV3AttentionQuant')
        self.assertIsNotNone(model.attn.scale_k)
        self.assertIsNotNone(model.attn.scale_v)
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.attn).__name__, 'NpuDeepseekV3AttentionQuant')
        for i in range(5):
            quant_out = model(self.hidden_states.npu(), past_key_values=self.kvcache)