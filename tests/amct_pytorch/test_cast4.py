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
import logging
import sys
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from utils import TestModel, TestModelBias

from amct_pytorch import quantize

HIF4_CAST_QUANT = 'HIF4CastQuant'
HIFLOAT4 = 'hifloat4'
CAST_ALGO = 'cast'

torch.manual_seed(0)

logger = logging.getLogger(__name__)


class TestCast4(unittest.TestCase):
    '''
    ST FOR HIFLOAT4 CAST (FAKE-QUANT) ALGORITHM

    HiFloat4 has no native torch_npu dtype, so the cast algorithm runs as a pure-FP
    simulation (vendored hifx4 quant-dequant). These tests therefore exercise only the
    quantize() + forward (simulation) path and assert the inserted module type; there is
    no convert() because no native HiFloat4 deploy op exists yet (Path 2 future work).
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        logger.info('TestCast4 START!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestCast4 END!')

    def setUp(self):
        # HiFloat4 fake-quant never calls torch_npu, but the quantize pipeline imports it at
        # module load; provide a stub so the import resolves on a non-NPU host.
        sys.modules['torch_npu'] = MagicMock()

    def tearDown(self):
        del sys.modules['torch_npu']

    def _assert_all_cast(self, model):
        self.assertEqual(type(model.linear1).__name__, HIF4_CAST_QUANT)
        self.assertEqual(type(model.linear2).__name__, HIF4_CAST_QUANT)
        self.assertEqual(type(model.linear3).__name__, HIF4_CAST_QUANT)

    def test_hif4_weights_tensor_sym_cast_success(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': HIFLOAT4,
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {CAST_ALGO},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        self._assert_all_cast(model)
        out = model(self.inputs)
        self.assertTrue(torch.isfinite(out).all())

    def test_hif4_weights_channel_sym_cast_success(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': HIFLOAT4,
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {CAST_ALGO},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        self._assert_all_cast(model)
        out = model(self.inputs)
        self.assertTrue(torch.isfinite(out).all())

    def test_hif4_tensor_tensor_sym_cast_success(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': HIFLOAT4,
                    'symmetric': True,
                    'strategy': 'tensor',
                },
                'inputs': {
                    'type': HIFLOAT4,
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {CAST_ALGO},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        self._assert_all_cast(model)
        out = model(self.inputs)
        self.assertTrue(torch.isfinite(out).all())

    def test_hif4_channel_tensor_sym_cast_success(self):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': HIFLOAT4,
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': HIFLOAT4,
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {CAST_ALGO},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        self._assert_all_cast(model)
        out = model(self.inputs)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
