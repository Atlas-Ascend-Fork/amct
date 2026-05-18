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
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class TestQuantTensorNoNpu(unittest.TestCase):
    """quant_tensor / quant_dequant_tensor must not require torch_npu for INT8."""

    def setUp(self):
        # sys.modules[key] = None blocks all subsequent "import key" with ImportError.
        # This is the correct way to simulate absence of an installed package.
        sys.modules['torch_npu'] = None

    def tearDown(self):
        sys.modules.pop('torch_npu', None)

    def test_quant_tensor_int8_no_torch_npu(self):
        """INT8 quant_tensor must work when torch_npu import is blocked."""
        from amct_pytorch.utils.quant_util import quant_tensor
        w = torch.randn(4, 4).to(torch.bfloat16)
        scale = torch.tensor([0.1]).to(torch.bfloat16)
        q, _ = quant_tensor(w, 'int8', scale)
        self.assertEqual(q.dtype, torch.int8)

    def test_quant_dequant_tensor_int8_no_torch_npu(self):
        """INT8 quant_dequant_tensor must work when torch_npu import is blocked."""
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        w = torch.randn(4, 4).to(torch.bfloat16)
        scale = torch.tensor([0.1]).to(torch.bfloat16)
        dq = quant_dequant_tensor(w, 'int8', scale)
        self.assertEqual(dq.dtype, torch.bfloat16)
        self.assertEqual(dq.shape, w.shape)
