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


class TestBaseQuantizeFakeQuant(unittest.TestCase):
    """_fake_quant_forward produces quant-dequant output, not raw float output."""

    def setUp(self):
        sys.modules.pop('torch_npu', None)

    def _make_weight_only_module(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        linear = nn.Linear(8, 4, bias=True).to(torch.bfloat16)
        cfg = {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': False},
        }
        return MinMaxQuant(linear, 'test_layer', cfg), linear

    def test_fake_quant_forward_differs_from_fp(self):
        """_fake_quant_forward output must differ from raw float output."""
        mod, linear = self._make_weight_only_module()
        x = torch.randn(2, 8).to(torch.bfloat16)
        fp_out = linear(x)
        fake_out = mod._fake_quant_forward(x)
        self.assertFalse(torch.allclose(fp_out, fake_out, atol=0),
                         "fake quant output must differ from float output")

    def test_fake_quant_forward_matches_manual_quant_dequant(self):
        """_fake_quant_forward must equal manual quant_dequant(weight) @ x."""
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        mod, _ = self._make_weight_only_module()
        x = torch.randn(2, 8).to(torch.bfloat16)
        fake_out = mod._fake_quant_forward(x)
        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w, mod.group_size)
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(fake_out, expected),
                        f"max diff: {(fake_out - expected).abs().max()}")
