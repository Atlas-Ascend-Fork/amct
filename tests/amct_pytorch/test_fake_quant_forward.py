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
import copy
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from mock_torch_npu import (
    mock_npu,
    mock_npu_convert_weight_to_int4pack,
    mock_npu_dtype_cast,
    mock_npu_dynamic_mx_quant,
    mock_npu_dynamic_quant,
    mock_npu_format_cast,
    mock_npu_quant_matmul,
    mock_npu_quantize,
    mock_npu_trans_quant_param,
    mock_npu_weight_quant_batchmatmul,
)

from amct_pytorch import quantize

torch.manual_seed(0)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32, bias=True)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class _ModelBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32, bias=False)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class FakeQuantTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_model = _Model().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)

    def setUp(self):
        m = MagicMock()
        m.npu_quantize = mock_npu_quantize
        m.npu_quant_matmul = mock_npu_quant_matmul
        m.npu_weight_quant_batchmatmul = mock_npu_weight_quant_batchmatmul
        m.npu_convert_weight_to_int4pack = mock_npu_convert_weight_to_int4pack
        m.npu_format_cast = mock_npu_format_cast
        m.npu_dtype_cast = mock_npu_dtype_cast
        m.npu_dynamic_mx_quant = mock_npu_dynamic_mx_quant
        m.npu_dynamic_quant = mock_npu_dynamic_quant
        m.npu_trans_quant_param = mock_npu_trans_quant_param
        sys.modules["torch_npu"] = m
        torch.Tensor.npu = mock_npu

    def tearDown(self):
        del sys.modules["torch_npu"]


class TestMinMaxFakeQuant(FakeQuantTestBase):
    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch(
        "amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_int8_int8_post_calibration_diverges_from_fp(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "tensor"},
                "inputs": {"type": "int8", "symmetric": True, "strategy": "tensor"},
            },
            "algorithm": {"minmax"},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        fp_ref = self.test_model(self.inputs)
        fake_out = model(self.inputs)
        self.assertFalse(torch.allclose(fp_ref, fake_out, atol=0))

    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch(
        "amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_weight_only_int8_diverges_from_fp(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
            },
            "algorithm": {"minmax"},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        fp_ref = self.test_model(self.inputs)
        fake_out = model(self.inputs)
        self.assertFalse(torch.allclose(fp_ref, fake_out, atol=0))


class TestSmoothQuantFakeQuant(FakeQuantTestBase):
    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch(
        "amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_int8_int8_diverges_from_fp(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
                "inputs": {"type": "int8", "symmetric": True, "strategy": "tensor"},
            },
            "algorithm": {"smoothquant": {"smooth_strength": 0.5}},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        fp_ref = self.test_model(self.inputs)
        fake_out = model(self.inputs)
        self.assertFalse(torch.allclose(fp_ref, fake_out, atol=0))

    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch("torch_npu.npu_format_cast", wraps=mock_npu_format_cast)
    @patch("torch_npu.npu_dtype_cast", wraps=mock_npu_dtype_cast)
    @patch("torch_npu.npu_dynamic_mx_quant", wraps=mock_npu_dynamic_mx_quant)
    @patch("torch_npu.npu_trans_quant_param", wraps=mock_npu_trans_quant_param)
    @patch(
        "amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_fp8_fp4_progressive_diverges_from_fp(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {
                    "type": "float4_e2m1",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 32,
                },
                "inputs": {
                    "type": "float8_e4m3fn",
                    "symmetric": True,
                    "strategy": "tensor",
                },
            },
            "algorithm": {"smoothquant": {"smooth_strength": 0.5}},
        }
        ref_model = _ModelBias().to(torch.bfloat16)
        model = copy.deepcopy(ref_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        fp_ref = ref_model(self.inputs)
        fake_out = model(self.inputs)
        self.assertFalse(torch.allclose(fp_ref, fake_out, atol=0))

    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch("torch_npu.npu_format_cast", wraps=mock_npu_format_cast)
    @patch("torch_npu.npu_dtype_cast", wraps=mock_npu_dtype_cast)
    @patch("torch_npu.npu_dynamic_mx_quant", wraps=mock_npu_dynamic_mx_quant)
    @patch("torch_npu.npu_trans_quant_param", wraps=mock_npu_trans_quant_param)
    @patch(
        "amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_fp8_fp4_progressive_passes_group_size(self, *_):
        from amct_pytorch.classic.quantize_op import smooth_quant_module
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {
                    "type": "float4_e2m1",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 32,
                },
                "inputs": {
                    "type": "float8_e4m3fn",
                    "symmetric": True,
                    "strategy": "tensor",
                },
            },
            "algorithm": {"smoothquant": {"smooth_strength": 0.5}},
        }
        model = copy.deepcopy(_ModelBias()).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        with patch.object(smooth_quant_module, 'apply_progressive_quant_dequant',
                          wraps=smooth_quant_module.apply_progressive_quant_dequant) as m:
            model(self.inputs)
        self.assertGreater(m.call_count, 0)
        for call in m.call_args_list:
            self.assertEqual(call.args[3], 32,
                             'group_size from quant_config must be forwarded to apply_progressive_quant_dequant')


class TestOfmrFakeQuant(FakeQuantTestBase):
    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch(
        "amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_hif8_hif8_post_calibration_runs(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {
                    "type": "hifloat8",
                    "symmetric": True,
                    "strategy": "channel",
                },
                "inputs": {"type": "hifloat8", "symmetric": True, "strategy": "tensor"},
            },
            "algorithm": {"ofmr"},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        out = model(self.inputs)
        self.assertEqual(type(model.linear1).__name__, "OfmrQuant")
        self.assertIsNotNone(out)


class TestAWQFakeQuant(FakeQuantTestBase):
    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch(
        "amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_awq_int8_diverges_from_fp(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
            },
            "algorithm": {"awq": {"grids_num": 20, "duo_scaling": False}},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        fp_ref = self.test_model(self.inputs)
        fake_out = model(self.inputs)
        self.assertFalse(torch.allclose(fp_ref, fake_out, atol=0))


class TestGPTQFakeQuant(FakeQuantTestBase):
    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch(
        "amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_gptq_int8_diverges_from_fp(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
            },
            "algorithm": {"gptq"},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        model(self.inputs)
        fp_ref = self.test_model(self.inputs)
        fake_out = model(self.inputs)
        self.assertFalse(torch.allclose(fp_ref, fake_out, atol=0))


class TestQuantileFakeQuant(FakeQuantTestBase):
    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch(
        "amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_static_hif8_post_calibration_runs(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {
                    "type": "hifloat8",
                    "symmetric": True,
                    "strategy": "channel",
                },
                "inputs": {"type": "hifloat8", "symmetric": True, "strategy": "tensor"},
            },
            "algorithm": {"quantile"},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        out = model(self.inputs)
        self.assertEqual(type(model.linear1).__name__, "QuantileQuant")
        self.assertIsNotNone(out)

    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch(
        "amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_dynamic_hif8_post_calibration_runs(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {
                    "type": "hifloat8",
                    "symmetric": True,
                    "strategy": "tensor",
                },
                "inputs": {
                    "type": "hifloat8",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
            },
            "algorithm": {"quantile"},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        out = model(self.inputs)
        self.assertEqual(type(model.linear1).__name__, "QuantileQuant")
        self.assertIsNotNone(out)

    @patch("torch_npu.npu_quantize", wraps=mock_npu_quantize)
    @patch("torch_npu.npu_quant_matmul", wraps=mock_npu_quant_matmul)
    @patch(
        "torch_npu.npu_weight_quant_batchmatmul",
        wraps=mock_npu_weight_quant_batchmatmul,
    )
    @patch(
        "torch_npu.npu_convert_weight_to_int4pack",
        wraps=mock_npu_convert_weight_to_int4pack,
    )
    @patch(
        "amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema",
        MagicMock(return_value=True),
    )
    def test_weight_only_hif8_post_calibration_runs(self, *_):
        cfg = {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {
                    "type": "hifloat8",
                    "symmetric": True,
                    "strategy": "channel",
                },
            },
            "algorithm": {"quantile"},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        out = model(self.inputs)
        self.assertEqual(type(model.linear1).__name__, "QuantileQuant")
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
