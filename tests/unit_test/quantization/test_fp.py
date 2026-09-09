# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY, register_dtype
from amct_pytorch.quantization.dtypes.fp import QuantDequantFp
from amct_pytorch.quantization.dtypes.mxfp_impl import unpack_mxfloat4_to_fp32


def test_fp_registered():
    register_dtype()
    assert DTYPE_REGISTRY.get("fp") is QuantDequantFp


@pytest.mark.parametrize("bits,maximum", [(4, 6.0), (8, 448.0)])
@pytest.mark.parametrize("is_act", [False, True])
def test_fp_uses_one_scale_for_whole_tensor(bits, maximum, is_act):
    x = torch.tensor([[0.0, 0.5, -1.0], [2.0, -4.0, maximum]]) * 3
    qdq = QuantDequantFp(bits, is_act=is_act)
    scale, q = qdq.quant(x)
    assert scale.numel() == 1
    assert scale.dtype == torch.float32
    assert scale.item() == 3
    torch.testing.assert_close(q * scale, x)
    torch.testing.assert_close(qdq(x), x)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp_zero_and_arbitrary_shape(bits, dtype):
    qdq = QuantDequantFp(bits)
    for x in (torch.zeros(2, 3, 17, dtype=dtype), torch.randn(2, 3, 17).to(dtype)):
        y = qdq(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()
        if not x.any():
            assert torch.equal(x, y)


@pytest.mark.parametrize("bits,maximum", [(4, 6.0), (8, 448.0)])
def test_fp_rounding_and_gradient(bits, maximum):
    x = torch.tensor([0.0, 1.1, -1.1, maximum], requires_grad=True)
    v = torch.zeros_like(x, requires_grad=True)
    qdq = QuantDequantFp(bits)
    y = qdq(x, v=v)
    expected = 1.0 if bits == 4 else 1.125
    torch.testing.assert_close(
        y.detach(), torch.tensor([0.0, expected, -expected, maximum])
    )
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
    assert not torch.equal(qdq(x, v=0.6), y)


@pytest.mark.parametrize("bits", [4, 8])
def test_fp_export_matches_fake_quant(bits):
    x = torch.tensor([[0.1, 0.7, -1.3, 18.0]])
    qdq = QuantDequantFp(bits)
    result = qdq.export_deploy(x, v=0.2)
    q = result["qweight"]
    assert q.device.type == "cpu"
    assert q.dtype == (torch.uint8 if bits == 4 else torch.float8_e4m3fn)
    decoded = unpack_mxfloat4_to_fp32(q) if bits == 4 else q.float()
    torch.testing.assert_close(decoded * result["weight_scale"], qdq(x, v=0.2))


def test_fp_passthrough_and_invalid_bits():
    x = torch.randn(2, 3)
    assert QuantDequantFp(16)(x) is x
    with pytest.raises(ValueError, match="bits"):
        QuantDequantFp(6)
    with pytest.raises(ValueError, match="16"):
        QuantDequantFp(16).export_deploy(x)


def test_fp4_export_rejects_odd_last_dimension():
    with pytest.raises(ValueError, match="even"):
        QuantDequantFp(4).export_deploy(torch.ones(2, 3))


def test_fp_deploy_config_describes_tensor_scaling():
    from amct_pytorch.common.models.llm.common.deploy_export import (
        generate_quant_config,
    )

    config = generate_quant_config(None, [], is_fp=True)
    assert config["format"] == "float-quantized"
    assert "weight_block_size" not in config
    for group in config["config_groups"].values():
        for target in ("input_activations", "weights"):
            assert group[target]["type"] == "float"
            assert group[target]["strategy"] == "tensor"
            assert group[target]["group_size"] is None


@pytest.mark.parametrize("bits", [4, 8])
def test_fp_quantizer_integration(bits):
    from types import SimpleNamespace

    from amct_pytorch.quantization.modules.quant_base import (
        ActivationQuantizer,
        WeightQuantizer,
    )

    register_dtype()
    args = SimpleNamespace(quant_dtype="fp", w_bits=bits, algos=[])
    x = torch.tensor([[0.1, 1.1, -1.5, 6.0]])
    for quantizer in (WeightQuantizer(args), ActivationQuantizer(args, bits)):
        quantizer.enable = True
        torch.testing.assert_close(quantizer(x), QuantDequantFp(bits)(x))


@pytest.mark.parametrize("bits", [4, 8])
def test_fp_scalar_empty_and_small_values(bits):
    qdq = QuantDequantFp(bits)
    for x in (torch.tensor(2.0), torch.empty(0, 3), torch.tensor([1e-38, -1e-38])):
        y = qdq(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
    assert qdq(torch.tensor(2.0)).item() == 2.0


@pytest.mark.parametrize("bits", [4, 8])
def test_fp_autoround_integration(bits):
    from types import SimpleNamespace

    from amct_pytorch.algorithms.quant.auto_round import AutoRound

    args = SimpleNamespace(w_size=(2, 4), quant_dtype="fp")
    algo = AutoRound(args, bits)
    x = torch.tensor([[0.1, 1.1, -1.5, 6.0], [0.2, 0.7, -1.3, 3.0]])
    qdq = QuantDequantFp(bits)
    torch.testing.assert_close(algo.quantize(x, qdq), qdq(x))
    algo.quantize(x, qdq).sum().backward()
    assert algo.value.grad is not None and torch.isfinite(algo.value.grad).all()
    assert algo.export_deploy(x, qdq)["weight_scale"].numel() == 1
