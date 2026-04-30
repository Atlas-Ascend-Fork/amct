import pytest
import torch

from amct_pytorch.quantization.dtypes.int import QuantDequantInt
from amct_pytorch.quantization.dtypes.mxfp import QuantDequantMx


@pytest.mark.parametrize("bits", [4, 8])
def test_mxfp_forward_shape_and_dtype_preserved(bits):
    qdq = QuantDequantMx(bits=bits)
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    y = qdq(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_mxfp_bits16_is_passthrough():
    qdq = QuantDequantMx(bits=16)
    x = torch.randn(2, 32)
    assert torch.equal(qdq(x), x)


def test_mxfp_block_size_must_divide_last_dim():
    qdq = QuantDequantMx(bits=8)
    x = torch.randn(2, 17)  # 17 not divisible by 32
    with pytest.raises(RuntimeError):
        qdq(x)


def test_mxfp_zero_input_round_trips_to_zero():
    qdq = QuantDequantMx(bits=8)
    x = torch.zeros(1, 32)
    y = qdq(x)
    assert torch.allclose(y, x)


@pytest.mark.parametrize("bits", [4, 8])
def test_mxfp_quantization_error_bounded(bits):
    qdq = QuantDequantMx(bits=bits)
    x = torch.randn(4, 128, dtype=torch.float32)
    y = qdq(x)
    rel_err = (y - x).abs().mean() / x.abs().mean().clamp_min(1e-6)
    # 4-bit is much coarser than 8-bit; loose upper bound just to catch regressions.
    assert rel_err.item() < (0.6 if bits == 4 else 0.2)


def test_int_weight_bits16_is_passthrough():
    qdq = QuantDequantInt(bits=16, is_act=False)
    x = torch.randn(4, 8)
    assert torch.equal(qdq(x), x)


@pytest.mark.parametrize("is_act,shape", [(False, (4, 16)), (True, (2, 4, 16))])
def test_int_forward_shape_dtype_preserved(is_act, shape):
    qdq = QuantDequantInt(bits=8, is_act=is_act)
    x = torch.randn(*shape, dtype=torch.float32)
    y = qdq(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_int_weight_quant_error_smaller_for_higher_bits():
    x = torch.randn(8, 64, dtype=torch.float32)
    err_8 = (QuantDequantInt(bits=8)(x) - x).abs().mean()
    err_4 = (QuantDequantInt(bits=4)(x) - x).abs().mean()
    assert err_8 < err_4


def test_int_act_quant_error_smaller_for_higher_bits():
    x = torch.randn(4, 32, dtype=torch.float32)
    err_8 = (QuantDequantInt(bits=8, is_act=True)(x) - x).abs().mean()
    err_4 = (QuantDequantInt(bits=4, is_act=True)(x) - x).abs().mean()
    assert err_8 < err_4


def test_int_quant_passes_gradient_through_ste():
    qdq = QuantDequantInt(bits=8)
    x = torch.randn(4, 8, requires_grad=True)
    y = qdq(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))
