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


def test_mxfp_forward_with_tensor_v():
    qdq = QuantDequantMx(bits=8)
    x = torch.randn(4, 64)
    v = torch.randn(4, 64)
    y = qdq(x, v=v)
    assert y.shape == x.shape


def test_mxfp_forward_bits16_is_passthrough():
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
    assert torch.allclose(x.grad, torch.ones_like(x), atol=2e-2, rtol=2e-2)

    assert not torch.all(x.grad == 0)


from amct_pytorch.quantization.dtypes.int_impl import (
    dynamic_per_token_quant,
    scale_fp32_to_u64,
    pack_4bit,
    weight_quant,
    int4_assistance_bias,
)
from amct_pytorch.quantization.dtypes.mxfp_impl import (
    down_size,
    up_size,
    unpack_uint4,
    pack_uint4 as mx_pack_uint4,
    f32_to_f4_unpacked,
    f4_unpacked_to_f32,
    round_to_decimal,
    shared_exponents,
    quantize_elewise,
)


class TestIntImpl:
    def test_dynamic_per_token_quant_preserves_shape(self):
        x = torch.randn(4, 32, dtype=torch.float32)
        y = dynamic_per_token_quant(x, bits=8)
        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_dynamic_per_token_quant_4bit_preserves_shape(self):
        x = torch.randn(4, 16, dtype=torch.float32)
        y = dynamic_per_token_quant(x, bits=4)
        assert y.shape == x.shape

    def test_scale_fp32_to_u64_shape(self):
        scale = torch.randn(2, 4, dtype=torch.float32)
        out = scale_fp32_to_u64(scale)
        assert out.dtype == torch.uint64

    def test_pack_4bit_shape(self):
        x = torch.randint(-8, 7, (16, 32), dtype=torch.int8)
        packed = pack_4bit(x)
        assert packed.dtype == torch.int8
        assert packed.shape[1] == 32

    def test_weight_quant_8bit_shape(self):
        w = torch.randn(8, 32, dtype=torch.float32)
        out = weight_quant(w, bits=8)
        assert out.shape == w.shape

    def test_weight_quant_with_tensor_v(self):
        w = torch.randn(8, 32, dtype=torch.float32)
        v = torch.randn(8, 1, dtype=torch.float32)
        out = weight_quant(w, bits=8, v=v)
        assert out.shape == w.shape

    def test_weight_quant_4bit_real_quant(self):
        w = torch.randn(8, 32, dtype=torch.float32)
        qw, scale, bias = weight_quant(w, bits=4, real_quant=True)
        assert qw.dtype == torch.int8
        assert qw.shape[1] == 32
        assert bias.shape == (8,)

    def test_int4_assistance_bias_shape(self):
        weight = torch.randint(-8, 7, (4, 32), dtype=torch.int8)
        scale = torch.randn(4, 1, dtype=torch.float32)
        bias = int4_assistance_bias(weight, scale)
        assert bias.shape == (4,)


class TestMxfpImpl:
    def test_down_size(self):
        assert down_size((4, 8)) == (4, 4)
        with pytest.raises(AssertionError):
            down_size((4, 7))

    def test_up_size(self):
        assert up_size((4, 4)) == (4, 8)

    def test_pack_unpack_roundtrip(self):
        x = torch.randint(0, 16, (4, 8), dtype=torch.uint8)
        packed = mx_pack_uint4(x)
        unpacked = unpack_uint4(packed)
        assert torch.equal(unpacked, x)

    def test_round_to_decimal(self):
        x = torch.tensor([1.0, 1.5, 1.76, 3.0], dtype=torch.float32)
        r = round_to_decimal(x)
        assert r[0].item() == 0.0
        assert r[3].item() == 1.0

    def test_shared_exponents_shape(self):
        x = torch.randn(2, 8, 32, dtype=torch.float32)
        e = shared_exponents(x, emax=8)
        assert e.shape == (2, 8, 1)

    def test_quantize_elewise_shape(self):
        x = torch.randn(2, 4, 16, dtype=torch.float32)
        q = quantize_elewise(x, min_exp=0, max_norm=6.0, shift_val=2)
        assert q.shape == x.shape

    def test_quantize_elewise_with_tensor_v(self):
        x = torch.randn(2, 4, 16, dtype=torch.float32)
        v = torch.randn(2, 4, 16, dtype=torch.float32)
        q = quantize_elewise(x, min_exp=0, max_norm=6.0, shift_val=2, v=v)
        assert q.shape == x.shape

    def test_f4_unpacked_to_f32_returns_float32(self):
        x = torch.randint(0, 16, (4, 8), dtype=torch.uint8)
        out = f4_unpacked_to_f32(x)
        assert out.dtype == torch.float32
        assert out.shape == x.shape


class TestMxfpDeploy:
    def test_deploy_8bit_returns_fp8_weights(self):
        qdq = QuantDequantMx(bits=8)
        x = torch.randn(4, 64, dtype=torch.float32)
        ExMx, E8M0 = qdq.deploy(x)
        assert ExMx.dtype == torch.float8_e4m3fn
        assert E8M0.dtype == torch.uint8

    def test_deploy_4bit_returns_packed_uint8(self):
        qdq = QuantDequantMx(bits=4)
        x = torch.randn(4, 64, dtype=torch.float32)
        ExMx, E8M0 = qdq.deploy(x)
        assert ExMx.dtype == torch.uint8
        assert E8M0.dtype == torch.uint8

    def test_export_deploy_returns_dict(self):
        qdq = QuantDequantMx(bits=8)
        x = torch.randn(4, 64, dtype=torch.float32)
        out = qdq.export_deploy(x)
        assert "qweight" in out
        assert "weight_scale" in out
