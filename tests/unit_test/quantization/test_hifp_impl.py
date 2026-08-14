#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from amct_pytorch.quantization.dtypes import hifp_impl
from amct_pytorch.quantization.dtypes.hifp_impl import (
    _amct_ops_hifloat8_fake_quant as amct_ops_hifloat8_fake_quant,
    _floor_log2_fp32 as floor_log2_fp32,
    _ldexp_fp32 as ldexp_fp32,
    _load_amct_ops_cast as load_amct_ops_cast,
    _load_amct_ops_hif4_cast as load_amct_ops_hif4_cast,
    _native_hifloat8_fake_quant as native_hifloat8_fake_quant,
    _pow2 as pow2_exact,
    _to_bf16 as to_bf16_exact,
)


@pytest.fixture(autouse=True)
def _clear_native_probe_cache():
    hifp_impl.is_native_hifloat8_cast_available.cache_clear()
    yield
    hifp_impl.is_native_hifloat8_cast_available.cache_clear()


def _fake_torch_npu(cast_fn=None, with_attrs=True):
    module = ModuleType("torch_npu")
    if with_attrs:
        module.hifloat8 = object()
        module.npu_dtype_cast = cast_fn or (
            lambda tensor, dtype, input_dtype=None: tensor
        )
    return module


def test_native_probe_returns_false_when_torch_npu_is_missing():
    with patch.dict(sys.modules, {"torch_npu": None}):
        assert not hifp_impl.is_native_hifloat8_cast_available()


def test_native_probe_returns_false_when_required_attrs_are_missing():
    module = _fake_torch_npu(with_attrs=False)
    with patch.dict(sys.modules, {"torch_npu": module}):
        assert not hifp_impl.is_native_hifloat8_cast_available()


def test_native_probe_returns_true_after_round_trip():
    calls = []

    def fake_cast(tensor, dtype, input_dtype=None):
        calls.append((dtype, input_dtype))
        return tensor

    module = _fake_torch_npu(cast_fn=fake_cast)
    with (
        patch.dict(sys.modules, {"torch_npu": module}),
        patch.object(torch.Tensor, "npu", lambda self: self, create=True),
    ):
        assert hifp_impl.is_native_hifloat8_cast_available()
    assert calls == [
        (module.hifloat8, None),
        (torch.float16, module.hifloat8),
    ]


def test_native_probe_returns_false_when_round_trip_raises():
    def fake_cast(tensor, dtype, input_dtype=None):
        raise RuntimeError("native hifloat8 cast is unavailable")

    module = _fake_torch_npu(cast_fn=fake_cast)
    with (
        patch.dict(sys.modules, {"torch_npu": module}),
        patch.object(torch.Tensor, "npu", lambda self: self, create=True),
    ):
        assert not hifp_impl.is_native_hifloat8_cast_available()


def test_native_wrapper_uses_expected_cast_arguments():
    calls = []

    def fake_cast(tensor, dtype, input_dtype=None):
        calls.append((tensor, dtype, input_dtype))
        return tensor

    module = _fake_torch_npu(cast_fn=fake_cast)
    x = torch.randn(2, 8, dtype=torch.bfloat16)

    with patch.dict(sys.modules, {"torch_npu": module}):
        out = native_hifloat8_fake_quant(x)

    assert out is x
    assert len(calls) == 2
    assert calls[0][0] is x
    assert calls[0][1:] == (module.hifloat8, None)
    assert calls[1][0] is x
    assert calls[1][1:] == (torch.bfloat16, module.hifloat8)


def test_amct_ops_loader_returns_none_when_module_is_missing():
    package = ModuleType("amct_ops")
    package.__path__ = []
    modules = {"amct_ops": package, "amct_ops.hifloat8_cast": None}
    with patch.dict(sys.modules, modules):
        assert load_amct_ops_cast() is None


def test_amct_ops_loader_returns_encode_and_decode():
    package = ModuleType("amct_ops")
    package.__path__ = []
    module = ModuleType("amct_ops.hifloat8_cast")
    module.encode_to_hifloat8 = lambda tensor: tensor
    module.decode_from_hifloat8 = lambda tensor, dtype: tensor.to(dtype)
    modules = {"amct_ops": package, "amct_ops.hifloat8_cast": module}
    with patch.dict(sys.modules, modules):
        encode, decode = load_amct_ops_cast()

    assert encode is module.encode_to_hifloat8
    assert decode is module.decode_from_hifloat8


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_amct_ops_wrapper_preserves_supported_dtype(dtype):
    encoded_dtypes = []

    def encode(tensor):
        encoded_dtypes.append(tensor.dtype)
        return tensor

    def decode(codes, output_dtype):
        return codes.to(output_dtype)

    x = torch.randn(2, 8, dtype=dtype)
    with patch.object(torch.Tensor, "npu", lambda self: self, create=True):
        out = amct_ops_hifloat8_fake_quant(x, encode, decode)

    assert encoded_dtypes == [dtype]
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_amct_ops_wrapper_uses_bfloat16_for_float32():
    encoded_dtypes = []

    def encode(tensor):
        encoded_dtypes.append(tensor.dtype)
        return tensor

    def decode(codes, output_dtype):
        return codes.to(output_dtype)

    x = torch.randn(2, 8, dtype=torch.float32)
    with patch.object(torch.Tensor, "npu", lambda self: self, create=True):
        out = amct_ops_hifloat8_fake_quant(x, encode, decode)

    assert encoded_dtypes == [torch.bfloat16]
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert out.device == x.device


def test_native_backend_is_preferred():
    x = torch.randn(2, 8, dtype=torch.float16)
    native_result = x + 1
    native_calls = []

    with (
        patch.object(hifp_impl, "is_native_hifloat8_cast_available", return_value=True),
        patch.object(
            hifp_impl,
            "_native_hifloat8_fake_quant",
            side_effect=lambda tensor: native_calls.append(tensor) or native_result,
        ),
    ):
        out = hifp_impl.hifloat8_fake_quant(x)

    assert out is native_result
    assert len(native_calls) == 1
    assert native_calls[0] is x
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_native_backend_does_not_load_amct_ops():
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    native_result = x + 1

    with (
        patch.object(hifp_impl, "is_native_hifloat8_cast_available", return_value=True),
        patch.object(
            hifp_impl, "_native_hifloat8_fake_quant", return_value=native_result
        ),
        patch.object(
            hifp_impl,
            "_load_amct_ops_cast",
            side_effect=AssertionError("amct_ops must not be loaded"),
        ),
    ):
        assert hifp_impl.hifloat8_fake_quant(x) is native_result


def test_native_unavailable_uses_amct_ops():
    x = torch.randn(2, 8, dtype=torch.float32)
    fallback_result = x + 1
    encode = object()
    decode = object()
    fallback_calls = []

    def fake_fallback(tensor, loaded_encode, loaded_decode):
        fallback_calls.append((tensor, loaded_encode, loaded_decode))
        return fallback_result

    with (
        patch.object(
            hifp_impl, "is_native_hifloat8_cast_available", return_value=False
        ),
        patch.object(
            hifp_impl,
            "_native_hifloat8_fake_quant",
            side_effect=AssertionError("native backend must not be called"),
        ),
        patch.object(hifp_impl, "_load_amct_ops_cast", return_value=(encode, decode)),
        patch.object(
            hifp_impl,
            "_amct_ops_hifloat8_fake_quant",
            side_effect=fake_fallback,
        ),
    ):
        out = hifp_impl.hifloat8_fake_quant(x)

    assert out is fallback_result
    assert len(fallback_calls) == 1
    assert fallback_calls[0][0] is x
    assert fallback_calls[0][1:] == (encode, decode)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_native_execution_error_falls_back_to_amct_ops(error_type):
    x = torch.randn(2, 8)
    fallback_result = x + 1

    def fail_native(tensor):
        raise error_type("native failed")

    with (
        patch.object(hifp_impl, "is_native_hifloat8_cast_available", return_value=True),
        patch.object(hifp_impl, "_native_hifloat8_fake_quant", side_effect=fail_native),
        patch.object(
            hifp_impl, "_load_amct_ops_cast", return_value=(object(), object())
        ),
        patch.object(
            hifp_impl,
            "_amct_ops_hifloat8_fake_quant",
            return_value=fallback_result,
        ),
    ):
        assert hifp_impl.hifloat8_fake_quant(x) is fallback_result


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_amct_ops_execution_error_raises_backend_requirement(error_type):
    x = torch.randn(2, 8)

    def fail_amct_ops(tensor, encode, decode):
        raise error_type("amct_ops failed")

    with (
        patch.object(
            hifp_impl, "is_native_hifloat8_cast_available", return_value=False
        ),
        patch.object(
            hifp_impl, "_load_amct_ops_cast", return_value=(object(), object())
        ),
        patch.object(
            hifp_impl,
            "_amct_ops_hifloat8_fake_quant",
            side_effect=fail_amct_ops,
        ),
    ):
        with pytest.raises(RuntimeError) as error:
            hifp_impl.hifloat8_fake_quant(x)

    assert "native HiFloat8 cast" in str(error.value)
    assert "install amct_ops" in str(error.value)


def test_both_backends_unavailable_raises_backend_requirement():
    x = torch.randn(2, 8)

    with (
        patch.object(
            hifp_impl, "is_native_hifloat8_cast_available", return_value=False
        ),
        patch.object(hifp_impl, "_load_amct_ops_cast", return_value=None),
    ):
        with pytest.raises(RuntimeError) as error:
            hifp_impl.hifloat8_fake_quant(x)

    assert "native HiFloat8 cast" in str(error.value)
    assert "install amct_ops" in str(error.value)


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_amct_ops_load_error_raises_backend_requirement(error_type):
    x = torch.randn(2, 8)

    with (
        patch.object(
            hifp_impl, "is_native_hifloat8_cast_available", return_value=False
        ),
        patch.object(
            hifp_impl,
            "_load_amct_ops_cast",
            side_effect=error_type("failed to load amct_ops"),
        ),
    ):
        with pytest.raises(RuntimeError) as error:
            hifp_impl.hifloat8_fake_quant(x)

    assert "native HiFloat8 cast" in str(error.value)
    assert "install amct_ops" in str(error.value)


def test_unexpected_native_error_is_not_swallowed():
    x = torch.randn(2, 8)

    def fail_native(tensor):
        raise ValueError("invalid input")

    with (
        patch.object(hifp_impl, "is_native_hifloat8_cast_available", return_value=True),
        patch.object(hifp_impl, "_native_hifloat8_fake_quant", side_effect=fail_native),
    ):
        with pytest.raises(ValueError, match="invalid input"):
            hifp_impl.hifloat8_fake_quant(x)


def test_both_backend_execution_failures_raise_backend_requirement():
    x = torch.randn(2, 8)
    native_attempts = []
    fallback_attempts = []

    def fail_native(tensor):
        native_attempts.append(tensor)
        raise RuntimeError("native failed")

    def fail_amct_ops(tensor, encode, decode):
        fallback_attempts.append(tensor)
        raise RuntimeError("amct_ops failed")

    with (
        patch.object(hifp_impl, "is_native_hifloat8_cast_available", return_value=True),
        patch.object(hifp_impl, "_native_hifloat8_fake_quant", side_effect=fail_native),
        patch.object(
            hifp_impl, "_load_amct_ops_cast", return_value=(object(), object())
        ),
        patch.object(
            hifp_impl,
            "_amct_ops_hifloat8_fake_quant",
            side_effect=fail_amct_ops,
        ),
    ):
        with pytest.raises(RuntimeError) as error:
            hifp_impl.hifloat8_fake_quant(x)

    assert "native HiFloat8 cast" in str(error.value)
    assert "install amct_ops" in str(error.value)
    assert native_attempts == [x]
    assert fallback_attempts == [x]


# ---------------------------------------------------------------------------
# HiF4 section
# ---------------------------------------------------------------------------


class _FakeNpuTensor:
    """Wraps a CPU tensor but reports device.type == 'npu' (CI-safe NPU dispatch)."""

    def __init__(self, inner):
        self._inner = inner
        self.device = SimpleNamespace(type="npu")
        self.dtype = inner.dtype

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---- bit-exact primitives ----


def test_floor_log2_fp32_returns_exact_exponents():
    x = torch.tensor(
        [1.0, 2.0, 0.5, 2.0**-48, 2.0**-126, 2.0**-149, 0.0, -4.0, 3.0],
        dtype=torch.float32,
    )
    got = floor_log2_fp32(x)
    expected = torch.tensor(
        [0.0, 1.0, -1.0, -48.0, -126.0, -149.0, float("-inf"), 2.0, 1.0],
        dtype=torch.float32,
    )
    assert torch.equal(got, expected)


def test_floor_log2_fp32_propagates_non_finite():
    x = torch.tensor([float("inf"), float("nan")], dtype=torch.float32)
    got = floor_log2_fp32(x)
    assert torch.isinf(got[0]) and torch.isnan(got[1])


def test_floor_log2_fp32_rejects_non_fp32():
    with pytest.raises(TypeError, match="float32"):
        floor_log2_fp32(torch.tensor([1.0], dtype=torch.bfloat16))


def test_pow2_and_ldexp_are_exact():
    e = torch.tensor([0.0, 3.0, -5.0], dtype=torch.float32)
    assert torch.equal(
        pow2_exact(e), torch.tensor([1.0, 8.0, 0.03125], dtype=torch.float32)
    )
    got = ldexp_fp32(
        torch.tensor([1.5, -2.0], dtype=torch.float32),
        torch.tensor([2.0, -1.0], dtype=torch.float32),
    )
    assert torch.equal(got, torch.tensor([6.0, -1.0], dtype=torch.float32))


def test_to_bf16_rounds_half_to_even_on_ties():
    # 1.01171875 = 1 + 1.5*2^-7: tie between 1.0078125 (odd) and 1.015625 (even) -> up
    # 1.00390625 = 1 + 0.5*2^-7: tie between 1.0 (even) and 1.0078125 (odd) -> down
    x = torch.tensor([1.01171875, 1.00390625], dtype=torch.float32)
    assert torch.equal(
        to_bf16_exact(x), torch.tensor([1.015625, 1.0], dtype=torch.float32)
    )


def test_to_bf16_regular_rounding_and_negatives():
    x = torch.tensor([1.0049, -1.0049, 1.006, 0.0], dtype=torch.float32)
    got = to_bf16_exact(x)
    assert got[0].item() == 1.0078125
    assert got[1].item() == -1.0078125
    assert got[2].item() == 1.0078125
    assert got[3].item() == 0.0


# ---- hif4_encode / hif4_decode ----


def test_hif4_encode_structure_and_roundtrip():
    xg = torch.zeros(1, 1, 8, 2, 4, dtype=torch.float32)
    xg[..., :4] = 1.0
    xg[..., 4:] = 3.0
    enc = hifp_impl.hif4_encode(xg)
    assert enc.block_nan.shape == (1, 1) and not enc.block_nan.any()
    assert enc.e1_8.shape == (1, 1, 8)
    assert enc.e1_16.shape == (1, 1, 8, 2)
    assert enc.code.shape == (1, 1, 8, 2, 4)
    assert (enc.q >= 4).all() and (enc.q <= 7).all()
    assert (enc.e_e6 >= -48).all() and (enc.e_e6 <= 15).all()
    assert (enc.de >= 0).all() and (enc.de <= 3).all()
    out = hifp_impl.hif4_decode(
        enc.sign_bit, enc.e6m2, enc.de, enc.code, ng=2, block_nan=enc.block_nan
    )
    assert out.shape == xg.shape
    assert out.abs().max().item() <= 7.0


def test_hif4_encode_flags_non_finite_blocks():
    xg = torch.ones(2, 2, 8, 2, 4, dtype=torch.float32)
    xg[0, 0, 0, 0, 0] = float("nan")
    xg[0, 1, 3, 1, 2] = float("inf")
    xg[1, 0, 5, 0, 1] = float("-inf")
    enc = hifp_impl.hif4_encode(xg)
    assert enc.block_nan.tolist() == [[True, True], [True, False]]
    assert enc.e6m2[1, 1].item() > 0


def test_hif4_encode_zero_block_uses_min_scale():
    xg = torch.zeros(1, 1, 8, 2, 4, dtype=torch.float32)
    enc = hifp_impl.hif4_encode(xg)
    assert not enc.block_nan.any()
    assert enc.e_e6.item() == -48
    assert (enc.code == 0).all()


def test_hif4_decode_without_block_nan():
    sign = torch.zeros(1, 1, 8, 2, 4, dtype=torch.int32)
    e6m2 = torch.full((1, 1), 0.5)
    de = torch.full((1, 1, 8, 2), 1.0)
    code = torch.full((1, 1, 8, 2, 4), 4, dtype=torch.int32)  # 4/4 = 1.0
    out = hifp_impl.hif4_decode(sign, e6m2, de, code, ng=2, block_nan=None)
    assert torch.equal(out, torch.full((1, 1, 8, 2, 4), 1.0))


def test_hif4_decode_block_nan_poisons_whole_block():
    sign = torch.zeros(1, 2, 8, 2, 4, dtype=torch.int32)
    e6m2 = torch.full((1, 2), 0.5)
    de = torch.zeros(1, 2, 8, 2)
    code = torch.full((1, 2, 8, 2, 4), 4, dtype=torch.int32)
    block_nan = torch.tensor([[True, False]])
    out = hifp_impl.hif4_decode(sign, e6m2, de, code, ng=2, block_nan=block_nan)
    assert torch.isnan(out[0, 0]).all()
    assert torch.equal(out[0, 1], torch.full((8, 2, 4), 0.5))


# ---- hif4_pack / hif4_unpack ----


def test_hif4_pack_unpack_roundtrip_matches_fake_quant():
    torch.manual_seed(7)
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    scale, value = hifp_impl.hif4_pack(x)
    assert scale.shape == (3, 4, 4) and scale.dtype == torch.uint8
    assert value.shape == (3, 128) and value.dtype == torch.uint8
    decoded = hifp_impl.hif4_unpack(scale, value)
    ref = hifp_impl.hifloat4_fake_quant(x).float()
    assert torch.equal(decoded, ref)


def test_hif4_pack_element0_in_low_nibble():
    x = torch.tile(
        torch.where(torch.arange(64) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0)),
        (4, 1),
    ).to(torch.bfloat16)
    _, value = hifp_impl.hif4_pack(x)
    assert ((value & 0x8) == 0).all()  # even (positive) elements in low nibble
    assert ((value & 0x80) != 0).all()  # odd (negative) elements in high nibble


def test_hif4_pack_nan_block_writes_ff_and_unpacks_to_nan():
    torch.manual_seed(11)
    x = torch.randn(2, 256, dtype=torch.bfloat16)
    x[0, 3] = float("nan")
    x[1, 64 + 5] = float("inf")
    scale, value = hifp_impl.hif4_pack(x)
    assert scale[0, 0, 0].item() == 0xFF
    assert scale[1, 1, 0].item() == 0xFF
    assert (scale[0, 1:, 0] != 0xFF).all()
    decoded = hifp_impl.hif4_unpack(scale, value)
    assert torch.isnan(decoded[0, :64]).all()
    assert torch.isnan(decoded[1, 64:128]).all()
    assert torch.isfinite(decoded[0, 64:128]).all()


def test_hif4_unpack_accepts_numpy_and_tensor_inputs():
    torch.manual_seed(13)
    x = torch.randn(1, 64, dtype=torch.bfloat16)
    scale, value = hifp_impl.hif4_pack(x)
    out_np = hifp_impl.hif4_unpack(scale.numpy(), value.numpy())
    out_ts = hifp_impl.hif4_unpack(scale, value)
    assert torch.equal(out_np, out_ts)


# ---- amct_ops loader for HiF4 ----


def test_hif4_loader_returns_none_when_module_is_missing():
    package = ModuleType("amct_ops")
    package.__path__ = []
    modules = {"amct_ops": package, "amct_ops.hifloat4_cast": None}
    with patch.dict(sys.modules, modules):
        assert load_amct_ops_hif4_cast() is None


def test_hif4_loader_returns_npu_fake_quant():
    package = ModuleType("amct_ops")
    package.__path__ = []
    module = ModuleType("amct_ops.hifloat4_cast")
    fq = object()
    module.hifloat4_fake_quant = fq
    modules = {"amct_ops": package, "amct_ops.hifloat4_cast": module}
    with patch.dict(sys.modules, modules):
        assert load_amct_ops_hif4_cast() is fq


# ---- hifloat4_fake_quant: CPU reference path ----


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hifloat4_fake_quant_cpu_preserves_dtype_and_shape(dtype):
    torch.manual_seed(17)
    x = torch.randn(2, 128, dtype=dtype)
    out = hifp_impl.hifloat4_fake_quant(x)
    assert out.shape == x.shape
    assert out.dtype == dtype


def test_hifloat4_fake_quant_raises_on_non_multiple_of_64():
    torch.manual_seed(19)
    x = torch.randn(2, 100, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        hifp_impl.hifloat4_fake_quant(x)


def test_hif4_pack_raises_on_non_multiple_of_64():
    x = torch.randn(2, 100, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        hifp_impl.hif4_pack(x)


def test_hifloat4_fake_quant_qdim_first_axis():
    torch.manual_seed(23)
    x = torch.randn(128, 64, dtype=torch.bfloat16)
    out = hifp_impl.hifloat4_fake_quant(x, qdim=0)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_hifloat4_fake_quant_qdim_normalizes_positive_index():
    torch.manual_seed(29)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    out_qdim1 = hifp_impl.hifloat4_fake_quant(x, qdim=1)
    out_qdim_neg1 = hifp_impl.hifloat4_fake_quant(x, qdim=-1)
    assert torch.equal(out_qdim1, out_qdim_neg1)


def test_hifloat4_fake_quant_zeros_stay_zero_and_nan_poisons_block():
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    x[0, 3] = float("nan")
    x[1, 64 + 5] = float("inf")
    out = hifp_impl.hifloat4_fake_quant(x)
    assert torch.isnan(out[0, :64]).all()
    assert torch.isnan(out[1, 64:128]).all()
    assert torch.equal(out[0, 64:128], torch.zeros(64, dtype=torch.bfloat16))
    assert torch.equal(out[1, :64], torch.zeros(64, dtype=torch.bfloat16))


def test_hifloat4_fake_quant_negative_zero_maps_to_zero():
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[0, 1] = -0.0
    out = hifp_impl.hifloat4_fake_quant(x)
    assert out[0, 1].item() == 0.0


def test_hifloat4_fake_quant_hand_computed_golden():
    # all-2.0 block: sf=E6M2(bf16(2/7))=0.3125, rec=bf16(6.4)=3.19921875,
    # x8=6.3984>=4 -> E1=2, x4=3.1992>=2 -> E2=2,
    # mant=bf16(2*rec/4)=1.59375 -> floor(4*m+0.5)/4=1.5 -> out=0.3125*2*2*1.5=1.875
    x = torch.full((1, 64), 2.0, dtype=torch.bfloat16)
    out = hifp_impl.hifloat4_fake_quant(x)
    assert torch.equal(out, torch.full((1, 64), 1.875, dtype=torch.bfloat16))


# ---- hifloat4_fake_quant: NPU dispatch ----


def test_hifloat4_dispatch_uses_npu_kernel_for_bf16_on_npu():
    inner = torch.randn(2, 64, dtype=torch.bfloat16)
    x = _FakeNpuTensor(inner)
    calls = []

    def fq(tensor, qdim=-1):
        calls.append((tensor, qdim))
        return tensor

    with patch.object(hifp_impl, "_load_amct_ops_hif4_cast", return_value=fq):
        out = hifp_impl.hifloat4_fake_quant(x, qdim=0)

    assert out is x
    assert calls == [(x, 0)]


def test_hifloat4_dispatch_falls_back_to_reference_when_kernel_missing():
    inner = torch.randn(2, 64, dtype=torch.bfloat16)
    x = _FakeNpuTensor(inner)
    with patch.object(hifp_impl, "_load_amct_ops_hif4_cast", return_value=None):
        out = hifp_impl.hifloat4_fake_quant(x)
    assert torch.equal(out, hifp_impl.hifloat4_fake_quant(inner))


def test_hifloat4_dispatch_skips_kernel_for_fp32_on_npu():
    inner = torch.randn(2, 64, dtype=torch.float32)
    x = _FakeNpuTensor(inner)
    with patch.object(
        hifp_impl,
        "_load_amct_ops_hif4_cast",
        side_effect=AssertionError("kernel must not be loaded"),
    ):
        out = hifp_impl.hifloat4_fake_quant(x)
    assert torch.equal(out, hifp_impl.hifloat4_fake_quant(inner))
