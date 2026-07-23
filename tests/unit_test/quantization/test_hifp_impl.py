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
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from amct_pytorch.quantization.dtypes import hifp_impl


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
    with patch.dict(sys.modules, {"torch_npu": module}), patch.object(
        torch.Tensor, "npu", lambda self: self, create=True
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
    with patch.dict(sys.modules, {"torch_npu": module}), patch.object(
        torch.Tensor, "npu", lambda self: self, create=True
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
        out = hifp_impl._native_hifloat8_fake_quant(x)

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
        assert hifp_impl._load_amct_ops_cast() is None


def test_amct_ops_loader_returns_encode_and_decode():
    package = ModuleType("amct_ops")
    package.__path__ = []
    module = ModuleType("amct_ops.hifloat8_cast")
    module.encode_to_hifloat8 = lambda tensor: tensor
    module.decode_from_hifloat8 = lambda tensor, dtype: tensor.to(dtype)
    modules = {"amct_ops": package, "amct_ops.hifloat8_cast": module}
    with patch.dict(sys.modules, modules):
        encode, decode = hifp_impl._load_amct_ops_cast()

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
        out = hifp_impl._amct_ops_hifloat8_fake_quant(x, encode, decode)

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
        out = hifp_impl._amct_ops_hifloat8_fake_quant(x, encode, decode)

    assert encoded_dtypes == [torch.bfloat16]
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert out.device == x.device


def test_native_backend_is_preferred():
    x = torch.randn(2, 8, dtype=torch.float16)
    native_result = x + 1
    native_calls = []

    with patch.object(
        hifp_impl, "is_native_hifloat8_cast_available", return_value=True
    ), patch.object(
        hifp_impl,
        "_native_hifloat8_fake_quant",
        side_effect=lambda tensor: native_calls.append(tensor) or native_result,
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

    with patch.object(
        hifp_impl, "is_native_hifloat8_cast_available", return_value=True
    ), patch.object(
        hifp_impl, "_native_hifloat8_fake_quant", return_value=native_result
    ), patch.object(
        hifp_impl,
        "_load_amct_ops_cast",
        side_effect=AssertionError("amct_ops must not be loaded"),
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

    with patch.object(
        hifp_impl, "is_native_hifloat8_cast_available", return_value=False
    ), patch.object(
        hifp_impl,
        "_native_hifloat8_fake_quant",
        side_effect=AssertionError("native backend must not be called"),
    ), patch.object(
        hifp_impl, "_load_amct_ops_cast", return_value=(encode, decode)
    ), patch.object(
        hifp_impl,
        "_amct_ops_hifloat8_fake_quant",
        side_effect=fake_fallback,
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

    with patch.object(
        hifp_impl, "is_native_hifloat8_cast_available", return_value=True
    ), patch.object(
        hifp_impl, "_native_hifloat8_fake_quant", side_effect=fail_native
    ), patch.object(
        hifp_impl, "_load_amct_ops_cast", return_value=(object(), object())
    ), patch.object(
        hifp_impl,
        "_amct_ops_hifloat8_fake_quant",
        return_value=fallback_result,
    ):
        assert hifp_impl.hifloat8_fake_quant(x) is fallback_result


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_amct_ops_execution_error_raises_backend_requirement(error_type):
    x = torch.randn(2, 8)

    def fail_amct_ops(tensor, encode, decode):
        raise error_type("amct_ops failed")

    with patch.object(
        hifp_impl, "is_native_hifloat8_cast_available", return_value=False
    ), patch.object(
        hifp_impl, "_load_amct_ops_cast", return_value=(object(), object())
    ), patch.object(
        hifp_impl,
        "_amct_ops_hifloat8_fake_quant",
        side_effect=fail_amct_ops,
    ):
        with pytest.raises(RuntimeError) as error:
            hifp_impl.hifloat8_fake_quant(x)

    assert "native HiFloat8 cast" in str(error.value)
    assert "install amct_ops" in str(error.value)


def test_both_backends_unavailable_raises_backend_requirement():
    x = torch.randn(2, 8)

    with patch.object(
        hifp_impl, "is_native_hifloat8_cast_available", return_value=False
    ), patch.object(hifp_impl, "_load_amct_ops_cast", return_value=None):
        with pytest.raises(RuntimeError) as error:
            hifp_impl.hifloat8_fake_quant(x)

    assert "native HiFloat8 cast" in str(error.value)
    assert "install amct_ops" in str(error.value)


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_amct_ops_load_error_raises_backend_requirement(error_type):
    x = torch.randn(2, 8)

    with patch.object(
        hifp_impl, "is_native_hifloat8_cast_available", return_value=False
    ), patch.object(
        hifp_impl,
        "_load_amct_ops_cast",
        side_effect=error_type("failed to load amct_ops"),
    ):
        with pytest.raises(RuntimeError) as error:
            hifp_impl.hifloat8_fake_quant(x)

    assert "native HiFloat8 cast" in str(error.value)
    assert "install amct_ops" in str(error.value)


def test_unexpected_native_error_is_not_swallowed():
    x = torch.randn(2, 8)

    def fail_native(tensor):
        raise ValueError("invalid input")

    with patch.object(
        hifp_impl, "is_native_hifloat8_cast_available", return_value=True
    ), patch.object(
        hifp_impl, "_native_hifloat8_fake_quant", side_effect=fail_native
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

    with patch.object(
        hifp_impl, "is_native_hifloat8_cast_available", return_value=True
    ), patch.object(
        hifp_impl, "_native_hifloat8_fake_quant", side_effect=fail_native
    ), patch.object(
        hifp_impl, "_load_amct_ops_cast", return_value=(object(), object())
    ), patch.object(
        hifp_impl,
        "_amct_ops_hifloat8_fake_quant",
        side_effect=fail_amct_ops,
    ):
        with pytest.raises(RuntimeError) as error:
            hifp_impl.hifloat8_fake_quant(x)

    assert "native HiFloat8 cast" in str(error.value)
    assert "install amct_ops" in str(error.value)
    assert native_attempts == [x]
    assert fallback_attempts == [x]
