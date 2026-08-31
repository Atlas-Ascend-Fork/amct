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
from types import SimpleNamespace

import pytest
import torch

from amct_pytorch.algorithms.quant.base import QuantAlgorithmBase
from amct_pytorch.algorithms.quant.gptq import GPTQ
from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.quantization.dtypes.hifp_impl import hifloat4_fake_quant


def _args(quant_dtype="int", w_size=(4, 64)):
    return SimpleNamespace(quant_dtype=quant_dtype, w_size=w_size)


def _make(quant_dtype="int", w_bits=8, in_features=64, out_features=4):
    args = _args(quant_dtype=quant_dtype, w_size=(out_features, in_features))
    return GPTQ(args, w_bits)


def _observe(gptq, inputs):
    gptq.is_observe = True
    gptq.observe_input(inputs)


def _single_group_expected(gptq, weight):
    """Closed-form expected output of quantize() when columns == HIFX4_GROUP_SIZE: with
    exactly one block and one group, both the intra-block and inter-block error-propagation
    steps are no-ops (their target slices are empty), so the result reduces to a direct
    quantize of the dead-zeroed, Hessian-permuted weight."""
    diag = torch.diag(gptq.hessian)
    dead = diag == 0
    w = weight.clone()
    w[:, dead] = 0
    perm = torch.argsort(diag, descending=True)
    invperm = torch.argsort(perm)
    return hifloat4_fake_quant(w[:, perm], qdim=-1)[:, invperm].to(weight.dtype)


# ---- registration / dispatch ---------------------------------------------------


def test_gptq_is_registered():
    assert ALGO_REGISTRY.get("gptq") is GPTQ
    item = ALGO_REGISTRY.get_item("gptq")
    assert item.metadata["targets"] == ("weight",)


def test_gptq_is_a_quant_algorithm_base_with_no_trainable_params():
    gptq = _make()
    assert isinstance(gptq, QuantAlgorithmBase)
    assert gptq.trainable_params() == []


def test_gptq_resolves_wts_type_and_is_hif4_per_dtype():
    for quant_dtype, w_bits, expected_quant_type, expected_is_hif4 in (
        ("hifp", 4, "hifloat4", True),
        ("int", 4, "int4", False),
        ("int", 8, "int8", False),
    ):
        # 64 columns: valid for every dtype above, required for the hifp/hif4 case.
        gptq = _make(
            quant_dtype=quant_dtype, w_bits=w_bits, in_features=64, out_features=4
        )
        assert gptq.wts_type == expected_quant_type
        assert gptq.is_hif4 == expected_is_hif4


def test_init_raises_when_hif4_column_count_not_multiple_of_64():
    with pytest.raises(ValueError, match="multiple of 64"):
        _make(quant_dtype="hifp", w_bits=4, in_features=100, out_features=4)


def test_init_raises_on_non_2d_weight_shape():
    args = _args(w_size=(4,))
    with pytest.raises(ValueError, match="2D weight tensors"):
        GPTQ(args, 8)


def test_init_raises_for_unsupported_dtypes():
    # mxfp and hifloat8 (hifp w_bits=8) are not implemented yet -- deliberately deferred to
    # keep the first cut to int + HiFloat4.
    with pytest.raises(ValueError, match="unsupported quant_dtype"):
        _make(quant_dtype="mxfp", w_bits=4, in_features=64, out_features=4)
    with pytest.raises(ValueError, match="only HiFloat4"):
        _make(quant_dtype="hifp", w_bits=8, in_features=64, out_features=4)


def test_init_raises_for_unsupported_int_bit_width():
    # No longer a GPTQ-level check -- bit_policy.py's yaml schema validation (_ALLOWED_BITS)
    # already rejects anything outside (4, 8, 16) upstream; 16 is schema-valid but unsupported
    # here, so this now surfaces as a plain KeyError from the dict lookup.
    with pytest.raises(KeyError):
        _make(quant_dtype="int", w_bits=16, in_features=64, out_features=4)


# ---- observe_input: gated on self.is_observe, not the call site ----------------


def test_observe_input_is_a_noop_when_not_observing():
    gptq = _make(in_features=8, out_features=2)
    gptq.is_observe = False
    gptq.observe_input(torch.randn(3, 8))
    assert int(gptq.nsamples.item()) == 0
    assert torch.equal(gptq.hessian, torch.zeros_like(gptq.hessian))


def test_observe_input_accumulates_hessian_when_observing():
    # 3D (batch, seq, cin) input: nsamples counts the batch dimension (standard GPTQ
    # semantics -- number of calibration sequences, not number of token-rows). A bare 2D
    # (N, cin) input is instead treated as a *single* sequence of N tokens (nsamples += 1),
    # see test_observe_input_treats_2d_input_as_one_sequence below.
    torch.manual_seed(0)
    gptq = _make(in_features=8, out_features=2)
    _observe(gptq, torch.randn(5, 1, 8))
    assert int(gptq.nsamples.item()) == 5
    assert torch.isfinite(gptq.hessian).all()
    assert not torch.equal(gptq.hessian, torch.zeros_like(gptq.hessian))

    h_after_first = gptq.hessian.clone()
    _observe(gptq, torch.randn(5, 1, 8))
    assert int(gptq.nsamples.item()) == 10
    assert not torch.equal(gptq.hessian, h_after_first)


def test_observe_input_treats_2d_input_as_one_sequence():
    torch.manual_seed(0)
    gptq = _make(in_features=8, out_features=2)
    _observe(gptq, torch.randn(5, 8))
    assert int(gptq.nsamples.item()) == 1
    assert torch.isfinite(gptq.hessian).all()
    assert not torch.equal(gptq.hessian, torch.zeros_like(gptq.hessian))


def test_observe_input_toggled_off_stops_accumulating():
    torch.manual_seed(0)
    gptq = _make(in_features=8, out_features=2)
    _observe(gptq, torch.randn(5, 1, 8))
    nsamples_after_observe = int(gptq.nsamples.item())

    gptq.is_observe = False
    gptq.observe_input(torch.randn(5, 1, 8))
    assert int(gptq.nsamples.item()) == nsamples_after_observe


# ---- quantize(): fallback vs closed-form solve ----------------------------------


def test_quantize_falls_back_to_quant_obj_when_no_hessian_built():
    gptq = _make(in_features=8, out_features=2)
    weight = torch.randn(2, 8)
    calls = []

    def quant_obj(w):
        calls.append(w)
        return w * 0

    out = gptq.quantize(weight, quant_obj)
    assert calls == [weight]
    assert torch.equal(out, weight * 0)


def test_single_group_optimization_matches_direct_hifloat4_quant():
    torch.manual_seed(0)
    gptq = _make(
        quant_dtype="hifp", w_bits=4, in_features=GPTQ.HIFX4_GROUP_SIZE, out_features=6
    )
    weight = torch.randn(6, GPTQ.HIFX4_GROUP_SIZE)
    _observe(gptq, torch.randn(200, GPTQ.HIFX4_GROUP_SIZE))

    optimized = gptq.quantize(weight, quant_obj=None)
    expected = _single_group_expected(gptq, weight)

    assert torch.equal(optimized, expected)
    assert not torch.equal(optimized, weight)


def test_dead_hessian_column_is_zeroed_before_quantization():
    torch.manual_seed(0)
    gptq = _make(
        quant_dtype="hifp", w_bits=4, in_features=GPTQ.HIFX4_GROUP_SIZE, out_features=6
    )
    weight = torch.randn(6, GPTQ.HIFX4_GROUP_SIZE)
    inputs = torch.randn(200, GPTQ.HIFX4_GROUP_SIZE)
    inputs[:, 5] = 0  # zero-variance column -> its Hessian diagonal entry is 0 ("dead")
    _observe(gptq, inputs)

    optimized = gptq.quantize(weight, quant_obj=None)
    expected = _single_group_expected(gptq, weight)

    assert torch.equal(optimized, expected)
    assert torch.equal(optimized[:, 5], torch.zeros(6))


def test_optimization_handles_multiple_blocks():
    torch.manual_seed(0)
    # 192 columns: two blocks (128 + 64), the second block is a single HIFX4 group.
    # 256 columns: two full 128-column blocks (two HIFX4 groups each).
    for columns in (192, 256):
        gptq = _make(quant_dtype="hifp", w_bits=4, in_features=columns, out_features=4)
        weight = torch.randn(4, columns)
        _observe(gptq, torch.randn(300, columns))

        optimized = gptq.quantize(weight, quant_obj=None)

        assert optimized.shape == weight.shape
        assert torch.isfinite(optimized).all()
        assert not torch.equal(optimized, weight)


def test_quantize_optimizes_weight_for_int8():
    torch.manual_seed(0)
    gptq = _make(quant_dtype="int", w_bits=8, in_features=32, out_features=4)
    weight = torch.randn(4, 32)
    _observe(gptq, torch.randn(50, 32))

    optimized = gptq.quantize(weight, quant_obj=None)
    assert optimized.shape == weight.shape
    assert not torch.equal(optimized, weight)


# ---- export_ptq_params / load_ptq_params ----------------------------------------


def test_export_ptq_params_empty_when_no_samples_observed():
    gptq = _make(in_features=8, out_features=2)
    assert gptq.export_ptq_params() == {}


def test_export_then_load_ptq_params_round_trips_hessian():
    torch.manual_seed(0)
    source = _make(in_features=8, out_features=2)
    _observe(source, torch.randn(6, 1, 8))

    params = source.export_ptq_params()
    assert set(params) == {"H", "nsamples"}
    assert params["nsamples"] == 6

    target = _make(in_features=8, out_features=2)
    target.load_ptq_params(params)

    assert int(target.nsamples.item()) == 6
    assert torch.equal(target.hessian, source.hessian)


def test_load_ptq_params_is_a_noop_without_h_key():
    gptq = _make(in_features=8, out_features=2)
    gptq.load_ptq_params({"nsamples": 5})
    assert int(gptq.nsamples.item()) == 0
