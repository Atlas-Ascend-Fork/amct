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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

sys.modules["torch_npu"] = MagicMock()

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.algorithms.quant.gptq import GPTQuant, _build_gptq_weights_cfg
from amct_pytorch.common.optimization import CalibrateBuildContext
from amct_pytorch.quantization.dtypes.hifp_impl import hifloat4_fake_quant


class _FakeQuantLinear:
    """Minimal duck-typed stand-in for QuantLinear -- GPTQuant.__init__ only ever reads
    .linear and .w_bits, and a real QuantLinear needs a lot of unrelated setup
    (WeightQuantizer, DTYPE_REGISTRY, build_algorithms_by_target)."""

    def __init__(self, linear, w_bits):
        self.linear = linear
        self.w_bits = w_bits


def _make(in_features=16, out_features=4, batch_num=1):
    linear = nn.Linear(in_features, out_features)
    # w_bits=4 -> hifloat4 (see _build_gptq_weights_cfg); these tests are all hifloat4-specific.
    quant_linear = _FakeQuantLinear(linear, w_bits=4)
    ctx = CalibrateBuildContext(args=SimpleNamespace(quant_dtype="hifp"), num_batches=batch_num)
    return GPTQuant(quant_linear, "layer", ctx), linear


def _single_group_expected(higptq, weight):
    """Closed-form expected output of get_opt_weight_and_quant_factor when
    columns == HIFX4_GROUP_SIZE: with exactly one block and one group, both the
    intra-block and inter-block error-propagation steps are no-ops (their target
    slices are empty), so the result reduces to a direct quantize of the
    dead-zeroed, Hessian-permuted weight."""
    diag = torch.diag(higptq.hessian)
    dead = diag == 0
    w = weight.clone()
    w[:, dead] = 0
    perm = torch.argsort(diag, descending=True)
    invperm = torch.argsort(perm)
    return hifloat4_fake_quant(w[:, perm], qdim=-1)[:, invperm].to(weight.dtype)


# ---- registration / dispatch ---------------------------------------------------


def test_gptq_and_higptq_alias_the_same_dispatcher():
    assert ALGO_REGISTRY.get("gptq") is GPTQuant
    assert ALGO_REGISTRY.get("higptq") is GPTQuant
    assert ALGO_REGISTRY.get_item("gptq").metadata["solver"] == "calibrate"
    assert ALGO_REGISTRY.get_item("higptq").metadata["solver"] == "calibrate"


def test_gptq_resolves_wts_type_and_is_hif4_per_dtype():
    for quant_dtype, w_bits, expected_quant_type, expected_is_hif4 in (
        ("hifp", 4, "hifloat4", True),
        ("hifp", 8, "hifloat8", False),  # hifloat8 is a plain per-column dtype, not hif4
        ("int", 4, "int4", False),
        ("int", 8, "int8", False),
        ("mxfp", 4, "mxfp4_e2m1", False),
        ("mxfp", 8, "mxfp8_e4m3fn", False),
    ):
        quant_linear = _FakeQuantLinear(nn.Linear(16, 4), w_bits=w_bits)
        ctx = CalibrateBuildContext(args=SimpleNamespace(quant_dtype=quant_dtype), num_batches=1)
        result = GPTQuant(quant_linear, "layer", ctx)
        assert result.wts_type == expected_quant_type
        assert result.is_hif4 == expected_is_hif4


# ---- _build_gptq_weights_cfg (per-layer dtype/bit-width -> weights_cfg) -----


def test_build_gptq_weights_cfg_selects_quant_type_by_dtype_and_bits():
    assert _build_gptq_weights_cfg("int", 4) == {
        "quant_type": "int4", "symmetric": True, "strategy": "channel"}
    assert _build_gptq_weights_cfg("int", 8) == {
        "quant_type": "int8", "symmetric": True, "strategy": "channel"}
    assert _build_gptq_weights_cfg("mxfp", 4) == {
        "quant_type": "mxfp4_e2m1", "symmetric": True, "strategy": "group", "group_size": 32}
    assert _build_gptq_weights_cfg("mxfp", 8) == {
        "quant_type": "mxfp8_e4m3fn", "symmetric": True, "strategy": "group", "group_size": 32}
    assert _build_gptq_weights_cfg("hifp", 4) == {
        "quant_type": "hifloat4", "symmetric": True, "strategy": "channel"}
    assert _build_gptq_weights_cfg("hifp", 8) == {
        "quant_type": "hifloat8", "symmetric": True, "strategy": "channel"}


def test_build_gptq_weights_cfg_rejects_unsupported_bit_width():
    with pytest.raises(ValueError, match="unsupported int w_bits"):
        _build_gptq_weights_cfg("int", 16)
    with pytest.raises(ValueError, match="unsupported mxfp w_bits"):
        _build_gptq_weights_cfg("mxfp", 16)
    with pytest.raises(ValueError, match="unsupported hif w_bits"):
        _build_gptq_weights_cfg("hifp", 16)


# ---- plain GPTQuant sanity (int path, exercised end-to-end through the state machine) ----


def test_plain_gptquant_forward_optimizes_weight_for_int8():
    torch.manual_seed(0)
    linear = nn.Linear(32, 4)
    quant_linear = _FakeQuantLinear(linear, w_bits=8)
    ctx = CalibrateBuildContext(args=SimpleNamespace(quant_dtype="int"), num_batches=2)
    gptq = GPTQuant(quant_linear, "layer", ctx)
    original_weight = linear.weight.detach().clone()

    gptq.forward(torch.randn(3, 32))
    assert torch.equal(linear.weight.detach(), original_weight)  # still calibrating

    gptq.forward(torch.randn(3, 32))
    assert not torch.equal(linear.weight.detach(), original_weight)  # optimized in place

    out = gptq.forward(torch.randn(5, 32))  # now on the fake_quant_forward path
    assert out.shape == (5, 4)


# ---- hif branch (get_opt_weight_and_quant_factor, fake_quant_forward) ---------


def test_cal_scale_offset_static_has_no_external_quant_factors():
    higptq, _ = _make()
    scale_w, offset_w = higptq.cal_scale_offset_static(higptq.weight)
    assert scale_w is None
    assert offset_w is None


def test_cal_quant_weight_group_delegates_to_hifloat4_fake_quant():
    higptq, _ = _make()
    for width in (17, 64):  # a partial group and a full HIFX4_GROUP_SIZE group
        w = torch.randn(higptq.weight.shape[0], width)
        assert torch.equal(higptq.cal_quant_weight_group(w), hifloat4_fake_quant(w, qdim=-1))


# ---- get_opt_weight_and_quant_factor -------------------------------------------


def test_single_group_optimization_matches_direct_hifloat4_quant():
    torch.manual_seed(0)
    higptq, _ = _make(in_features=GPTQuant.HIFX4_GROUP_SIZE, out_features=6)
    higptq.update_hessian(torch.randn(200, GPTQuant.HIFX4_GROUP_SIZE))

    hessian_before = higptq.hessian.clone()
    weight_before = higptq.weight.detach().clone()
    optimized, scale_w, offset_w = higptq.get_opt_weight_and_quant_factor()

    higptq.hessian = hessian_before  # get_opt_weight_and_quant_factor deletes self.hessian
    expected = _single_group_expected(higptq, weight_before)

    assert scale_w is None and offset_w is None
    assert torch.equal(optimized, expected)
    assert not torch.equal(optimized, weight_before)


def test_dead_hessian_column_is_zeroed_before_quantization():
    torch.manual_seed(0)
    higptq, _ = _make(in_features=GPTQuant.HIFX4_GROUP_SIZE, out_features=6)
    inputs = torch.randn(200, GPTQuant.HIFX4_GROUP_SIZE)
    inputs[:, 5] = 0  # zero-variance column -> its Hessian diagonal entry is 0 ("dead")
    higptq.update_hessian(inputs)

    hessian_before = higptq.hessian.clone()
    weight_before = higptq.weight.detach().clone()
    optimized, _, _ = higptq.get_opt_weight_and_quant_factor()

    higptq.hessian = hessian_before
    expected = _single_group_expected(higptq, weight_before)

    assert torch.equal(optimized, expected)
    assert torch.equal(optimized[:, 5], torch.zeros(6))


def test_optimization_handles_column_counts_not_aligned_to_group_or_block_size():
    torch.manual_seed(0)
    # 100 columns: one block (block_size=128), two HIFX4 groups (64 + 36).
    # 200 columns: two blocks (128 + 72), with the second block itself split into two groups.
    for columns in (100, 200):
        higptq, _ = _make(in_features=columns, out_features=4)
        higptq.update_hessian(torch.randn(300, columns))
        weight_before = higptq.weight.detach().clone()

        optimized, scale_w, offset_w = higptq.get_opt_weight_and_quant_factor()

        assert optimized.shape == weight_before.shape
        assert scale_w is None and offset_w is None
        assert torch.isfinite(optimized).all()
        assert not torch.equal(optimized, weight_before)


# ---- fake_quant_forward caching -------------------------------------------------


def test_fake_quant_forward_quantizes_once_and_reuses_cache():
    higptq, _ = _make(in_features=16, out_features=4)
    with patch(
        "amct_pytorch.algorithms.quant.gptq.hifloat4_fake_quant", wraps=hifloat4_fake_quant
    ) as mock_fq:
        out1 = higptq.fake_quant_forward(torch.randn(2, 16))
        out2 = higptq.fake_quant_forward(torch.randn(3, 16))

    assert mock_fq.call_count == 1
    assert higptq.fake_quant_cache_ready is True
    assert out1.shape == (2, 4)
    assert out2.shape == (3, 4)


# ---- forward() end-to-end (hif branch state machine) ---------------------------


def test_forward_updates_weight_after_batch_num_then_switches_to_fake_quant():
    higptq, linear = _make(in_features=32, out_features=4, batch_num=2)
    original_weight = linear.weight.detach().clone()

    out1 = higptq.forward(torch.randn(3, 32))
    assert higptq.cur_batch == 1
    assert torch.equal(linear.weight.detach(), original_weight)  # still calibrating

    out2 = higptq.forward(torch.randn(3, 32))
    assert higptq.cur_batch == 3
    assert not torch.equal(linear.weight.detach(), original_weight)  # optimized in place
    assert higptq.scale_w is None and higptq.offset_w is None

    out3 = higptq.forward(torch.randn(5, 32))  # now on the fake_quant_forward path
    assert out1.shape == (3, 4)
    assert out2.shape == (3, 4)
    assert out3.shape == (5, 4)
