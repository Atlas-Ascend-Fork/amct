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

from amct_pytorch.algorithms.quant.auto_clip import LAC, LWC

# ---- LWC -----------------------------------------------------------------


def _lwc_args(w_size=(4, 8), quant_dtype="int"):
    return SimpleNamespace(w_size=w_size, quant_dtype=quant_dtype)


def test_lwc_clip_dim_for_int_dtype_uses_first_dim():
    lwc = LWC(_lwc_args(w_size=(4, 8), quant_dtype="int"), w_bits=8)
    assert lwc.clip_dim == 4
    assert lwc.clip_factor_min.shape == (4, 1)
    assert lwc.clip_factor_max.shape == (4, 1)


def test_lwc_clip_dim_for_mxfp_uses_w_size_product_over_32():
    lwc = LWC(_lwc_args(w_size=(8, 64), quant_dtype="mxfp"), w_bits=8)
    assert lwc.clip_dim == 8 * 64 // 32


def test_lwc_init_value_equals_4():
    lwc = LWC(_lwc_args(), w_bits=8)
    assert torch.allclose(lwc.clip_factor_min.data, torch.full_like(lwc.clip_factor_min.data, 4.0))
    assert torch.allclose(lwc.clip_factor_max.data, torch.full_like(lwc.clip_factor_max.data, 4.0))


def test_lwc_trainable_params_returns_both_clip_factors():
    lwc = LWC(_lwc_args(), w_bits=8)
    params = lwc.trainable_params()
    # Identity check — `in` would invoke tensor.__eq__ and raise.
    assert any(p is lwc.clip_factor_min for p in params)
    assert any(p is lwc.clip_factor_max for p in params)


def test_lwc_export_load_round_trip():
    lwc = LWC(_lwc_args(), w_bits=8)
    with torch.no_grad():
        lwc.clip_factor_min.fill_(1.5)
        lwc.clip_factor_max.fill_(2.5)
    params = lwc.export_ptq_params()
    assert torch.equal(params["clip_factor_min"], lwc.clip_factor_min.detach().cpu())
    assert torch.equal(params["clip_factor_max"], lwc.clip_factor_max.detach().cpu())

    other = LWC(_lwc_args(), w_bits=8)
    other.load_ptq_params(params)
    assert torch.equal(other.clip_factor_min.data, lwc.clip_factor_min.data)
    assert torch.equal(other.clip_factor_max.data, lwc.clip_factor_max.data)


def test_lwc_apply_clip_with_zero_factors_clamps_to_half_amplitude():
    # sigmoid(0) = 0.5 -> max clipped to 0.5 * row_max, min clipped to 0.5 * row_min.
    lwc = LWC(_lwc_args(w_size=(2, 4), quant_dtype="int"), w_bits=8)
    with torch.no_grad():
        lwc.clip_factor_min.zero_()
        lwc.clip_factor_max.zero_()
    x = torch.tensor([[-2.0, -1.0, 1.0, 2.0], [-4.0, 0.0, 0.0, 4.0]])
    y = lwc(x)
    # Row 0: max=2, min=-2 -> clamp to [-1, 1]; row 1: [-2, 2].
    assert torch.equal(y[0], torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    assert torch.equal(y[1], torch.tensor([-2.0, 0.0, 0.0, 2.0]))


def test_lwc_forward_preserves_shape_for_mxfp():
    lwc = LWC(_lwc_args(w_size=(2, 32), quant_dtype="mxfp"), w_bits=8)
    x = torch.randn(2, 32)
    assert lwc(x).shape == x.shape


# ---- LAC -----------------------------------------------------------------


def _lac_args(is_per_tensor=False):
    return SimpleNamespace(is_per_tensor=is_per_tensor)


def test_lac_observe_mode_updates_min_max_buffers():
    lac = LAC(_lac_args())
    lac.is_observe = True
    x1 = torch.tensor([[-3.0, 5.0]])
    x2 = torch.tensor([[-1.0, 7.0]])
    out1 = lac(x1)
    out2 = lac(x2)
    assert torch.equal(out1, x1)
    assert torch.equal(out2, x2)
    assert lac.maxval.item() == 7.0
    assert lac.minval.item() == -3.0


def test_lac_clip_per_tensor_uses_observed_buffers():
    lac = LAC(_lac_args(is_per_tensor=True))
    with torch.no_grad():
        lac.maxval.fill_(2.0)
        lac.minval.fill_(-2.0)
        lac.clip_factor_min.zero_()
        lac.clip_factor_max.zero_()
    # sigmoid(0)=0.5 -> clip range [-1, 1]
    x = torch.tensor([[-3.0, 0.5, 1.5, 3.0]])
    y = lac(x)
    assert torch.equal(y, torch.tensor([[-1.0, 0.5, 1.0, 1.0]]))


def test_lac_clip_per_token_path_preserves_shape():
    lac = LAC(_lac_args(is_per_tensor=False))
    with torch.no_grad():
        lac.clip_factor_min.zero_()
        lac.clip_factor_max.zero_()
    x = torch.randn(2, 3, 4)
    y = lac(x)
    assert y.shape == x.shape


def test_lac_export_load_round_trip_includes_buffers():
    lac = LAC(_lac_args())
    with torch.no_grad():
        lac.clip_factor_min.fill_(2.0)
        lac.clip_factor_max.fill_(3.0)
        lac.maxval.fill_(10.0)
        lac.minval.fill_(-9.0)
    params = lac.export_ptq_params()
    assert set(params) == {"clip_factor_min", "clip_factor_max", "maxval", "minval"}

    other = LAC(_lac_args())
    other.load_ptq_params(params)
    assert other.clip_factor_min.item() == 2.0
    assert other.clip_factor_max.item() == 3.0
    assert other.maxval.item() == 10.0
    assert other.minval.item() == -9.0


def test_lac_trainable_params_returns_both_clip_factors():
    lac = LAC(_lac_args())
    params = lac.trainable_params()
    assert any(p is lac.clip_factor_min for p in params)
    assert any(p is lac.clip_factor_max for p in params)
