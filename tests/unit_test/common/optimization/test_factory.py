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
import torch.nn as nn

from amct_pytorch.common.optimization import factory

# ---- get_n_set_parameters_byname / set_require_grad_all -----------------


def _named_param_model():
    m = nn.Module()
    m.encoder_weight = nn.Parameter(torch.zeros(3))
    m.encoder_bias = nn.Parameter(torch.zeros(3))
    m.decoder_weight = nn.Parameter(torch.zeros(3))
    return m


def test_get_n_set_parameters_byname_matches_substring_and_marks_grad():
    m = _named_param_model()
    factory.set_require_grad_all(m, requires_grad=False)
    selected = list(factory.get_n_set_parameters_byname(m, ["encoder"]))
    assert len(selected) == 2
    for p in selected:
        assert p.requires_grad is True
    # Decoder param remains frozen.
    assert m.decoder_weight.requires_grad is False


def test_get_n_set_parameters_byname_returns_iterator():
    m = _named_param_model()
    out = factory.get_n_set_parameters_byname(m, ["decoder"])
    assert iter(out) is out  # is an iterator, not a list


def test_set_require_grad_all_toggles_every_parameter():
    m = _named_param_model()
    factory.set_require_grad_all(m, requires_grad=False)
    assert all(not p.requires_grad for p in m.parameters())
    factory.set_require_grad_all(m, requires_grad=True)
    assert all(p.requires_grad for p in m.parameters())


def test_check_params_grad_returns_none():
    # Smoke test — only the side-effect logging matters here.
    m = _named_param_model()
    assert factory.check_params_grad(m) is None


# ---- build_optimizer -----------------------------------------------------


def _params():
    return [nn.Parameter(torch.zeros(2))]


def test_build_optimizer_default_is_adamw():
    args = SimpleNamespace()
    opt = factory.build_optimizer(args, _params())
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_adam_uses_args_lr_and_weight_decay():
    args = SimpleNamespace(optimizer="adam", base_lr=2e-4, weight_decay=0.1)
    opt = factory.build_optimizer(args, _params())
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == pytest.approx(2e-4)
    assert opt.defaults["weight_decay"] == pytest.approx(0.1)


def test_build_optimizer_sgd_uses_momentum_default_when_missing():
    args = SimpleNamespace(optimizer="sgd", base_lr=1e-3)
    opt = factory.build_optimizer(args, _params())
    assert isinstance(opt, torch.optim.SGD)
    assert opt.defaults["momentum"] == pytest.approx(0.9)


def test_build_optimizer_unknown_raises():
    args = SimpleNamespace(optimizer="lamb")
    with pytest.raises(ValueError, match="Unsupported optimizer 'lamb'"):
        factory.build_optimizer(args, _params())


# ---- build_lr_scheduler --------------------------------------------------


def _adam(params=None):
    return torch.optim.AdamW(params or _params())


@pytest.mark.parametrize("name", ["none", "", "NONE"])
def test_build_lr_scheduler_returns_none_for_disabled(name):
    args = SimpleNamespace(lr_scheduler=name)
    assert factory.build_lr_scheduler(args, _adam()) is None


def test_build_lr_scheduler_cosine_uses_args_for_t_max():
    args = SimpleNamespace(
        lr_scheduler="cosine",
        base_lr=1e-3,
        nsamples=64,
        cali_bsz=8,
        epochs=4,
    )
    sched = factory.build_lr_scheduler(args, _adam())
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 4 * (64 // 8)
    assert sched.eta_min == pytest.approx(1e-3 * 1e-3)


def test_build_lr_scheduler_step_uses_args_step_and_gamma():
    args = SimpleNamespace(lr_scheduler="step", lr_step_size=3, lr_gamma=0.5)
    sched = factory.build_lr_scheduler(args, _adam())
    assert isinstance(sched, torch.optim.lr_scheduler.StepLR)
    assert sched.step_size == 3
    assert sched.gamma == pytest.approx(0.5)


def test_build_lr_scheduler_step_clamps_zero_step_size_to_one():
    args = SimpleNamespace(lr_scheduler="step", lr_step_size=0)
    sched = factory.build_lr_scheduler(args, _adam())
    assert sched.step_size == 1


def test_build_lr_scheduler_unknown_raises():
    args = SimpleNamespace(lr_scheduler="warmup")
    with pytest.raises(ValueError, match="Unsupported lr scheduler 'warmup'"):
        factory.build_lr_scheduler(args, _adam())
