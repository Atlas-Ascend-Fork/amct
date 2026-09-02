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
import copy

import pytest
import torch

from amct_pytorch.common.optimization.sign_sgd import SignSGD, required


def _param(value=1.0, grad=None):
    p = torch.nn.Parameter(torch.full((2,), value))
    if grad is not None:
        p.grad = torch.full((2,), grad)
    return p


# ---- _RequiredParameter ----------------------------------------------------


def test_required_parameter_repr():
    assert repr(required) == "<required parameter>"


# ---- __init__ validation ---------------------------------------------------


def test_init_negative_lr_raises():
    with pytest.raises(ValueError, match="Invalid learning rate"):
        SignSGD([_param()], lr=-0.1)


def test_init_negative_momentum_raises():
    with pytest.raises(ValueError, match="Invalid momentum value"):
        SignSGD([_param()], lr=0.1, momentum=-1.0)


def test_init_negative_weight_decay_raises():
    with pytest.raises(ValueError, match="Invalid weight_decay value"):
        SignSGD([_param()], lr=0.1, weight_decay=-1.0)


def test_init_nesterov_without_momentum_raises():
    with pytest.raises(ValueError, match="Nesterov momentum requires"):
        SignSGD([_param()], lr=0.1, momentum=0, nesterov=True)


def test_init_nesterov_with_dampening_raises():
    with pytest.raises(ValueError, match="Nesterov momentum requires"):
        SignSGD([_param()], lr=0.1, momentum=0.9, dampening=0.1, nesterov=True)


def test_init_default_lr_is_required_sentinel():
    opt = SignSGD([_param()])
    assert opt.defaults["lr"] is required


# ---- __setstate__ -----------------------------------------------------------


def test_setstate_restores_missing_defaults():
    p = _param()
    opt = SignSGD([p], lr=0.1)
    for key in ("nesterov", "maximize", "foreach", "differentiable"):
        del opt.param_groups[0][key]

    restored = copy.deepcopy(opt)

    group = restored.param_groups[0]
    assert group["nesterov"] is False
    assert group["maximize"] is False
    assert group["foreach"] is None
    assert group["differentiable"] is False


# ---- step() / _sign_sgd() ---------------------------------------------------


def test_step_no_grad_params_are_noop():
    p = _param()
    opt = SignSGD([p], lr=0.1)

    loss = opt.step()

    assert loss is None
    assert torch.equal(p, torch.full((2,), 1.0))


def test_step_updates_param_by_sign_of_gradient():
    p = _param(value=1.0, grad=5.0)
    opt = SignSGD([p], lr=0.1)

    opt.step()

    assert torch.allclose(p, torch.full((2,), 0.9))


def test_step_negative_gradient_direction():
    p = _param(value=1.0, grad=-5.0)
    opt = SignSGD([p], lr=0.1)

    opt.step()

    assert torch.allclose(p, torch.full((2,), 1.1))


def test_step_uses_closure_for_loss():
    p = _param(value=1.0, grad=1.0)
    opt = SignSGD([p], lr=0.1)
    calls = []

    def closure():
        calls.append(1)
        return torch.tensor(42.0)

    loss = opt.step(closure)

    assert loss.item() == pytest.approx(42.0)
    assert len(calls) == 1


def test_step_maximize_flips_direction():
    p = _param(value=1.0, grad=5.0)
    opt = SignSGD([p], lr=0.1, maximize=True)

    opt.step()

    assert torch.allclose(p, torch.full((2,), 1.1))


def test_step_weight_decay_can_flip_sign():
    p = _param(value=1.0, grad=-0.01)
    opt = SignSGD([p], lr=0.1, weight_decay=1.0)

    # d_p = grad + weight_decay * param = -0.01 + 1.0 * 1.0 = 0.99 > 0
    opt.step()

    assert torch.allclose(p, torch.full((2,), 0.9))


def test_step_momentum_creates_and_reuses_buffer():
    p = _param(value=1.0, grad=1.0)
    opt = SignSGD([p], lr=0.1, momentum=0.9)

    opt.step()
    buf = opt.state[p]["momentum_buffer"]
    assert torch.allclose(buf, torch.full((2,), 1.0))
    assert torch.allclose(p, torch.full((2,), 0.9))

    p.grad = torch.full((2,), -1.0)
    opt.step()
    buf = opt.state[p]["momentum_buffer"]
    # buf = momentum * buf + (1 - dampening) * d_p = 0.9*1.0 + 1.0*(-1.0) = -0.1
    assert torch.allclose(buf, torch.full((2,), -0.1))
    assert torch.allclose(p, torch.full((2,), 1.0))


def test_step_nesterov_uses_lookahead_buffer():
    p = _param(value=1.0, grad=1.0)
    opt = SignSGD([p], lr=0.1, momentum=0.9, nesterov=True)

    # buf = clone(d_p) = 1.0; d_p = d_p + momentum*buf = 1.9 -> sign = 1
    opt.step()

    assert torch.allclose(p, torch.full((2,), 0.9))
