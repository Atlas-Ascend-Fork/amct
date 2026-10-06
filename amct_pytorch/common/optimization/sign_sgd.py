# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# Adapted from Intel auto-round v0.13.1:
#     https://github.com/intel/auto-round/blob/v0.13.1/auto_round/algorithms/quantization/sign_round/sign_sgd.py
#     Copyright (c) 2023 Intel Corporation -- Apache-2.0
#
# The upstream auto-round file is itself derived from PyTorch's
# torch/optim/sgd.py (Optimizer subclass, step(), momentum / weight-decay /
# nesterov handling, _RequiredParameter, _use_grad_for_differentiable):
#     Copyright (c) 2016 Facebook, Inc. (Adam Paszke) and affiliates -- BSD-3-Clause
#
# Full Apache-2.0 and BSD-3-Clause license texts are provided in
# Third_Party_Open_Source_Software_Notice at the repository root.
# ----------------------------------------------------------------------------

from typing import List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["SignSGD"]


class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults["differentiable"])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret

    return _use_grad


class SignSGD(Optimizer):
    """SGD variant that steps by the *sign* of the (optionally momentum- and
    weight-decay-adjusted) gradient instead of the gradient itself:
    ``param -= lr * sign(d_p)``. Used by AutoRound-style algorithms (e.g.
    HiAutoRound's k/v compensation) where the learned offsets only need a
    consistent step direction, not a magnitude proportional to gradient size.

    Args:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the params based on the
            objective, instead of minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of the
            optimizer is used (default: None)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize=False,
        foreach: Optional[bool] = None,
        differentiable=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the
                model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            _sign_sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def _sign_sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    """Functional API that performs the SignSGD computation for one step."""
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(torch.sign(d_p), alpha=-lr)
