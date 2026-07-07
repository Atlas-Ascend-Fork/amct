# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any
from loguru import logger
import torch


def get_n_set_parameters_byname(model, required_names):
    params = []
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return iter(params)


def check_params_grad(model):
    for name, param in model.named_parameters():
        logger.info(f"{name} : {param.requires_grad}")
    return


def set_require_grad_all(model, requires_grad):
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad
    return


def build_optimizer(args: Any, parameters):
    optimizer_name = getattr(args, "optimizer", "adamw").lower()
    base_lr = getattr(args, "base_lr", 1e-5)
    weight_decay = getattr(args, "weight_decay", 0.0)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=base_lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = getattr(args, "momentum", 0.9)
        return torch.optim.SGD(
            parameters,
            lr=base_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")


def build_lr_scheduler(args: Any, optimizer):
    scheduler_name = getattr(args, "lr_scheduler", "none").lower()

    if scheduler_name in {"none", ""}:
        return None

    if scheduler_name == "cosine":
        base_lr = getattr(args, "base_lr", 1e-5)
        nsamples = getattr(args, "nsamples", 128)
        cali_bsz = getattr(args, "cali_bsz", 1)
        epochs = getattr(args, "epochs", 20)
        t_max = epochs * (nsamples // cali_bsz)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=base_lr * 1e-3,
        )

    if scheduler_name == "step":
        step_size = max(int(getattr(args, "lr_step_size", 1)), 1)
        gamma = getattr(args, "lr_gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    raise ValueError(f"Unsupported lr scheduler '{scheduler_name}'.")
