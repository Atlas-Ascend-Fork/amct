# coding=utf-8
# Adapted from
# https://github.com/ruikangliu/FlatQuant/blob/main/flatquant/function_utils.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2024 ruikangliu. Licensed under MIT.
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

from loguru import logger


def get_init_scale(w_smax, x_smax, alpha=0.5):
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-5)


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
