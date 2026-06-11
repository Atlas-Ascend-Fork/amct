# coding=utf-8
# Adapted from
# https://github.com/ruikangliu/FlatQuant/blob/main/flatquant/quant_utils.py
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

from inspect import Parameter, signature

import torch
import torch.nn as nn
from loguru import logger

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY


def get_algo_names_by_target(args, target):
    selected = []
    for algo_name in args.algos:
        algo_item = ALGO_REGISTRY.get_item(algo_name)
        targets = tuple(algo_item.metadata.get("targets", ()))
        if not targets:
            raise ValueError(
                f"Algorithm '{algo_name}' is missing registry metadata 'targets'."
            )
        if target in targets:
            selected.append(algo_name)
    return selected


def build_algorithms_by_target(args, target, *ctor_args):
    algorithms = nn.ModuleDict()
    algo_names = get_algo_names_by_target(args, target)
    for algo_name in algo_names:
        algo_item = ALGO_REGISTRY.get_item(algo_name)
        targets = tuple(algo_item.metadata.get("targets", ()))
        if not targets:
            raise ValueError(
                f"Algorithm '{algo_name}' is missing registry metadata 'targets'."
            )
        if target not in targets:
            raise ValueError(
                f"Algorithm '{algo_name}' cannot be used for target '{target}'. "
                f"Declared targets: {targets}"
            )
        algorithms[algo_name] = _build_algorithm(algo_item.target, args, *ctor_args)
    if target == "structure":
        if len(algorithms) == 0:
            return None
        if len(algorithms) > 1:
            raise ValueError(
                f"Only one '{target}' algorithm is supported here, got: {list(algorithms.keys())}"
            )
        return next(iter(algorithms.values()))
    else:
        return algorithms


def _build_algorithm(algo_cls, args, *ctor_args):
    init_params = list(signature(algo_cls.__init__).parameters.values())
    positional_params = [
        param for param in init_params[1:]
        if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(param.kind == Parameter.VAR_POSITIONAL for param in init_params)
    if has_varargs or len(positional_params) > 1:
        return algo_cls(args, *ctor_args)
    return algo_cls(args)


def set_quantizer_state(model, enable=True):
    for m in model.modules():
        if isinstance(m, (WeightQuantizer, ActivationQuantizer)):
            m.enable = enable
    return model


def set_weight_quantizer_state(model, enable=True):
    for m in model.modules():
        if isinstance(m, WeightQuantizer):
            m.enable = enable
    return model


def set_act_quantizer_state(model, enable=True):
    for m in model.modules():
        if isinstance(m, ActivationQuantizer):
            m.enable = enable
    return model


class ActivationQuantizer(torch.nn.Module):

    def __init__(self, args, bits):
        super(ActivationQuantizer, self).__init__()
        self.args = args
        self.bits = bits
        self.algorithms = nn.ModuleDict()
        self._init_algo()
        self.quant_obj = DTYPE_REGISTRY.get(args.quant_dtype)(bits=self.bits, is_act=True)
        self.enable = False

    def deploy(self):
        pass

    def fake_quant(self, x):
        return self.quant_obj(x)

    def forward(self, x):
        if not self.enable:
            return x
        for algo in self.algorithms.values():
            x = algo(x)
        return self.fake_quant(x)

    def load_deploy(self, scale, zero):
        pass

    def trainable_params(self):
        params = []
        for algo in self.algorithms.values():
            if hasattr(algo, "trainable_params"):
                params.extend(algo.trainable_params())
        return params

    def _init_algo(self):
        self.algorithms = build_algorithms_by_target(self.args, "activation")


class WeightQuantizer(torch.nn.Module):

    def __init__(self, args, w_bits=None):
        super(WeightQuantizer, self).__init__()
        self.args = args
        self.bits = w_bits if w_bits is not None else args.w_bits
        self.algorithms = nn.ModuleDict()
        self._init_algo()
        self.quant_obj = DTYPE_REGISTRY.get(args.quant_dtype)(bits=self.bits)
        self.enable = False

    def algo_forward(self, x):
        quantize_algo = None
        for algo in self.algorithms.values():
            quantize_fn = getattr(algo, "quantize", None)
            if callable(quantize_fn):
                if quantize_algo is not None:
                    raise ValueError("Only one weight algorithm with a custom quantize() hook is supported.")
                quantize_algo = algo
                continue
            x = algo(x)
        return x, quantize_algo

    def export_deploy(self, x):
        x, quantize_algo = self.algo_forward(x)
        if quantize_algo is not None:
            export_fn = getattr(quantize_algo, "export_deploy", None)
            if not callable(export_fn):
                raise NotImplementedError("export_deploy() does not support custom weight quantize() hooks yet.")
            return export_fn(x, self.quant_obj)
        export_fn = getattr(self.quant_obj, "export_deploy", None)
        if not callable(export_fn):
            raise NotImplementedError(
                f"Quant dtype '{self.args.quant_dtype}' does not implement export_deploy()."
            )
        return export_fn(x)

    def fake_quant(self, x):
        return self.quant_obj(x)

    def forward(self, x):
        if not self.enable:
            return x
        x, quantize_algo = self.algo_forward(x)
        if quantize_algo is not None:
            return quantize_algo.quantize(x, self.quant_obj)
        return self.fake_quant(x)

    def observe_input(self, x, weight=None):
        for algo in self.algorithms.values():
            observe_input = getattr(algo, "observe_input", None)
            if callable(observe_input):
                observe_input(x, weight)

    def trainable_params(self):
        params = []
        for algo in self.algorithms.values():
            if hasattr(algo, "trainable_params"):
                params.extend(algo.trainable_params())
        return params

    def _init_algo(self):
        self.algorithms = build_algorithms_by_target(self.args, "weight", self.bits)
