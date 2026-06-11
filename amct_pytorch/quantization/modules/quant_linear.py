# coding=utf-8
# Adapted from
# https://github.com/ruikangliu/FlatQuant/blob/main/flatquant/flat_linear.py
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

import torch
import torch.nn as nn
import torch.nn.functional as F


from amct_pytorch.quantization.modules.quant_base import WeightQuantizer


class QuantLinear(nn.Module):
    def __init__(self, args, linear: nn.Linear, w_bits: int = None, name: str = None):
        super(QuantLinear, self).__init__()
        self.args = args
        self.name = name
        self.linear = linear
        self.args.w_size = self.linear.weight.data.shape
        self.w_bits = w_bits if w_bits is not None else args.w_bits
        self.weight_quantizer = WeightQuantizer(self.args, w_bits=self.w_bits)
        self.eval_mode = False
        self.cached_eval_weight = None
        self._cached_transform_key = None

    def forward(
        self,
        hidden_states,
        structure_transform=None,
    ):
        if self.eval_mode:
            transform_key = None if structure_transform is None else id(structure_transform)
            if self.cached_eval_weight is None or self._cached_transform_key != transform_key:
                weight = self.linear.weight
                if structure_transform is not None:
                    weight = structure_transform(weight, inv_t=True, name=self.name)
                self.cached_eval_weight = self.weight_quantizer(weight).detach()
                self._cached_transform_key = transform_key
            weight = self.cached_eval_weight
        else:
            weight = self.linear.weight
            if structure_transform is not None:
                weight = structure_transform(weight, inv_t=True, name=self.name)
            self.weight_quantizer.observe_input(hidden_states, weight)
            weight = self.weight_quantizer(weight)
        bias = self.linear.bias
        output = F.linear(hidden_states, weight, bias)
        return output

    def export_deploy(self, structure_transform=None):
        weight = self.linear.weight
        if structure_transform is not None:
            weight = structure_transform(weight, inv_t=True, name=self.name)
        payload = self.weight_quantizer.export_deploy(weight)
        return payload
