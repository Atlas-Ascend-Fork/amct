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


from cores.quantization.node import WeightQuantizer


class QuantLinear(nn.Module):
    def __init__(self, args, linear: nn.Linear, w_bits=None, lwc=None):
        super(QuantLinear, self).__init__()
        self.args = args
        self.linear = linear
        self.w_bits = w_bits
        self.weight_quantizer = WeightQuantizer()
        if w_bits is None:
            self.weight_quantizer.configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
        else:
            self.weight_quantizer.configure(w_bits, perchannel=True, sym=not(args.w_asym), mse=False)

        self.lwc = lwc if lwc is not None else args.lwc
        if self.lwc:
            lwc_dim = self.linear.weight.shape[0] if self.lwc else -1
            init_value = 4.
            self.clip_factor_w_max = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.clip_factor_w_min = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.sigmoid = nn.Sigmoid()

        self._eval_mode = False

    def apply_wclip(self, weight):
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax *= self.sigmoid(self.clip_factor_w_max.to(weight.device))
        wmin *= self.sigmoid(self.clip_factor_w_min.to(weight.device))
        weight = torch.clamp(weight, min=wmin, max=wmax)
        return weight

    def _ori_forward(self, hidden_states):
        return self.linear(hidden_states)

    def _train_forward(self, hidden_states):
        weight = self.linear.weight.data
        # quantization-adaptive transform
        # learnable weight clipping
        if self.lwc:
            weight = self.apply_wclip(weight)

        # quantize weight
        self.weight_quantizer.find_params(weight)
        weight = self.weight_quantizer(weight)

        bias = self.linear.bias
        output = F.linear(hidden_states, weight, bias)
        return output

    def forward(self, hidden_states):
        if self.w_bits == 16:
            return self._ori_forward(hidden_states)
        if not self._eval_mode:
            return self._train_forward(hidden_states)
        else:
            return self._eval_forward(hidden_states)

    def _eval_forward(self, hidden_states):
        return F.linear(hidden_states, self.weight, self.bias)

