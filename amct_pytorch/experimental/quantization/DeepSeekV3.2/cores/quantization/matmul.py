# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

from cores.quantization.node import ActivationQuantizer


class QuantizedMatmul(nn.Module):
    def __init__(self, args, bits=8, lwc=False):
        super(QuantizedMatmul, self).__init__()
        self.args = args
        self.bits = bits
        self.weight_quantizer = ActivationQuantizer(bits=bits, sym=True, lac=lwc)
        self.lwc = lwc
        self.lwc_initial = False
        self._eval_mode = False

    def _ori_forward(self, act, weight):
        return torch.matmul(act, weight)

    def _train_forward(self, act, weight):
        # quantize weight
        weight = weight.transpose(-1, -2)
        weight = self.weight_quantizer(weight)
        weight = weight.transpose(-1, -2)

        output = torch.matmul(act, weight)
        return output

    def forward(self, act, weight):
        if self.bits == 16:
            return self._ori_forward(act, weight)
        if not self._eval_mode:
            return self._train_forward(act, weight)
        else:
            return self._eval_forward(act, weight)

    def _eval_forward(self, act, weight):
        output = torch.matmul(act, weight)
        return output

