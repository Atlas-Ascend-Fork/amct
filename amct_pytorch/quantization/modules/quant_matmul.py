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

from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer, build_algorithms_by_target


class QuantizedMatmul(nn.Module):
    def __init__(self, quant_args, l_bits=8, r_bits=8, left_dim=None, right_dim=None, transpose_right=True):
        super(QuantizedMatmul, self).__init__()
        self.quant_args = quant_args
        self.l_bits = l_bits
        self.r_bits = r_bits
        self.transpose_right = transpose_right
        self.enable_attn_cache = "attn-cache" in self.quant_args.quant_target
        self.left_transform = None
        self.right_transform = None
        self.l_node = ActivationQuantizer(quant_args, bits=self.l_bits)
        self.r_node = ActivationQuantizer(quant_args, bits=self.r_bits)
        self._init_structure_transforms(left_dim, right_dim)
        self.eval_mode = False

    def forward(self, left, right):
        if not self.enable_attn_cache:
            if self.transpose_right:
                right = right.transpose(-2, -1)
            return torch.matmul(left, right)
        if self.left_transform is not None:
            left = self.left_transform(left)
        if self.right_transform is not None:
            right = self.right_transform(right)
        left = self.l_node(left)
        right = self.r_node(right)
        if self.transpose_right:
            right = right.transpose(-2, -1)
        output = torch.matmul(left, right)
        return output

    def _init_structure_transforms(self, left_dim, right_dim):
        if not self.enable_attn_cache:
            return
        if left_dim is not None:
            ctx = AlgoBuildContext(matrix_size=128, dim_size=left_dim)
            self.left_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        if right_dim is not None:
            ctx = AlgoBuildContext(matrix_size=128, dim_size=right_dim)
            self.right_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
