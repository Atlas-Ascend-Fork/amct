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
'''Shared activation quantization modules.'''

import torch
from torch import nn


class SharedQuantize(nn.Module):
    '''Execute one activation quantization for a shared-input group.'''

    def __init__(self, activation_scale, activation_offset=None, act_type='int8'):
        super().__init__()
        self.act_type = act_type
        self.register_buffer(
            'act_scale',
            (1 / activation_scale).reshape(-1).to(torch.float32),
        )
        offset = (
            activation_offset
            if activation_offset is not None
            else torch.zeros_like(activation_scale)
        )
        self.register_buffer('act_offset', offset.reshape(-1).to(torch.float32))

    def forward(self, x):
        import torch_npu

        return torch_npu.npu_quantize(
            x,
            self.act_scale,
            self.act_offset,
            dtype=self.act_type,
            axis=-1,
            div_mode=False,
        )


class FusedPreOp(nn.Module):
    '''Wrap the original pre_op with one shared activation quantization.'''

    def __init__(self, pre_op, shared_quantize):
        super().__init__()
        self.pre_op = pre_op
        self.shared = shared_quantize

    def forward(self, *args, **kwargs):
        return self.shared(self.pre_op(*args, **kwargs))


class SharedDynamicQuantize(nn.Module):
    '''Execute one dynamic per-token quantization for a shared-input group.'''

    def __init__(self, act_type='int8'):
        super().__init__()
        self.act_type = act_type

    def forward(self, x):
        import torch_npu

        return torch_npu.npu_dynamic_quant(x, dst_type=self.act_type)
