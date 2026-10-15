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

    def __init__(
        self,
        activation_scale,
        activation_offset=None,
        act_type='int8',
        output_dtype=torch.float16,
        act_granularity='tensor',
    ):
        super().__init__()
        self.act_type = act_type
        self.output_dtype = output_dtype
        # Mirror NpuQuantizationLinear._init_act_quant_params: act_scale is
        # 1/scale_d cast to the deploy output dtype, offset is repeated to
        # the scale shape, and token input quant uses axis=-2.
        scale_d = activation_scale.reshape(-1)
        self.register_buffer(
            'act_scale',
            (1 / scale_d).to(output_dtype),
        )
        offset_d = (
            activation_offset
            if activation_offset is not None
            else torch.zeros(1, device=scale_d.device, dtype=scale_d.dtype)
        )
        self.register_buffer(
            'act_offset',
            offset_d.reshape(-1).expand_as(scale_d).clone().to(output_dtype),
        )
        self.quantize_axis = -2 if act_granularity == 'token' else -1

    def forward(self, x):
        import torch_npu

        if self.act_type == 'hifloat8':
            dtype = torch_npu.hifloat8
        elif self.act_type == 'float8_e4m3fn':
            dtype = torch.float8_e4m3fn
        elif self.act_type == 'int4':
            dtype = torch_npu.int4
        elif self.act_type == 'int8':
            dtype = torch.int8
        else:
            raise ValueError(f'Unsupported activation dtype: {self.act_type}')
        return torch_npu.npu_quantize(
            x,
            self.act_scale,
            self.act_offset,
            dtype=dtype,
            axis=self.quantize_axis,
            div_mode=False,
        )


class FusedPreOp(nn.Module):
    '''Wrap the original pre_op with one shared activation quantization.'''

    def __init__(self, pre_op, shared_quantize):
        super().__init__()
        self.pre_op = pre_op
        self.shared = shared_quantize

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Accept a legacy pre-op path for already-fused deployment tensors.
        # Shared quantization buffers are still required by strict loading.
        for name in self.pre_op.state_dict():
            old_key = prefix + name
            new_key = prefix + 'pre_op.' + name
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, *args, **kwargs):
        return self.shared(self.pre_op(*args, **kwargs))


class SharedDynamicQuantize(nn.Module):
    '''Execute one dynamic per-token quantization for a shared-input group.'''

    def __init__(self, act_type='int8'):
        super().__init__()
        self.act_type = act_type

    def forward(self, x):
        import torch_npu

        if self.act_type == 'hifloat8':
            return torch_npu.npu_dynamic_quant(
                x, dst_type=torch_npu.hifloat8, dst_type_max=15
            )
        if self.act_type == 'int8':
            return torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        raise ValueError(f'Unsupported activation dtype: {self.act_type}')
