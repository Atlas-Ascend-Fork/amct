# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
import torch
from amct_pytorch.utils.quant_util import quant_weight, apply_smooth_weight
from amct_pytorch.quantize_op.utils import apply_progressive_quant
from amct_pytorch.utils.vars import HIFLOAT8, FLOAT8_E4M3FN, INT8, FLOAT4_E2M1
from amct_pytorch.utils.check_params import check_parameters_in_schema


class NpuQuantizationLinear(torch.nn.Module):
    """
    Function: Customized torch.nn.Module of the npu quantized linear operator.
    APIs: forward
    """
    def __init__(self, quant_module):
        super().__init__()
        import torch_npu
        self.output_dtype = quant_module.weight.dtype
        device = quant_module.weight.device
        self.group_size = quant_module.group_size
        self.act_type = quant_module.act_type
        self.wts_type = quant_module.wts_type
        self._init_npu_quantize_type()
        self.act_granularity = 'PER_TENSOR'
        self.quantize_axis = -1
        scale_d_tensor = quant_module.scale_d.to(device=device)
        offset_d_tensor = quant_module.offset_d if quant_module.offset_d is not None else torch.tensor([0])
        offset_d_tensor = offset_d_tensor.repeat(scale_d_tensor.shape)
        if self.act_type == INT8 and scale_d_tensor.dim() > 1:
            self.act_granularity = 'PER_TOKEN'
            self.quantize_axis = -2
        self.register_buffer('act_scale', 
                             (1 / scale_d_tensor).reshape(-1).to(device=device).to(self.output_dtype))
        self.register_buffer('act_offset', 
                             offset_d_tensor.reshape(-1).to(device=device).to(self.output_dtype))

        weight = quant_module.weight.data
        self.scale_w_tensor = None
        if quant_module.scale is not None:
            weight = apply_smooth_weight(quant_module.scale, weight)
            self.register_buffer('scale_factor', (1 / quant_module.scale).to(device=device))
        weight_tensor = self._get_quantize_wts(quant_module, weight, device)
        self.register_buffer('quantized_weight', weight_tensor)
        self.offset_bias = None
        if quant_module.offset_d is not None:
            offset_reshape = quant_module.offset_d.float().repeat(1, quant_module.weight.shape[-1])
            self.offset_bias = offset_reshape.float() @ weight_tensor.float()
            self.offset_bias = -1 * self.offset_bias.round().to(torch.int32).reshape(-1)

        if self.act_granularity == 'PER_TOKEN':
            self.register_buffer('deq_scale', self.scale_w_tensor)
            self.register_buffer('pertoken_scale',
                quant_module.scale_d.reshape(-1).to(torch.float32).to(device=device))
            self.y_scale = None
            self.group_sizes = None
        elif self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
            self.register_buffer('deq_scale', self.scale_w2)
            self.pertoken_scale = None
            self.register_buffer('y_scale', self.y_scale_tensor)
            self.group_sizes = [0, 0, 32]
        else:
            deq_scale_tensor = quant_module.scale_d.to(device=device) * self.scale_w_tensor
            self.register_buffer('deq_scale', deq_scale_tensor)
            self.pertoken_scale = None
            self.y_scale = None
            self.group_sizes = None

        self._init_bias(quant_module, scale_d_tensor, self.scale_w_tensor)

    def forward(self, x):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        import torch_npu

        if hasattr(self, 'scale_factor'):
            x = x * self.scale_factor
        quant_x = torch_npu.npu_quantize(x, self.act_scale, self.act_offset, dtype=self.npu_quantize_act_type,
            axis=self.quantize_axis, div_mode=False)
        if self.is_new_torch_npu:
            if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                ori_shape = x.shape
                x = x.reshape(-1, x.shape[-1])
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
                bias=self.bias, output_dtype=self.output_dtype,
                x1_dtype=self.x1_dtype, x2_dtype=self.x2_dtype,
                group_sizes=self.group_sizes, y_scale=self.y_scale)
            if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                output = output.reshape(*ori_shape[:-1], -1)
        else:
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
                bias=self.bias, output_dtype=self.output_dtype)

        return output

    def _init_npu_quantize_type(self):
        import torch_npu
        self.is_new_torch_npu = False
        if check_parameters_in_schema(torch_npu.npu_quant_matmul, 'x1_dtype'):
            self.is_new_torch_npu = True
        self.x1_dtype = None
        self.x2_dtype = None
        if self.act_type == HIFLOAT8:
            self.x1_dtype = torch_npu.hifloat8
            self.npu_quantize_act_type = torch_npu.hifloat8
        elif self.act_type == FLOAT8_E4M3FN:
            self.npu_quantize_act_type = torch.float8_e4m3fn
        elif self.act_type == INT8:
            self.npu_quantize_act_type = torch.int8
        else:
            raise ValueError('Not supported dtype {}'.format(self.act_type))

        if self.wts_type == HIFLOAT8:
            self.x2_dtype = torch_npu.hifloat8
        elif self.wts_type == FLOAT4_E2M1:
            self.x2_dtype = torch_npu.float4_e2m1fn_x2

    def _init_bias(self, module, scale_d_tensor, scale_w_tensor):
        """
        Function: init bias for npu op
        Args:
            module: quant module 
            scale_d_tensor: torch.tensor, scale factor for activation
            scale_w_tensor: torch.tensor, scale factor for weight
        Returns:
            torch.tensor
        """
        if module.bias is not None:
            bias = module.bias
            if self.act_granularity == 'PER_TENSOR':
                bias_tensor = bias.data / (scale_d_tensor * scale_w_tensor)
                if self.act_type == INT8:
                    bias_tensor = bias.round().to(torch.int32)
                self.register_buffer('bias', bias_tensor)
            elif self.act_granularity == 'PER_TOKEN':
                self.register_buffer('bias', bias)

            if self.offset_bias is not None:
                self.bias = self.bias + self.offset_bias
        else:
            if self.offset_bias is None:
                self.bias = None 
            else:
                self.bias = self.offset_bias

    def _get_quantize_wts(self, quant_module, weight, device):
        """
        Function: get quantize weight
        Args:
            quant_module: quant module 
            weight: torch.tensor, weight of quant module
            device: 
        Returns:
            torch.tensor
        """
        import torch_npu
        if hasattr(quant_module, 'scale_w1') and quant_module.scale_w1 is not None:
            scale_w1 = quant_module.scale_w1.to(device=device)
            scale_w2 = quant_module.scale_w2.to(device=device)
            weight_tensor = \
                apply_progressive_quant(weight, scale_w1, scale_w2, self.group_size)
            self.scale_w2 = scale_w2.reshape(weight.shape[0], -1).to(torch.bfloat16)
            y_scale_tensor = quant_module.scale_d.to(device=device) * scale_w1
            self.y_scale_tensor = torch_npu.npu_trans_quant_param(y_scale_tensor).reshape(1, -1)
        else:
            self.scale_w_tensor = quant_module.scale_w.to(device=device)
            weight_tensor = quant_weight(weight, self.wts_type, self.scale_w_tensor, group_size=self.group_size)
            weight_tensor = torch.transpose(weight_tensor, 1, 0)
            self.scale_w_tensor = self.scale_w_tensor.reshape(-1)
        return weight_tensor