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
from amct_pytorch.common.utils.quant_util import quant_weight, apply_smooth_weight
from amct_pytorch.classic.quantize_op.utils import apply_progressive_quant
from amct_pytorch.common.utils.vars import HIFLOAT8, FLOAT8_E4M3FN, INT8, INT4, FLOAT4_E2M1, INT32_MAX, INT32_MIN
from amct_pytorch.common.utils.check_params import check_parameters_in_schema


class NpuQuantizationLinear(torch.nn.Module):
    """
    Function: Customized torch.nn.Module of the npu quantized linear operator.
    APIs: forward
    """
    def __init__(self, quant_module):
        """
        Function: Initialize NPU quantized linear operator
        Args:
            quant_module: quantization module containing quantization parameters
        """
        super().__init__()
        import torch_npu
        self.output_dtype = quant_module.weight.dtype
        device = quant_module.weight.device
        self.group_size = quant_module.group_size
        self.act_type = quant_module.act_type
        self.wts_type = quant_module.wts_type
        self._init_npu_quantize_type()
        self.act_granularity = getattr(quant_module, 'act_granularity', 'tensor')
        self.quantize_axis = -1
        self.pertoken_scale = None
        self.y_scale = None
        self.group_sizes = None
        self.dynamic = getattr(quant_module, 'dynamic', None)
        self._init_weight_quant(quant_module, device)
        self._init_deq_weight_scale(quant_module, device)
        self._init_bias(quant_module)
        if self.dynamic is None or self.dynamic is False:
            self._init_act_quant_params(quant_module, device)
            self._init_deq_act_scale(quant_module, device)

    def forward(self, x):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        import torch_npu

        if hasattr(self, 'scale_factor'):
            x = x * self.scale_factor
        
        if self.act_granularity == 'token':
            # per-token activation quant is inherently dynamic: the token dim equals
            # batch * seqlen and changes per input, so the scale must be computed at
            # runtime. HIF8 needs dst_type_max=15; INT8 uses the default range.
            if self.act_type == HIFLOAT8:
                quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=self.npu_quantize_act_type,
                    dst_type_max=15)
            else:
                quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=self.npu_quantize_act_type)
            self.pertoken_scale = pertoken_scale.reshape(-1).to(torch.float32)
        else:
            quant_x = torch_npu.npu_quantize(x, self.act_scale, self.act_offset, dtype=self.npu_quantize_act_type,
                axis=self.quantize_axis, div_mode=False)
        if self.is_new_torch_npu:
            if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                ori_shape = quant_x.shape
                quant_x = quant_x.reshape(-1, x.shape[-1])
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

    def _init_act_quant_params(self, quant_module, device):
        """
        Function: Initialize activation quantization parameters
        Args:
            quant_module: quantization module containing quantization parameters
            device: device to place tensors on
        Returns:
            scale_d_tensor: activation scale tensor
        """
        scale_d_tensor = quant_module.scale_d.to(device=device)
        offset_d_tensor = quant_module.offset_d if quant_module.offset_d is not None else torch.tensor([0])
        offset_d_tensor = offset_d_tensor.repeat(scale_d_tensor.shape)
        if self.act_granularity == 'token':
            self.quantize_axis = -2
        self.register_buffer('act_scale',
                             (1 / scale_d_tensor).reshape(-1).to(device=device).to(self.output_dtype))
        self.register_buffer('act_offset',
                             offset_d_tensor.reshape(-1).to(device=device).to(self.output_dtype))

    def _init_weight_quant(self, quant_module, device):
        """
        Function: Initialize weight quantization
        Args:
            quant_module: quantization module containing quantization parameters
            device: device to place tensors on
        """
        weight = quant_module.weight.data
        self.scale_w_tensor = None
        if quant_module.scale is not None:
            weight = apply_smooth_weight(quant_module.scale, weight)
            self.register_buffer('scale_factor', (1 / quant_module.scale).to(device=device))
        weight_tensor = self._get_quantize_wts(quant_module, weight, device)
        if self.wts_type in [HIFLOAT8, INT8, INT4, FLOAT8_E4M3FN]:
            self.register_buffer('quantized_weight', weight_tensor.contiguous())
        else:
            self.register_buffer('quantized_weight', weight_tensor)
        self.offset_bias = None
        if quant_module.offset_d is not None:
            # offset_bias is a per-cout correction, so it must use the real int4 weight
            # values [cin, cout] rather than the int8-packed bytes [cin, cout // 2].
            offset_weight = getattr(self, 'weight_before_pack', weight_tensor)
            offset_reshape = quant_module.offset_d.float().repeat(1, quant_module.weight.shape[-1])
            self.offset_bias = offset_reshape.float() @ offset_weight.float()
            self.offset_bias = -1 * self.offset_bias.round().to(torch.int32).reshape(-1)

    def _init_deq_act_scale(self, quant_module, device):
        """
        Function: Initialize dequantization scale parameters
        Args:
            quant_module: quantization module containing quantization parameters
            device: device to place tensors on
        """
        if self.act_granularity == 'token':
            self.pertoken_scale = quant_module.scale_d.reshape(-1).to(torch.float32).to(device=device)
        elif self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
            self.y_scale = self.y_scale_tensor
            self.group_sizes = [0, 0, 32]
    
    def _init_deq_weight_scale(self, quant_module, device):
        """
        Function: Initialize dequantization scale parameters
        Args:
            quant_module: quantization module containing quantization parameters
            device: device to place tensors on
        """
        if self.act_granularity == 'token':
            self.register_buffer('deq_scale', self.scale_w_tensor)
        elif self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
            self.register_buffer('deq_scale', self.scale_w2)
        elif self.act_type == INT8 and self.output_dtype == torch.float16:
            # deq_scale must be set to int64 of npu op when fp16 activation do int8 quantization
            import torch_npu
            deq_scale_tensor = quant_module.scale_d.to(device=device) * self.scale_w_tensor
            deq_scale_tensor = torch_npu.npu_trans_quant_param(deq_scale_tensor.reshape(1, -1)).reshape(-1)
            self.register_buffer('deq_scale', deq_scale_tensor)
        else:
            deq_scale_tensor = quant_module.scale_d.to(device=device) * self.scale_w_tensor
            self.register_buffer('deq_scale', deq_scale_tensor)

    def _init_npu_quantize_type(self):
        """
        Function: Initialize NPU quantization type and check torch_npu version
        """
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
        elif self.wts_type == INT4:
            # int8 * int4: npu_quant_matmul requires x2_dtype be int4 while the packed
            # quantized_weight is physically stored as int8 (two int4 per byte).
            self.x2_dtype = torch_npu.int4

    def _init_bias(self, module):
        """Decide bias form and set self.bias (passed to npu_quant_matmul, may be None):
          - INT8 PER_TENSOR + non-dynamic: quantize bias to INT32
          - FP8*FP8 / HIFLOAT8*HIFLOAT8 / INT8 PER_TOKEN / dynamic: keep fp
          - module.bias is None: handle offset_bias if present
        """
        if module.bias is None:
            self.bias = self.offset_bias
            return

        if (self.act_granularity == 'tensor'
                and self.dynamic is not True
                and self.act_type == INT8):
            deq_scale = module.scale_d * self.scale_w_tensor
            bias_tensor = (module.bias.data / deq_scale).round().clamp(INT32_MIN, INT32_MAX).to(torch.int32)
            self.register_buffer('bias', bias_tensor)
        else:
            self.register_buffer('bias', module.bias.to(torch.float32))

        if self.offset_bias is not None:
            self.bias = self.bias + self.offset_bias

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
            if self.wts_type == INT4:
                # int8 * int4 (the only int4 combo reaching this op): npu_quant_matmul requires
                # the int4 weight packed into int8 (two int4 spliced into one byte) along the
                # cout axis. The packed weight is passed with shape [k, n // 2] (k=cin, n=cout).
                # Keep the unpacked [k, n] int4 values for the offset_bias computation, which
                # needs the real per-cout weight rather than the packed bytes.
                self.weight_before_pack = weight_tensor
                weight_tensor = self._pack_int4_to_int8(weight_tensor).to(device=device)
            self.scale_w_tensor = self.scale_w_tensor.reshape(-1)
        return weight_tensor

    def _pack_int4_to_int8(self, weight_tensor):
        """
        Function: pack int4 weight (values in [-8, 7]) into int8, splicing two adjacent
            int4 elements along the cout (last) axis into one byte. The low nibble holds
            the even-indexed element and the high nibble holds the odd-indexed element.
        Args:
            weight_tensor: torch.tensor of int4 values stored as int32, shape [k, n] (cin, cout)
        Returns:
            torch.tensor of dtype int8, shape [k, n // 2]
        """
        if weight_tensor.shape[-1] % 2 != 0:
            raise RuntimeError(
                "int8 * int4 packing requires an even cout dim, current is {}".format(
                    weight_tensor.shape[-1]))
        # keep only the low 4 bits so negative values map to their two's-complement nibble
        vals = (weight_tensor & 0x0F).to(torch.int16)
        low = vals[..., 0::2]
        high = vals[..., 1::2]
        combined = (low | (high << 4)).to(torch.int8)
        return combined.contiguous()
