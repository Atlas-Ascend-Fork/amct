#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
from torch.autograd import Function

from ....amct_pytorch.common.utils.vars_util import RNN_TENSOR_NUM
from ....amct_pytorch.utils.log import LOGGER
from ....amct_pytorch.custom_op import arq_retrain_forward_pytorch
from ....amct_pytorch.custom_op import arq_retrain_backward_pytorch
from ....amct_pytorch.custom_op.utils import check_quant_data
from ....amct_pytorch.custom_op.utils import check_group_param
from ....amct_pytorch.custom_op.utils import process_tensor_shape
from ....amct_pytorch.utils.vars import QUANTIZE_LINEAR
from ....amct_pytorch.utils.vars import DEQUANTIZE_LINEAR
from ....amct_pytorch.utils.vars import TRANSPOSE
from ....amct_pytorch.utils.weight_quant_api import adjust_axis_for_group_wise


class ArqRetrainFunction(Function):
    """
    Function: Run weight retrain process for quantization of the given layer.
    APIs: forward, backward
    """
    @staticmethod
    def forward(ctx, weight_tensor, scale, offset, wts_param, offset_deploy=None, group=1, axis=0):
        """
        Function: ArqRetrain foward funtion.
        """
        # check weight tensor
        check_quant_data(weight_tensor, 'weight')
        weight_tensor_processed = process_tensor_shape(weight_tensor, wts_param.get('module_type'),
            wts_param.get('module'))
        
        # trans quantized axis to 0 in group wise
        group_wise = check_group_param(weight_tensor, wts_param.get('channel_wise'), group, axis)
        if group_wise:
            processed_tensor = adjust_axis_for_group_wise(axis, weight_tensor)
        scale_out, offset_out, quantized_weight = arq_retrain_forward_pytorch(
            weight_tensor_processed,
            wts_param.get('num_bits'),
            wts_param.get('channel_wise'),
            wts_param.get('with_offset'))

        scale.data.copy_(scale_out.data)
        offset.data.copy_(offset_out.data)
        quantized_weight = process_tensor_shape(quantized_weight, wts_param.get('module_type'),
            wts_param.get('module'))
        quantized_weight = quantized_weight.to(weight_tensor.device)
        
        # transpose wts axis back in group wise
        if group_wise and axis > 0:
            processed_tensor = adjust_axis_for_group_wise(axis, quantized_weight)

        return quantized_weight, scale, offset

    @staticmethod
    def backward(ctx, grad_outputs, grad_scale, grad_offset):
        """
        Function: ArqRetrain backward funtion required by torch torch.autograd.
        """
        grad_input = arq_retrain_backward_pytorch(grad_outputs)
        ret = (grad_input, None, None, None, None)
        return ret


class ArqRetrainFuncQAT(ArqRetrainFunction):
    @staticmethod
    def symbolic(g, *inputs):
        """
        Turn ARQ retrain op to onnx QDQ structure.
        Args:
            g (Graph): graph to write the ONNX representation into.
        """
        module_type = inputs[3].get('module_type')
        if module_type in ["ConvTranspose1d", "ConvTranspose2d"]:
            quant = g.op(QUANTIZE_LINEAR, inputs[0], inputs[1], inputs[4])
            out_node = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[4])
        elif module_type == 'Conv1d':
            transpose = g.op(TRANSPOSE, inputs[0], perm_i=list([1, 0, 2]))
            quant = g.op(QUANTIZE_LINEAR, transpose, inputs[1], inputs[4])
            dequant = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[4])
            out_node = g.op(TRANSPOSE, dequant, perm_i=list([1, 0, 2]))
        elif module_type == 'Conv2d':
            transpose = g.op(TRANSPOSE, inputs[0], perm_i=list([1, 0, 2, 3]))
            quant = g.op(QUANTIZE_LINEAR, transpose, inputs[1], inputs[4])
            dequant = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[4])
            out_node = g.op(TRANSPOSE, dequant, perm_i=list([1, 0, 2, 3]))
        elif module_type == 'Conv3d':
            transpose = g.op(TRANSPOSE, inputs[0], perm_i=list([1, 0, 2, 3, 4]))
            quant = g.op(QUANTIZE_LINEAR, transpose, inputs[1], inputs[4])
            dequant = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[4])
            out_node = g.op(TRANSPOSE, dequant, perm_i=list([1, 0, 2, 3, 4]))
        elif module_type == 'Linear':
            quant = g.op(QUANTIZE_LINEAR, inputs[0], inputs[1], inputs[4])
            out_node = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[4])
        elif module_type in RNN_TENSOR_NUM:
            shape = g.op('Constant', value_t=torch.tensor(
                [1, RNN_TENSOR_NUM.get(module_type) * inputs[3]['module'].hidden_size, -1], dtype=torch.int64))
            reshape = g.op('Reshape', inputs[0], shape)
            quant = g.op(QUANTIZE_LINEAR, reshape, inputs[1], inputs[4])
            out_node = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[4])
        LOGGER.logi(
            f'Convert ARQ op to onnx QuantizeLinear and DequantizeLinear op successfully.')
        return out_node, None, None
