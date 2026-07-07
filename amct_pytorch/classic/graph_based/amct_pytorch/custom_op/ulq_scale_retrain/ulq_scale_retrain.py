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
from torch.autograd import Function
import torch

from ....amct_pytorch.common.utils.vars_util import RNN_TENSOR_NUM
from ....amct_pytorch.utils.log import LOGGER
from ....amct_pytorch.custom_op import ulq_scale_retrain_forward_pytorch
from ....amct_pytorch.custom_op import ulq_scale_retrain_backward_pytorch
from ....amct_pytorch.custom_op.utils import check_quant_data
from ....amct_pytorch.custom_op.utils import check_group_param
from ....amct_pytorch.custom_op.utils import process_tensor_shape
from ....amct_pytorch.utils.vars import QUANTIZE_LINEAR
from ....amct_pytorch.utils.vars import DEQUANTIZE_LINEAR
from ....amct_pytorch.utils.vars import TRANSPOSE
from ....amct_pytorch.utils.weight_quant_api import adjust_axis_for_group_wise


MODULE_TYPE = 'module_type'


class UlqScaleRetrainFunction(Function):
    """
    Function: Run calibration process for quantization of the given layer.
    APIs: forward
    """
    @staticmethod
    def forward(ctx, inputs, scale, offset, wts_qat_param, cur_batch, offset_deploy=None, group=1, axis=0):
        """
        Function: UlqRetrain foward funtion.
        """
        # check input data
        check_quant_data(inputs, 'weights')
        
        # trans quantized axis to 0 in group wise
        group_wise = check_group_param(inputs, wts_qat_param.get('channel_wise'), group, axis)
        if group_wise:
            processed_tensor = adjust_axis_for_group_wise(axis, inputs)

        inputs_processed = process_tensor_shape(inputs, wts_qat_param.get(MODULE_TYPE),
            wts_qat_param.get('module'))

        ctx.inputs = inputs_processed
        ctx.module_type = wts_qat_param.get(MODULE_TYPE)
        results = ulq_scale_retrain_forward_pytorch(
            inputs_processed,
            scale,
            offset,
            wts_qat_param.get('num_bits'),
            wts_qat_param.get('channel_wise'),
            wts_qat_param.get('arq_init') and cur_batch == 0,
            wts_qat_param.get('s_rec_flag'),
            group)

        outputs, scale_out, offset_out = results
        scale_out = scale_out.reshape(scale.shape)
        offset_out = offset_out.reshape(offset.shape)
        scale.data.copy_(scale_out.data)
        offset.data.copy_(offset_out.data)

        ctx.scale = scale
        ctx.num_bits = wts_qat_param.get('num_bits')
        ctx.s_rec_flag = wts_qat_param.get('s_rec_flag')
        ctx.group = group
        ctx.axis = axis

        outputs = process_tensor_shape(outputs, wts_qat_param.get(MODULE_TYPE),
            wts_qat_param.get('module'))
        # transpose wts axis back in group wise
        if group_wise and axis > 0:
            processed_tensor = adjust_axis_for_group_wise(axis, outputs)
        return outputs, scale, offset

    @staticmethod
    def backward(ctx, grad_outputs, grad_scale, grad_offset):
        """
        Function: UlqRetrain backward funtion required by torch
                  torch.autograd.
        """
        res = ulq_scale_retrain_backward_pytorch(
            ctx.inputs,
            grad_outputs,
            ctx.scale,
            ctx.num_bits,
            ctx.s_rec_flag,
            ctx.group,
            ctx.axis)

        grad_input, grad_scale = res

        if ctx.module_type in ['ConvTranspose1d', 'ConvTranspose2d']:
            grad_input = grad_input.transpose(1, 0)
        ret = (grad_input, grad_scale, None, None, None, None, None, None)
        return ret


class UlqScaleRetrainFuncQAT(UlqScaleRetrainFunction):
    @staticmethod
    def symbolic(g, *inputs):
        """
        Turn ULQ scale retrain op to onnx QDQ structure.
        Args:
            g (Graph): graph to write the ONNX representation into.
        """
        module_type = inputs[3].get(MODULE_TYPE)
        if module_type in ["ConvTranspose1d", "ConvTranspose2d"]:
            quant = g.op(QUANTIZE_LINEAR, inputs[0], inputs[1], inputs[5])
            out_node = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[5])
        elif module_type == 'Conv1d':
            transpose = g.op(TRANSPOSE, inputs[0], perm_i=list([1, 0, 2]))
            quant = g.op(QUANTIZE_LINEAR, transpose, inputs[1], inputs[5])
            dequant = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[5])
            out_node = g.op(TRANSPOSE, dequant, perm_i=list([1, 0, 2]))
        elif module_type == 'Conv2d':
            transpose = g.op(TRANSPOSE, inputs[0], perm_i=list([1, 0, 2, 3]))
            quant = g.op(QUANTIZE_LINEAR, transpose, inputs[1], inputs[5])
            dequant = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[5])
            out_node = g.op(TRANSPOSE, dequant, perm_i=list([1, 0, 2, 3]))
        elif module_type == 'Conv3d':
            transpose = g.op(TRANSPOSE, inputs[0], perm_i=list([1, 0, 2, 3, 4]))
            quant = g.op(QUANTIZE_LINEAR, transpose, inputs[1], inputs[5])
            dequant = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[5])
            out_node = g.op(TRANSPOSE, dequant, perm_i=list([1, 0, 2, 3, 4]))
        elif module_type == 'Linear':
            quant = g.op(QUANTIZE_LINEAR, inputs[0], inputs[1], inputs[5])
            out_node = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[5])
        elif module_type in RNN_TENSOR_NUM:
            shape = g.op('Constant', value_t=torch.tensor(
                    [1, RNN_TENSOR_NUM.get(module_type) * inputs[3]['module'].hidden_size, -1], dtype=torch.int64))
            reshape = g.op('Reshape', inputs[0], shape)
            quant = g.op(QUANTIZE_LINEAR, reshape, inputs[1], inputs[5])
            out_node = g.op(DEQUANTIZE_LINEAR, quant, inputs[1], inputs[5])
        LOGGER.logi(f'Convert ULQ scale op to onnx QuantizeLinear and DequantizeLinear op successfully.')
        return out_node, None, None
