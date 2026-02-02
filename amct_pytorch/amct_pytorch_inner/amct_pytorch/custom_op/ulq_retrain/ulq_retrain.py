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

from ....amct_pytorch.utils.log import LOGGER
from ....amct_pytorch.custom_op import ulq_retrain_forward_pytorch
from ....amct_pytorch.custom_op import ulq_retrain_backward_pytorch
from ....amct_pytorch.custom_op.utils import check_quant_data


class UlqRetrainFunction(Function):
    """
    Function: Run calibration process for quantization of the given layer.
    APIs: forward
    """
    @staticmethod
    def forward(ctx,
                inputs, clip_max, clip_min, clip_max_pre, clip_min_pre,
                act_qat_param, cur_batch, need_sync,
                process_group, world_size):
        """
        Function: UlqRetrain foward funtion.
        """
        check_quant_data(inputs, 'activation')
        ctx.inputs = inputs
        outputs, scale, offset, clip_max, clip_min = ulq_retrain_forward_pytorch(
            inputs,
            clip_max,
            clip_min,
            clip_max_pre,
            clip_min_pre,
            act_qat_param.get('num_bits'),
            act_qat_param.get('fixed_min'),
            act_qat_param.get('asymmetric', True))

        if need_sync:
            clip_max_all = torch.empty(
                world_size, 1, dtype=clip_max.dtype, device=clip_max.device)
            clip_min_all = torch.empty(
                world_size, 1, dtype=clip_min.dtype, device=clip_min.device)

            clip_max_l = list(clip_max_all.unbind(0))
            clip_min_l = list(clip_min_all.unbind(0))

            clip_max_all_reduce = torch.distributed.all_gather(
                clip_max_l, clip_max, process_group, async_op=True)
            clip_min_all_reduce = torch.distributed.all_gather(
                clip_min_l, clip_min, process_group, async_op=True)

            # wait on the async communication to finish
            clip_max_all_reduce.wait()
            clip_min_all_reduce.wait()
            clip_max_tmp = clip_max_all.mean()
            clip_min_tmp = clip_min_all.mean()
            clip_max.data.copy_(clip_max_tmp.data)
            clip_min.data.copy_(clip_min_tmp.data)

        ctx.clip_max = clip_max
        ctx.clip_min = clip_min
        ctx.num_bits = act_qat_param.get('num_bits')
        ctx.asymmetric = act_qat_param.get('asymmetric', True)

        return outputs, scale, offset, clip_max, clip_min

    @staticmethod
    def backward(ctx,
                 grad_outputs, grad_scale,
                 grad_offset, grad_max, grad_min):
        """
        Function: UlqRetrain backward funtion required by torch
                  torch.autograd.
        """
        res = ulq_retrain_backward_pytorch(ctx.inputs,
            grad_outputs,
            ctx.clip_max,
            ctx.clip_min,
            ctx.num_bits,
            ctx.asymmetric)

        grad_input, grad_acts_clip_max, grad_acts_clip_min = res
        return grad_input, grad_acts_clip_max, grad_acts_clip_min, \
            None, None, None, None, None, None, None


class UlqRetrainFuncQAT(UlqRetrainFunction):
    @staticmethod
    def symbolic(g, *inputs):
        """
        Turn ULQ retrain op to onnx QDQ structure.
        Args:
            g (Graph): graph to write the ONNX representation into.
        """
        output = g.op("DequantizeLinear",
                      g.op("QuantizeLinear", inputs[0],
                           inputs[5].get("acts_scale"),
                           inputs[5].get("acts_offset")),
                      inputs[5].get("acts_scale"),
                      inputs[5].get("acts_offset"))
        LOGGER.logi(f'Convert ULQ op to onnx to onnx QuantizeLinear and DequantizeLinear op successfully.')
        ret = (output, None, None, None, None)
        return ret
