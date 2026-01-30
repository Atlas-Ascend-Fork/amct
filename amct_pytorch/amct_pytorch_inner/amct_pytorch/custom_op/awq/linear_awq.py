#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

LinearAWQuant module and Function

"""
import torch
from torch import nn
import torch.nn.functional as F

from amct_pytorch.algo.awq import search_scale, apply_scale, search_clip, \
    apply_clip_by_weight_granularity, get_quant_scale
from amct_pytorch.utils.data_utils import check_data_type, check_linear_input_dim
from amct_pytorch.utils.vars import MXFP4_E2M1, FLOAT4_E2M1, FLOAT4_E1M2, INT4, INT8


class LinearAWQuant(nn.Module):
    """
    Function: Customized torch.nn.Module of the LinearAWQuant class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 record_module,
                 layer_name,
                 quant_config,
                 do_awq=True):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        record_module: customed record module. To read and write.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__()
        self.weight = ori_module.weight
        self.bias = ori_module.bias
        self.ori_module = ori_module if do_awq else None
        self.post_quant_record_module = record_module
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.do_awq = do_awq
        self.wts_asymmetric = quant_config.get('wts_asymmetric')

    def forward(self, inputs):
        """
        Function: LinearAWQuant foward funtion.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        check_data_type(inputs.dtype, (torch.float16, torch.bfloat16))
        with torch.no_grad():
            check_linear_input_dim(inputs)
            inputs = inputs.to(self.weight.device)
            output = F.linear(inputs, self.weight, self.bias)
        if not self.do_awq:
            return output

        scale_awq = search_scale(inputs, [self.ori_module], self.ori_module, self.quant_config)

        scale_list = {self.layer_name[0]: scale_awq}
        named_linear = {self.layer_name[0]: self.ori_module}
        input_feat = {self.layer_name[0]: inputs}
        apply_scale(scale_list, named_linear, input_feat)

        clip_max_list = search_clip(named_linear, input_feat, self.quant_config)

        quant_params = {
            self.layer_name[0]: {
                'awq_quantize': {
                    'scale': 1 / scale_list[self.layer_name[0]].detach().cpu(),
                    'clip_max': clip_max_list[self.layer_name[0]].detach().cpu() \
                        if self.layer_name[0] in clip_max_list else None
                }
            }
        }


        # calculate the scale of pergroup 
        if self.quant_config.get('wts_type') in (FLOAT4_E2M1, FLOAT4_E1M2, INT4, INT8):
            cliped_weight = apply_clip_by_weight_granularity(clip_max_list, named_linear, \
                self.quant_config.get('weight_granularity'), self.quant_config.get("group_size"))
            scale_group_list, offset_group_list = get_quant_scale(cliped_weight, self.quant_config)

            quant_factor = {
                "quant_factors": {
                    'scale_w': scale_group_list[self.layer_name[0]].detach().cpu()
                }
            }
            if self.wts_asymmetric:
                quant_factor["quant_factors"]["offset_w"] = \
                    offset_group_list[self.layer_name[0]].detach().cpu()

            quant_params[self.layer_name[0]].update(quant_factor)

        quant_info = {'wts_type': self.quant_config.get('wts_type'),
                         'weight_compress_only': True,
                         'round_mode': 'RINT'}
        
        if self.quant_config.get("group_size") is not None:
            quant_info['group_size'] = self.quant_config.get("group_size")

        self.post_quant_record_module(self.layer_name, quant_info, quant_params)
        return output