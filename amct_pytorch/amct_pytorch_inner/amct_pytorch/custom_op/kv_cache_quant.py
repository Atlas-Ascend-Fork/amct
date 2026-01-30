#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

KVCacheQuant module and Function

"""

import torch
from torch import nn


class KVCacheQuant(nn.Module):
    """
    Function: Customized torch.nn.Module of the kv-cache calibration class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 cali_module,
                 record_module,
                 layer_name,
                 cali_algo_params):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Quantized_type.
        cali_module: calibration module. IFMR or HFMG
        record_module: customed record module. To read and write.
        layer_name: ori_module's name.
        cali_algo_params: calibration algorithm parameters.
        """
        super().__init__()
        self.ori_module = ori_module
        self.cali_module = cali_module
        self.record_module = record_module
        self.layer_name = layer_name
        self.cali_algo_params = cali_algo_params
        self.cur_batch = 0

    def forward(self, inputs):
        """
        Function: KVCacheQuant foward funtion.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        with torch.no_grad():
            # step 1. original module forward
            output = self.ori_module(inputs)

            # step 2. calibration module (ifmr/hfmg) forward
            self.cur_batch += 1
            if self.cur_batch <= self.cali_algo_params.get('batch_num'):
                quant_info = self.cali_module.forward(output)
                cali_done = quant_info.flag
                if cali_done:
                    # step 3. save scale and offset to record_module
                    scale = [float(each_scale.cpu()) for each_scale in quant_info.scale]
                    offset = [int(each_offset.cpu()) for each_offset in quant_info.offset]
                    self.record_module(self.layer_name, 'kv_cache',
                                       {'scale': scale,
                                        'offset': offset})
            return output
