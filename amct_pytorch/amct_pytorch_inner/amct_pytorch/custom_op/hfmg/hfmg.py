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

from collections import namedtuple
from torch import nn
import torch
from ....amct_pytorch.common.utils.vars_util import PER_TENSOR_IDX, PER_CHANNEL_IDX
from ....amct_pytorch.custom_op import hfmg_arq_pytorch
from ....amct_pytorch.custom_op import hfmg_merge_pytorch
from ....amct_pytorch.custom_op import hfmg_forward_pytorch
from ....amct_pytorch.custom_op.utils import check_quant_data
from ....amct_pytorch.utils.log import LOGGER

QuantInfo = namedtuple('QuantInfo', ['flag', 'scale', 'offset', 'clip_max', 'clip_min'])


class HFMG(nn.Module):
    """
    Function: Run calibration process for quantization of the given layer.
    APIs: forward
    """
    def __init__(self,
                 layers_name,
                 num_bits=8,
                 batch_num=2,
                 with_offset=True,
                 nbins=4096,
                 quant_granularity=PER_TENSOR_IDX):
        super().__init__()
        self.param = {}
        self.param['num_bits'] = num_bits
        self.param['batch_num'] = batch_num
        self.param['with_offset'] = with_offset
        self.param['nbins'] = nbins
        self.param['quant_granularity'] = quant_granularity

        self.layers_name = layers_name

        self.register_buffer('histogram', torch.zeros(nbins))
        self.nbins = nbins
        self.hfmg_data = []
        self.cur_batch = 0
        self.calibrated_flag = False

    def forward(self, inputs):
        """
        Function: HFMG foward funtion.
        """
        self.cur_batch += 1
        if inputs.dtype is torch.float16:
            inputs = inputs.to(dtype=torch.float)
        if self.param['quant_granularity'] == PER_CHANNEL_IDX:
            inputs_list = list(torch.chunk(inputs, inputs.shape[-1], dim=-1))
            self._merge_histogram(inputs_list)
        else:
            self._merge_histogram([inputs])
        LOGGER.logd("Accumulated {} batch to do layer {} data calibration.".format(
            self.cur_batch, self.layers_name))

        hfmg_param = self.param
        do_calibration = (self.cur_batch == hfmg_param.get('batch_num'))
        clip_max = torch.tensor(0, device=inputs.device)
        clip_min = torch.tensor(0, device=inputs.device)
        if not do_calibration:
            input_min = torch.min(inputs)
            input_max = torch.max(inputs)
            scale, offset = hfmg_arq_pytorch(input_min, input_max,
                self.param.get('num_bits'), self.param.get('with_offset'))

            return QuantInfo._make([self.calibrated_flag, [scale], [offset], clip_max, clip_min])

        LOGGER.logi("Use {} batch to do layer {} data calibration.".format(
            hfmg_param.get('batch_num'), self.layers_name))
        scale_list, offset_list = [], []
        for min_val, max_val, histogram in self.hfmg_data:
            min_max = torch.cat([min_val.reshape(-1), max_val.reshape(-1)])
            scale, offset, clip_max, clip_min = hfmg_forward_pytorch(histogram,
                                                    min_max,
                                                    hfmg_param.get('num_bits'),
                                                    hfmg_param.get('with_offset'),
                                                    hfmg_param.get('nbins'))
            scale_list.append(scale)
            offset_list.append(offset)
        del self.hfmg_data
        self.hfmg_data = []
        self.calibrated_flag = True
        LOGGER.logi("Do layer {} data calibration succeeded!"
                    .format(self.layers_name), 'HFMG')

        return QuantInfo._make([self.calibrated_flag, scale_list, offset_list, clip_max, clip_min])

    def _merge_histogram(self, inputs_list):
        """ Accumulate data for hfmg and search"""
        if self.cur_batch > self.param.get('batch_num'):
            return

        if len(self.hfmg_data) == 0:
            for index, data in enumerate(inputs_list):
                check_quant_data(data, 'activation')
                min_val = torch.min(data)
                max_val = torch.max(data)
                histc_min = float(min_val.cpu().detach().numpy())
                histc_max = float(max_val.cpu().detach().numpy())
                histogram = torch.histc(
                    data, self.nbins, min=histc_min, max=histc_max).contiguous()
                self.hfmg_data.append((min_val, max_val, histogram))
        elif len(self.hfmg_data) == len(inputs_list):
            for index, data in enumerate(inputs_list):
                check_quant_data(data, 'activation')
                min_val = self.hfmg_data[index][0]
                max_val = self.hfmg_data[index][1]
                new_min = torch.min(data)
                new_max = torch.max(data)
                combined_min = torch.min(new_min, min_val)
                combined_max = torch.max(new_max, max_val)
                combined_min_max = torch.cat([combined_min.reshape(-1), combined_max.reshape(-1)])
                histc_min = float(combined_min.cpu().detach().numpy())
                histc_max = float(combined_max.cpu().detach().numpy())
                new_hist = torch.histc(
                    data, self.nbins, min=histc_min, max=histc_max).contiguous()
 
                min_max = torch.cat([min_val.reshape(-1), max_val.reshape(-1)])
                merged_hist = hfmg_merge_pytorch(self.hfmg_data[index][2], min_max, new_hist,
                    combined_min_max, self.nbins)
                self.hfmg_data[index] = (combined_min, combined_max, merged_hist)
        else:
            raise RuntimeError("Layer {} cout length({}) not equal to accumulated data cout length({})!"
                .format(self.layers_name, len(inputs_list), len(self.hfmg_data)), 'HFMG')

        LOGGER.logi(
            "Doing layer {} data calibration: data already preprocess {}/{}"
            .format(self.layers_name, self.cur_batch, self.param.get('batch_num')),
            'HFMG')
