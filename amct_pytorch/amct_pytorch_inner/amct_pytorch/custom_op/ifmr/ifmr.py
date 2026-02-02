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
from torch import nn # pylint: disable=E0401
import torch # pylint: disable=E0401
from ....amct_pytorch.common.utils.vars_util import PER_TENSOR_IDX, PER_CHANNEL_IDX
from ....amct_pytorch.custom_op import ifmr_forward_pytorch
from ....amct_pytorch.custom_op.utils import check_quant_data
from ....amct_pytorch.utils.log import LOGGER

QuantInfo = namedtuple('QuantInfo', ['flag', 'scale', 'offset', 'clip_max', 'clip_min'])


class IFMR(nn.Module): # pylint: disable=R0903
    """
    Function: Run calibration process for quantization of the given layer.
    APIs: forward
    """
    def __init__(self, # pylint: disable=R0913
                 layers_name,
                 num_bits=8,
                 batch_num=2,
                 with_offset=False,
                 max_percentile=0.999999,
                 min_percentile=0.999999,
                 search_start=0.7,
                 search_end=1.3,
                 search_step=0.01,
                 quant_granularity=PER_TENSOR_IDX):
        super().__init__()
        self.param = {}
        self.param['num_bits'] = num_bits
        self.param['batch_num'] = batch_num
        self.param['with_offset'] = with_offset
        self.param['max_percentile'] = max_percentile
        self.param['min_percentile'] = min_percentile
        self.param['search_start'] = search_start
        self.param['search_end'] = search_end
        self.param['search_step'] = search_step
        self.param['quant_granularity'] = quant_granularity

        self.layers_name = layers_name

        self.ifmr_data = []
        self.cur_batch = 0
        self.calibrated_flag = False

    def forward(self, inputs): # pylint: disable=W0221
        """
        Function: IFMR foward funtion.
        """
        with torch.no_grad():
            self.cur_batch += 1
            if inputs.dtype is torch.float16:
                inputs = inputs.to(dtype=torch.float)
            if self.param['quant_granularity'] == PER_CHANNEL_IDX:
                inputs_list = list(torch.chunk(inputs, inputs.shape[-1], dim=-1))
                self._accm_data(inputs_list)
            else:
                self._accm_data([inputs])
            LOGGER.logd("Accumulated {} batch to do layer {} data calibration.".format(
                self.cur_batch, self.layers_name))

            ifmr_param = self.param
            do_calibration = (self.cur_batch == ifmr_param.get('batch_num'))
            if not do_calibration:
                scale = torch.tensor(1.0) # pylint: disable=E1102
                offset = torch.tensor(0) # pylint: disable=E1102
                clip_max = torch.tensor(0)
                clip_min = torch.tensor(0)
                return QuantInfo._make([self.calibrated_flag, [scale], [offset], clip_max, clip_min])

            device = inputs.device
            LOGGER.logi("Use {} batch to do layer {} data calibration.".format(
                ifmr_param.get('batch_num'), self.layers_name))
            scale_list, offset_list = [], []
            
            for data in self.ifmr_data:
                scale, offset, clip_max, clip_min = self.apply_ifmr(
                    data, device, ifmr_param)
                scale_list.append(scale)
                offset_list.append(offset)

            del self.ifmr_data
            self.ifmr_data = []

            self.calibrated_flag = True
            LOGGER.logd("Do layer {} data calibration succeeded!"
                        .format(self.layers_name), 'IFMR')

        return QuantInfo._make([self.calibrated_flag, scale_list, offset_list, clip_max, clip_min])

    def apply_ifmr(self, data, device, ifmr_param):
        """apply ifmr algorithm on different devices to do activation calibrate"""
        scale, offset, clip_max, clip_min = ifmr_forward_pytorch(data,
                                        device.index,
                                        ifmr_param.get('num_bits'),
                                        ifmr_param.get('with_offset'),
                                        ifmr_param.get('max_percentile'),
                                        ifmr_param.get('min_percentile'),
                                        ifmr_param.get('search_start'),
                                        ifmr_param.get('search_end'),
                                        ifmr_param.get('search_step'))
        return scale, offset, clip_max, clip_min

    def _accm_data(self, inputs_list):
        """ Accumulate data for ifmr and search"""
        if self.cur_batch > self.param.get('batch_num'):
            return

        if len(self.ifmr_data) == 0:
            for index, data in enumerate(inputs_list):
                check_quant_data(data, 'activation')
                self.ifmr_data.append(data.cpu().reshape([-1]))
        elif len(self.ifmr_data) == len(inputs_list):
            for index, data in enumerate(inputs_list):
                check_quant_data(data, 'activation')
                self.ifmr_data[index] = torch.cat([self.ifmr_data[index], data.cpu().reshape([-1])])
        else:
            raise RuntimeError("Layer {} cout length({}) not equal to accumulated data cout length({})!"
                .format(self.layers_name, len(inputs_list), len(self.ifmr_data)), 'IFMR')
        LOGGER.logd(
            "Doing layer {} data calibration: data already stored {}/{}"
            .format(self.layers_name, self.cur_batch, self.param.get('batch_num')),
            'IFMR')
