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
import os
import copy
from collections.abc import Iterable

import torch
import torch.nn as nn

from .....amct_pytorch.common.utils.check_params import check_params
from .....amct_pytorch.utils.log import LOGGER
from .....amct_pytorch.custom_op.ifmr import ifmr
from .....amct_pytorch.custom_op.hfmg import hfmg
from .....amct_pytorch.custom_op.recorder.recorder import Recorder
from .....amct_pytorch.common.utils import files as files_util
from .....amct_pytorch.common.config.field import NUM_OF_BINS_RANGE
from .....amct_pytorch.utils.vars import BATCH_NUM, ASYMMETRIC, QUANT_GRANULARITY, MAX_PERCENTILE,\
    MIN_PERCENTILE, SEARCH_RANGE, PER_TENSOR, PER_CHANNEL, ACT_ALGO, WITH_OFFSET, \
    NUM_BITS, SEARCH_STEP, IFMR, HFMG
from .....amct_pytorch.common.utils.vars_util import DEFAULT_MAX_PERCENTILE, DEFAULT_MIN_PERCENTILE,\
    DEFAULT_SEARCH_RANGE_START, DEFAULT_SEARCH_RANGE_END, DEFAULT_SEARCH_STEP, DEFUALT_NUM_OF_BINS,\
    PER_TENSOR_IDX, PER_CHANNEL_IDX

QUANT_GRANULARITY_MAP = {
    PER_TENSOR: PER_TENSOR_IDX,
    PER_CHANNEL: PER_CHANNEL_IDX
}

COMMON_PARAMS_CHECK = {
    BATCH_NUM: (int, lambda x: x > 0),
    ASYMMETRIC: (bool, None),
    QUANT_GRANULARITY: (str, (PER_TENSOR, PER_CHANNEL)),
    ACT_ALGO: (str, (IFMR, HFMG)),
}

IFMR_PARAMS_CHECK = {
    MAX_PERCENTILE: (float, lambda x: x >= 0.5 and x <= 1.0),
    MIN_PERCENTILE: (float, None),
    SEARCH_RANGE: ((list, tuple), lambda x: len(x) == 2 and x[0] < x[1] and x[0] > 0),
    SEARCH_STEP: (float, lambda x: x > 0)
}

HFMG_PARAMS_CHECK = {
    'num_of_bins': (int, NUM_OF_BINS_RANGE)
}


class QuantCalibrationOp(nn.Module):
    """
    A amct pytorch module used to do calibration for inputs with ifmr/hfmg
    """
    @check_params(record_file=str,
                  quant_algo_params=(type(None), dict),
                  quant_method=str)
    def __init__(self,
                 record_file,
                 quant_algo_params=None,
                 quant_method='kv_cache_quant'):
        super().__init__()
        files_util.is_valid_name(record_file, 'record_file')
        self.record_file = os.path.realpath(record_file)
        check_exist = False
        if os.path.exists(self.record_file) and os.path.getsize(self.record_file) != 0:
            check_exist = True
        files_util.create_empty_file(self.record_file, check_exist=check_exist)
        if quant_algo_params is None:
            quant_algo_params = dict()
        self.quant_algo, self.quant_params = QuantCalibrationOp.parse_quant_algo_params(quant_algo_params)
        self.calibration_module_map = dict()

        self.cur_batch = dict()
        self.batch_num = quant_algo_params.get(BATCH_NUM, 1)
        if quant_method == 'kv_cache_quant':
            self.quant_factor = 'kv_cache'
            self.record_keyword = 'kv_cache_value'
        else:
            raise RuntimeError(
                'Unsupported quant_method {}. supported quant_method includes kv_cache_quant'.format(quant_method))
        self.recorder = Recorder(
            self.record_file, enable_kv_cache_quant=True)

    @staticmethod
    def parse_quant_algo_params(quant_algo_params):
        """
        Parse the user quant params and fill in the default value on quant algo
        Args:
            quant_algo_params (dict): quant params given by user
        """
        act_algo = quant_algo_params.get(ACT_ALGO, IFMR)
        quant_params = dict()
        quant_params[BATCH_NUM] = quant_algo_params.get(BATCH_NUM, 1)
        quant_params[WITH_OFFSET] = quant_algo_params.get(ASYMMETRIC, True)
        quant_granularity = quant_algo_params.get(
            QUANT_GRANULARITY, PER_TENSOR)
        quant_params[QUANT_GRANULARITY] = QUANT_GRANULARITY_MAP.get(
            quant_granularity)
        params_check_map = copy.deepcopy(COMMON_PARAMS_CHECK)
        if act_algo == IFMR:
            quant_params[MAX_PERCENTILE] = quant_algo_params.get(
                MAX_PERCENTILE, DEFAULT_MAX_PERCENTILE)
            quant_params[MIN_PERCENTILE] = quant_algo_params.get(
                MIN_PERCENTILE, DEFAULT_MIN_PERCENTILE)
            quant_params[SEARCH_RANGE] = quant_algo_params.get(
                SEARCH_RANGE, [DEFAULT_SEARCH_RANGE_START, DEFAULT_SEARCH_RANGE_END])
            if not isinstance(quant_params.get(SEARCH_RANGE), (tuple, list)) or len(quant_params.get(SEARCH_RANGE)) < 2:
                raise RuntimeError(
                    'Invalid quant params search range for it is not tuple or list or its length smaller than 2')
            quant_params['search_start'] = quant_params.get(SEARCH_RANGE)[0]
            quant_params['search_end'] = quant_params.get(SEARCH_RANGE)[1]
            quant_params[SEARCH_STEP] = quant_algo_params.get(
                SEARCH_STEP, DEFAULT_SEARCH_STEP)

            params_check_map.update(copy.deepcopy(IFMR_PARAMS_CHECK))
            QuantCalibrationOp.check_quant_params(
                quant_algo_params, params_check_map)
            del quant_params[SEARCH_RANGE]

        elif act_algo == HFMG:
            quant_params['nbins'] = quant_algo_params.get(
                'num_of_bins', DEFUALT_NUM_OF_BINS)
            params_check_map.update(copy.deepcopy(HFMG_PARAMS_CHECK))
            QuantCalibrationOp.check_quant_params(
                quant_algo_params, params_check_map)
        else:
            raise RuntimeError(
                'your act_algo {} if not supported. supported act_algo includes ifmr, hfmg'.format(act_algo))

        return act_algo, quant_params

    @staticmethod
    def check_quant_params(quant_algo_params, param_check_map):
        """
        check parsed quant params and user original quant params on act_algo
        Args:
            quant_algo_params(dict): user original quant params
            param_check_map(dict): check rules based on the algorithm type parameter
        """
        for param_name, _ in quant_algo_params.items():
            if param_name not in param_check_map.keys():
                if quant_algo_params.get('act_algo', IFMR) == IFMR and param_name in HFMG_PARAMS_CHECK:
                    raise RuntimeError(
                        'Parameter {} only supported in HFMG while your act_algo is IFMR'.format(param_name))
                if quant_algo_params.get('act_algo') == HFMG and param_name in IFMR_PARAMS_CHECK:
                    raise RuntimeError(
                        'Parameter {} only supported in IFMR while your act_algo is HFMG'.format(param_name))
                raise RuntimeError('Unknown parameter {}'.format(param_name))

        for param_name, param_val in quant_algo_params.items():
            if param_check_map.get(param_name) is None:
                continue
            if not isinstance(param_val, param_check_map.get(param_name)[0]):
                raise TypeError('Quant parameter {} should be {} but your input is {}'.format(
                    param_name, param_check_map.get(param_name)[0], param_val))
            if not param_check_map.get(param_name)[1]:
                continue
            elif isinstance(param_check_map.get(param_name)[1], Iterable):
                if param_val not in param_check_map.get(param_name)[1]:
                    raise RuntimeError('Quant parameter {} scope is {} while your input is {}'.format(
                        param_name, param_check_map.get(param_name)[1], param_val))
            else:
                if not param_check_map.get(param_name)[1](param_val):
                    raise RuntimeError(
                        'Quant parameter {} {} is illeagal'.format(param_name, param_val))

    @check_params(calibrated_layer_name=str)
    def forward(self, calibrated_layer_name, inputs):
        """
        do forward
        Args:
            calibrated_layer_name(str): layer named written in record file
            inputs(torch.Tensor): user model input
        """
        with torch.no_grad():
            if calibrated_layer_name not in self.cur_batch:
                self.cur_batch[calibrated_layer_name] = 0
                self.quant_params['layers_name'] = calibrated_layer_name
                if self.quant_algo == IFMR:
                    self.calibration_module_map[calibrated_layer_name] = ifmr.IFMR(**self.quant_params)
                else:
                    self.calibration_module_map[calibrated_layer_name] = hfmg.HFMG(**self.quant_params)
            self.cur_batch[calibrated_layer_name] += 1
            if self.cur_batch.get(calibrated_layer_name) <= self.batch_num:
                self.calibrate_process(calibrated_layer_name, inputs)
        return inputs

    def calibrate_process(self, calibrated_layer_name, inputs):
        """
        core process of doing calibration
        Args:
            calibrated_layer_name(str): layer named written in record file
            inputs(torch.Tensor): user input
        """
        quant_info = self.calibration_module_map.get(calibrated_layer_name).forward(inputs)
        calibration_flag = quant_info.flag

        if calibration_flag:
            self.recorder.record_quant_layer([calibrated_layer_name])
            scale = list(map(lambda x: x.cpu().tolist(), quant_info.scale))
            offset = list(map(lambda x: int(x.cpu().tolist()), quant_info.offset))

            if self.recorder.check_layer_recorded(calibrated_layer_name, self.record_keyword):
                LOGGER.logw('Layer {} already have {} in record file {}. It will be overwritten by AMCT'.format(
                    calibrated_layer_name, self.record_keyword, self.record_file))
            self.recorder.forward([calibrated_layer_name],
                                  self.quant_factor,
                                  {'scale': scale,
                                   'offset': offset,
                                   })
