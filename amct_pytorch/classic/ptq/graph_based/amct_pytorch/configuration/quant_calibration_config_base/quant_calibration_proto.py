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
from collections import OrderedDict, defaultdict, namedtuple
from enum import IntEnum, unique
from google.protobuf import text_format

from ....amct_pytorch.common.utils.util import check_no_repeated
from ....amct_pytorch.common.utils.util import find_repeated_items
from ....amct_pytorch.common.utils.util import proto_float_to_python_float
from ....amct_pytorch.proto import quant_calibration_config_pb2
from ....amct_pytorch.utils.log import LOGGER
from ....amct_pytorch.common.config.proto_config import ProtoDataType
from ....amct_pytorch.common.config.proto_config import ProtoConfig
from ....amct_pytorch.utils.vars import ASYMMETRIC, BATCH_NUM, ACTIVATION_OFFSET,\
    QUANT_GRANULARITY, MAX_PERCENTILE, MIN_PERCENTILE, SEARCH_RANGE, NUM_OF_BINS, IFMR, HFMG, SEARCH_STEP

QuantKeywordInfo = namedtuple('QuantKeywordInfo', ['layer_key', 'common_config_key', 'config_key'])
QUANT_METHOD_INFO_MAP = {
    'kv_quant': QuantKeywordInfo('kv_cache_quant_layers', 'kv_cache_quant_config', 'kv_data_quant_config')
}


class QuantCalibrationProtoConfig():
    """
    Function: Cope with simple config file from proto.
    APIs:
        get_proto_global_config
        get_override_layers
        read_override_layer_config
    """
    def __init__(self, config_proto_file, model):
        self.proto_config = self.read(config_proto_file)
        self.override_layer_proto = {}

    @staticmethod
    def read(config_proto_file):
        """ Read config from config_proto_file. """
        proto_config = quant_calibration_config_pb2.AMCTQuantCaliConfig()
        with open(config_proto_file, 'rb') as cfg_file:
            pbtxt_string = cfg_file.read()
            text_format.Merge(pbtxt_string, proto_config)
        return proto_config

    @staticmethod
    def _get_calibration_config(config):
        """ generate layer config base on common config """
        layer_config = OrderedDict()
        if config.HasField('ifmr_quantize'):
            QuantCalibrationProtoConfig._get_ifmr_config(config, layer_config)
        elif hasattr(config, 'hfmg_quantize') and config.HasField('hfmg_quantize'):
            QuantCalibrationProtoConfig._get_hfmg_config(config, layer_config)

        return layer_config

    @staticmethod
    def _get_hfmg_config(config, act_params):
        """extract hfmg configs"""
        act_params['act_algo'] = HFMG
        hfmg_quantize = config.hfmg_quantize
        num_of_bins = hfmg_quantize.num_of_bins
        act_params[NUM_OF_BINS] = num_of_bins
        if hfmg_quantize.HasField(ASYMMETRIC):
            act_params[ASYMMETRIC] = hfmg_quantize.asymmetric
        else:
            act_params[ASYMMETRIC] = None
        if hfmg_quantize.HasField(QUANT_GRANULARITY):
            act_params[QUANT_GRANULARITY] = hfmg_quantize.quant_granularity

    @staticmethod
    def _get_ifmr_config(config, act_params):
        """extract ifmr configs"""
        ifmr_quantize = config.ifmr_quantize
        act_params['act_algo'] = IFMR
        act_params[MAX_PERCENTILE] = \
            proto_float_to_python_float(ifmr_quantize.max_percentile)
        act_params[MIN_PERCENTILE] = \
            proto_float_to_python_float(ifmr_quantize.min_percentile)
        act_params[SEARCH_RANGE] = [
            proto_float_to_python_float(ifmr_quantize.search_range_start),
            proto_float_to_python_float(ifmr_quantize.search_range_end)
        ]
        act_params[SEARCH_STEP] = proto_float_to_python_float(
            ifmr_quantize.search_step)
        if ifmr_quantize.HasField(ASYMMETRIC):
            act_params[ASYMMETRIC] = ifmr_quantize.asymmetric
        else:
            act_params[ASYMMETRIC] = None
        if ifmr_quantize.HasField(QUANT_GRANULARITY):
            act_params[QUANT_GRANULARITY] = ifmr_quantize.quant_granularity

    def get_proto_global_config(self):
        """parse global config"""
        global_config = OrderedDict()
        if hasattr(self.proto_config, BATCH_NUM):
            global_config[BATCH_NUM] = self._get_batch_num()
            if global_config[BATCH_NUM] < 1:
                raise ValueError("batch_num({}) should be greater than zero".format(global_config[BATCH_NUM]))
        if hasattr(self.proto_config, ACTIVATION_OFFSET):
            global_config[ACTIVATION_OFFSET] = self._get_activation_offset()
        return global_config

    def get_quant_config(self, common_config_key):
        """get data config"""
        if self.proto_config.HasField(common_config_key) and \
            getattr(self.proto_config, common_config_key).HasField('calibration_config'):
            data_quant_config = getattr(self.proto_config, common_config_key).calibration_config
            return QuantCalibrationProtoConfig._get_calibration_config(data_quant_config)
        else:
            return OrderedDict()

    def get_quant_layers(self, quant_method):
        """
        Function: Get the list of quant skip layers.
        Params: None
        Return: a list, contain skip layers.
        """
        common_config_key = QUANT_METHOD_INFO_MAP.get(quant_method).common_config_key
        if not self.proto_config.HasField(common_config_key):
            return list()
        quant_layers = list(getattr(self.proto_config, common_config_key).quant_layers)
        repeated_layers = find_repeated_items(quant_layers)
        check_no_repeated(repeated_layers, QUANT_METHOD_INFO_MAP.get(quant_method).layer_key)
        return quant_layers

    def get_override_layers(self):
        """
        Function: Get the list of quant calibration override layers.
        Params: None
        Returns: a list, quant calibration override layers
        """
        override_layers = defaultdict(list)
        for config in self.proto_config.override_layers_configs:
            for quant_info in QUANT_METHOD_INFO_MAP.values():
                if hasattr(config, quant_info.config_key):
                    override_layers[quant_info.layer_key].append(config.layer_name)
            # record override_layers config
            self.override_layer_proto[config.layer_name] = config

        return override_layers

    def read_override_layer_config(self, override_layer, config_key):
        """ Read the config of one override_layer. """
        config = self.override_layer_proto.get(override_layer)
        return QuantCalibrationProtoConfig._get_calibration_config(getattr(config, config_key))

    def _get_batch_num(self):
        """ get batch_num from proto """
        if self.proto_config.HasField(BATCH_NUM):
            return self.proto_config.batch_num
        return 1

    def _get_activation_offset(self):
        """ get batch activation_offset from proto """
        if self.proto_config.HasField(ACTIVATION_OFFSET):
            return self.proto_config.activation_offset
        return None
