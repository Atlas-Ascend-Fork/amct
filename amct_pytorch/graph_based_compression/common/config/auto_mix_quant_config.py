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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from google.protobuf import text_format

from ...proto.basic_info_pb2 import AutoMixedPrecisionConfig
from ...proto.basic_info_pb2 import QuantBitLimit
from ..utils import vars_util
from ..utils import util

COMPRESSION_RATION_RANGE = [1, 16]


class AutoMixedPrecisionConfigHelper:
    """ Help to process AutoMixedPrecisionConfig."""
    bit_mapping = {
        QuantBitLimit.DataType.FLOAT: '16',
        QuantBitLimit.DataType.INT8: '8',
        QuantBitLimit.DataType.INT4: '4'
    }

    def __init__(self, config_proto):
        """ Init func.

        Args:
            config_proto (AutoMixedPrecisionConfig): config proto to process
        """
        self.config_proto = config_proto

    @property
    def compress_ratio(self):
        """ get compress_ratio. """
        return util.proto_float_to_python_float(self.config_proto.compress_ratio)

    @property
    def quant_bit_limit(self):
        """ get quant_bit_limit. """
        search_limit = {}
        for limit in self.config_proto.quant_bit_limit:
            search_limit[limit.layer_name] = [self.bit_mapping.get(data_type) for data_type in limit.data_range]
        return search_limit

    @property
    def test_iteration(self):
        """ get test_iteration. """
        if self.config_proto.HasField('test_iteration'):
            test_iteration = self.config_proto.test_iteration
        else:
            test_iteration = -1
        return test_iteration

    @property
    def ptq_cfg(self):
        """ get ptq_cfg. """
        config_defination = self.config_proto.ptq_cfg
        # do not configure in amc cfg returns ''
        if config_defination == '':
            return None
        # otherwise
        return os.path.realpath(config_defination)

    @property
    def override_qat_cfg(self):
        """ get override_qat_cfg. """
        config_defination = self.config_proto.override_qat_cfg
        if config_defination == '':
            return None
        return os.path.realpath(config_defination)

    @staticmethod
    def check_search_limit(search_limit, quant_layer2type, mode):
        """ check search_limit with giving quant_layer2type and quant mode.

        Args:
            search_limit (dict): key is layer and value is list of support quant bit.
            quant_layer2type (dict): key is layer and value is type.
            mode (string): how to do quant, ptq or qat.

        Raises:
            RuntimeError: layer in search_limit cannot do quant for it not in quant_layer2type.
            ValueError: layer cannot do quant in some bit.
        """
        for layer in search_limit:
            if layer not in quant_layer2type:
                raise RuntimeError('cannot search layer {} for it not in graph/model or cannot do quant ' \
                                   'in mode of {}.'.format(layer, mode))
            # AvgPool cannot support 4 bit.
            if quant_layer2type.get(layer) in ['AvgPool', 'AvgPool2d'] and \
                str(vars_util.INT4_BIT) in search_limit.get(layer):
                raise ValueError('{} with type {} cannot quant to 4 bit.'.format(layer, quant_layer2type.get(layer)))
    
    @staticmethod
    def _check_compress_ratio(value):
        """ Check compress_ratio in proto(AutoMixedPrecisionConfig).

        Args:
            value (float): compress_ratio's value

        Raises:
            ValueError: value is invalid.
        """
        if value <= COMPRESSION_RATION_RANGE[0] or value > COMPRESSION_RATION_RANGE[1]:
            raise ValueError("Can not support compression_ratio not in ({}, {}]"
                             .format(COMPRESSION_RATION_RANGE[0], COMPRESSION_RATION_RANGE[1]))

    @staticmethod
    def _check_file(file_name, item_name):
        """ check file from proto. """
        file_name = os.path.realpath(file_name)
        if not os.path.exists(file_name):
            raise ValueError("The {} ({} in AutoMixedPrecisionConfig) does not exist, please check the file path."
                             .format(file_name, item_name))

    @classmethod
    def read_file(cls, config_file):
        """ Read one file to parse AutoMixedPrecisionConfig.

        Args:
            config_file (string): file to read and parse.

        Raises:
            RuntimeError: content in file doesn't match AutoMixedPrecisionConfig.

        Returns:
            AutoMixedPrecisionConfig: proto parsed.
        """
        config_file = os.path.realpath(config_file)
        config_proto = AutoMixedPrecisionConfig()
        with open(config_file, 'r') as fid:
            pbtxt_string = fid.read()
            try:
                text_format.Merge(pbtxt_string, config_proto)
            except text_format.ParseError as e:
                raise RuntimeError(
                    "the config_file{%s} cannot be parsered, please ensure "\
                    "it matches with AutoMixedPrecisionConfig!"
                    % (config_file)) from e

        return config_proto

    def parse(self):
        """ check the content in config_proto. """
        for field in ['compress_ratio', 'test_iteration']:
            if not self.config_proto.HasField(field):
                raise ValueError("Cannot find essential field {} in AutoMixedPrecisionConfig.".format(field))

        self._check_compress_ratio(self.config_proto.compress_ratio)
        self._check_ptq_cfg()
        self._check_override_qat_cfg()

    def _check_ptq_cfg(self):
        """ check ptq_cfg in proto(AutoMixedPrecisionConfig). """
        config_defination = self.config_proto.ptq_cfg
        if config_defination != '':
            self._check_file(config_defination, 'ptq_cfg')

    def _check_override_qat_cfg(self):
        """ check override_qat_cfg in proto(AutoMixedPrecisionConfig). """
        config_defination = self.config_proto.override_qat_cfg
        if config_defination != '':
            self._check_file(config_defination, 'override_qat_cfg')
