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

from collections import OrderedDict
from enum import IntEnum, unique
from google.protobuf import text_format

from ....amct_pytorch.common.utils.util import find_repeated_items
from ....amct_pytorch.common.utils.vars_util import INT4, INT8
from ....amct_pytorch.proto import distill_config_pb2
from ....amct_pytorch.utils.log import LOGGER
from ....amct_pytorch.utils.vars import CHANNEL_WISE
from ....amct_pytorch.utils.vars import DST_TYPE
from ....amct_pytorch.utils.vars import BATCH_NUM
from ....amct_pytorch.common.config.proto_config import ProtoDataType


ULQ_DISTILL = 'ulq_distill'
GROUP_SIZE = 'group_size'


class DistillProtoConfig():
    """
    Function: Cope with simple config file from proto.
    APIs: parse_data_type
        get_proto_global_config
        get_distill_groups
        get_distill_data_quant_config
        get_distill_weight_quant_config
        get_quant_skip_layers
        get_quant_skip_layer_types
        get_override_layers
        get_override_layer_types
        read_override_layer_config
        read_override_type_config

    """
    def __init__(self, config_proto_file, capacity=None):
        self.proto_config = self.read(config_proto_file)
        self.override_layer_proto = {}
        self.override_type_proto = {}
        self.capacity = capacity
        if self.capacity is not None:
            self.distill_type = self.capacity.get_value('DISTILL_TYPES')

    @staticmethod
    def read(config_proto_file):
        """ Read config from config_proto_file. """
        proto_config = distill_config_pb2.AMCTDistillConfig()
        with open(config_proto_file, 'rb') as cfg_file:
            pbtxt_string = cfg_file.read()
            text_format.Merge(pbtxt_string, proto_config)

        return proto_config

    @staticmethod
    def parse_data_type(data_type_proto):
        dtype_define = {
            ProtoDataType.INT4.value: INT4,
            ProtoDataType.INT8.value: INT8,
        }
        if data_type_proto not in dtype_define:
            raise ValueError('Not support dtype of {}.'.format(data_type_proto))
        return dtype_define[data_type_proto]

    def get_proto_global_config(self):
        """parse proto config"""
        global_config = OrderedDict()
        if hasattr(self.proto_config, BATCH_NUM):
            global_config[BATCH_NUM] = self._get_batch_num()
            if global_config[BATCH_NUM] < 1:
                raise ValueError("batch_num({}) should be greater than zero".format(global_config[BATCH_NUM]))
        if hasattr(self.proto_config, GROUP_SIZE):
            global_config[GROUP_SIZE] = self._get_group_size()
            if global_config[GROUP_SIZE] < 1:
                raise ValueError("group_size({}) should be greater than zero".format(global_config[GROUP_SIZE]))
        if hasattr(self.proto_config, 'data_dump'):
            global_config['data_dump'] = self._get_data_dump()
        return global_config

    def get_distill_groups(self):
        """parse distill group config"""
        distill_groups = list()
        if hasattr(self.proto_config, 'distill_group'):
            distill_groups = self._get_distill_groups()
        return distill_groups

    def get_distill_data_quant_config(self):
        """get data config"""
        data_quant_config = self.proto_config.distill_data_quant_config
        return self._get_distill_data_config(data_quant_config)

    def get_distill_weight_quant_config(self):
        """get weights config"""
        distill_weight_config = self.proto_config.distill_weight_quant_config
        return self._get_distill_weight_config(distill_weight_config)

    def get_quant_skip_layers(self):
        """
        Function: Get the list of quant skip layers.
        Params: None
        Return: a list, contain skip layers.
        """
        quant_skip_layers = list(self.proto_config.quant_skip_layers)
        repeated_layers = find_repeated_items(quant_skip_layers)
        if repeated_layers:
            LOGGER.logw("Please delete repeated items in quant_skip_layers, "
                        "repeated items are {}.".format(repeated_layers),
                        module_name="DistillProto")
        # remove the redundant layers
        quant_skip_layers = list(set(quant_skip_layers))
        return quant_skip_layers

    def get_quant_skip_layer_types(self):
        """
        Function: Get the list of quant skip layer types.
        Params: None.
        Return: a list, contain skip types.
        """
        quant_skip_layer_types = list(self.proto_config.quant_skip_layer_types)
        repeated_types = find_repeated_items(quant_skip_layer_types)
        if repeated_types:
            LOGGER.logw("Please delete repeated items in quant_skip_layer_types, "
                        "repeated items are {}.".format(repeated_types),
                        module_name="DistillProto")
        # remove the redundant types
        quant_skip_layer_types = list(set(quant_skip_layer_types))
        return quant_skip_layer_types

    def get_override_layers(self):
        """
        Function: Get the list of distill override layers.
        Params: None
        Returns: a list, distill override layers
        """
        override_layers = list()
        for config in self.proto_config.distill_override_layers:
            override_layers.append(config.layer_name)
            # record override_layers config
            self.override_layer_proto[config.layer_name] = config
        # check repated items in distill_override_layers
        repeated_layers = find_repeated_items(override_layers)
        if repeated_layers:
            raise ValueError(
                "Please delete repeated items in distill_override_layers, "
                "repeated items are {}.".format(repeated_layers))

        return override_layers

    def get_override_layer_types(self):
        """
        Function: Get the list of distill override layer types.
        Params: None
        Returns: a list, distill override layer types
        """
        override_types = list()
        for config in self.proto_config.distill_override_layer_types:
            override_types.append(config.layer_type)
            # record override_types config
            self.override_type_proto[config.layer_type] = config
        # check repated items in distill_override_layer_types
        repeated_types = find_repeated_items(override_types)
        if repeated_types:
            raise ValueError(
                "Please delete repeated items in distill_override_layer_types,  "
                "repeated items are {}.".format(repeated_types))

        return override_types

    def read_override_layer_config(self, override_layer):
        """ Read the config of one override_layer. """
        config = self.override_layer_proto.get(override_layer)
        distill_data_params = self._get_distill_data_config(
            config.distill_data_quant_config)
        distill_weight_params = self._get_distill_weight_config(
            config.distill_weight_quant_config)

        return distill_data_params, distill_weight_params

    def read_override_type_config(self, override_layer_type):
        """ Read the config of one override_layer_type. """
        config = self.override_type_proto.get(override_layer_type)
        distill_data_params = self._get_distill_data_config(
            config.distill_data_quant_config)
        distill_weight_params = self._get_distill_weight_config(
            config.distill_weight_quant_config)

        return distill_data_params, distill_weight_params

    def _get_batch_num(self):
        if self.proto_config.HasField(BATCH_NUM):
            return self.proto_config.batch_num
        return 1

    def _get_group_size(self):
        if self.proto_config.HasField(GROUP_SIZE):
            return self.proto_config.group_size
        return 1

    def _get_data_dump(self):
        if self.proto_config.HasField('data_dump'):
            return self.proto_config.data_dump
        return False

    def _get_distill_groups(self):
        distill_groups = list()
        config_items = list(self.proto_config.distill_group)
        for item in config_items:
            distill_group = OrderedDict()
            distill_group['start_layer'] = item.start_layer_name
            distill_group['end_layer'] = item.end_layer_name
            distill_groups.append(distill_group)
        return distill_groups

    def _check_distill_data_type(self, data_type):
        """ check int4 distill capacity and config"""
        int4_distill_enable = False
        if self.capacity.get_value('INT4_DISTILL') is not None:
            int4_distill_enable = self.capacity.get_value(
                'INT4_DISTILL')
        if not int4_distill_enable and data_type == INT4:
            raise ValueError("Int4 distillation is not supported.")

    def _get_distill_data_config(self, config_item):
        distill_data_params = OrderedDict()
        distill_data_params['algo'] = 'ulq_quantize'
        distill_data_config = config_item.ulq_quantize
        if distill_data_config.HasField('clip_max_min'):
            clip_max = distill_data_config.clip_max_min.clip_max
            clip_min = distill_data_config.clip_max_min.clip_min
            distill_data_params['clip_max'] = clip_max
            distill_data_params['clip_min'] = clip_min
        if distill_data_config.HasField('fixed_min'):
            distill_data_params['fixed_min'] = distill_data_config.fixed_min
        if hasattr(distill_data_config, DST_TYPE):
            if distill_data_config.HasField(DST_TYPE):
                distill_data_params[DST_TYPE] = self.parse_data_type(
                    distill_data_config.dst_type)
            else:
                # default value is INT8
                distill_data_params[DST_TYPE] = INT8
            self._check_distill_data_type(distill_data_params[DST_TYPE])
        return distill_data_params

    def _get_distill_weight_config(self, config_item):
        distill_weight_params = OrderedDict()
        if config_item.HasField('arq_distill'):
            distill_weight_params['algo'] = 'arq_distill'
            distill_weight_config = config_item.arq_distill
            if distill_weight_config.HasField(CHANNEL_WISE):
                distill_weight_params[
                    CHANNEL_WISE] = distill_weight_config.channel_wise
            if hasattr(distill_weight_config, DST_TYPE):
                if distill_weight_config.HasField(DST_TYPE):
                    distill_weight_params[
                        DST_TYPE] = self.parse_data_type(
                            distill_weight_config.dst_type)
                else:
                    distill_weight_params[DST_TYPE] = INT8
                self._check_distill_data_type(distill_weight_params[DST_TYPE])
        elif hasattr(config_item, ULQ_DISTILL) and config_item.HasField(ULQ_DISTILL):
            distill_weight_params['algo'] = ULQ_DISTILL
            distill_weight_config = config_item.ulq_distill

            if distill_weight_config.HasField(CHANNEL_WISE):
                distill_weight_params[
                    CHANNEL_WISE] = distill_weight_config.channel_wise

            if distill_weight_config.HasField(DST_TYPE):
                distill_weight_params[
                    DST_TYPE] = self.parse_data_type(
                        distill_weight_config.dst_type)
            else:
                distill_weight_params[DST_TYPE] = INT8
            self._check_distill_data_type(distill_weight_params[DST_TYPE])

        return distill_weight_params
