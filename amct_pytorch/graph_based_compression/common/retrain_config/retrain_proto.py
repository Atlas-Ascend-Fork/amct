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
from enum import unique, Enum
from google.protobuf import text_format

from ...proto import retrain_config_pb2
from ..utils.util import proto_float_to_python_float
from ..utils.vars_util import INT4, INT8, INT16
from ...utils.log import LOGGER

DT_UNKNOWN = 'UNKNOWN'
SELECTIVE_PRUNE_M4N2 = 'M4N2'
FILTER = 'filter_prune'
SELECTIVE = 'selective_prune'
NO_PRUNE = 'no_prune_enable'
CHANNEL_WISE = 'channel_wise'
ALGO = 'algo'
BATCH_NUM = 'batch_num'
DST_TYPE = 'dst_type'
ULQ_RETRAIN = 'ulq_retrain'
FAKEQUANT_PRECISION_MODE = 'fakequant_precision_mode'


@unique
class FakequantPrecisionMode(Enum):
    DEFAULT = 0
    FORCE_FP16_QUANT = 1


class RetrainProtoConfig():
    """
    Function: Cope with simple config file from proto.
    APIs: get_skip_layers,
          get_override_layers,
          read_override_config,
          read_conv_calibration_config,
          read_fc_calibration_config
    """
    def __init__(self, config_file, capacity=None):
        self.config_file = config_file
        self.proto_config = self._read()
        self.override_config_proto = {}
        self.override_type_proto = {}
        self.capacity = capacity
        if self.capacity is not None:
            self.quantizable_type = self.capacity.get_value('RETRAIN_TYPES')

        self.has_prune = False
        if hasattr(self.proto_config, 'prune_config'):
            self.has_prune = True

    @staticmethod
    def __check_proto_override_types_enable(override_layer_types):
        """
        Funtion:
        check proto override layer types enable.
        Paramters:
        override_layer_types: proto, proto's override_layer_types.
        Return: (layer_retrain, layer_prune)
        layer_retrain: bool, T / F.
        False:
        1. have no override_layer_types.
        2. have no retrain data and weight.
        True: otherwise.
        layer_prune: bool, T / F.
        False:
        1. have no override_layer_configs.
        2. have no prune_config in override_layer_types.
        True: otherwise.
        """
        if not override_layer_types:
            return (False, False)
        layer_retrain = False
        layer_prune = False
        for config in override_layer_types:
            if config.HasField('retrain_data_quant_config') or \
                config.HasField('retrain_weight_quant_config'):
                layer_retrain = True
            if hasattr(config, 'prune_config') and config.HasField('prune_config'):
                layer_prune = True
        return (layer_retrain, layer_prune)

    @staticmethod
    def __check_proto_override_configs_enable(override_layer_configs):
        """
        Funtion:
        check proto override layer configs enable.
        Paramters:
        override_layer_configs: proto, proto's override_layer_configs.
        Return: (layer_retrain, layer_prune)
        layer_retrain: bool, T / F.
        False:
        1. have no override_layer_configs.
        2. have no retrain data and weight.
        True: otherwise.
        layer_prune: bool, T / F.
        False:
        1. have no override_layer_configs.
        2. have no prune_config in override_layer_configs.
        True: otherwise.
        """
        if not override_layer_configs:
            return (False, False)
        layer_retrain = False
        layer_prune = False
        for config in override_layer_configs:
            if config.HasField('retrain_data_quant_config') or \
                config.HasField('retrain_weight_quant_config'):
                layer_retrain = True
            if hasattr(config, 'prune_config') and config.HasField('prune_config'):
                layer_prune = True
        return (layer_retrain, layer_prune)

    @staticmethod
    def _parse_data_type(data_type_proto):
        """
        Funtion: parse DataType in retrain.proto trans to str
        Paramters:
            data_type_proto: DataType in retrain.proto
                0: INT4, 1: INT8, 2: INT16
        Return: data_type: str
        """
        if data_type_proto == 0:
            return INT4
        if data_type_proto == 1:
            return INT8
        if data_type_proto == 2:
            return INT16
        return DT_UNKNOWN

    @staticmethod
    def _parse_n_out_of_m_type(data_type_proto):
        if data_type_proto == 0:
            return SELECTIVE_PRUNE_M4N2
        raise ValueError('n_out_of_m_type is not M4N2, '
                         'only M4N2 is support in n_out_of_m_type.')

    @staticmethod
    def _get_prune_config(config_item):
        prune_param = {}
        if not config_item.ListFields():
            #prune config is empty
            return prune_param

        algo_config = {}
        if config_item.HasField('filter_pruner'):
            algo_config['prune_type'] = "filter_prune"
            filter_pruner = config_item.filter_pruner
            if not filter_pruner.HasField('balanced_l2_norm_filter_prune'):
                raise ValueError('only balanced_l2_norm_filter_prune is support, '
                                ' but balanced_l2_norm_filter_prune is not found.')
            algo_config[ALGO] = "balanced_l2_norm_filter_prune"
            if not filter_pruner.balanced_l2_norm_filter_prune.HasField('prune_ratio'):
                raise ValueError("prune_ratio is required for balanced_l2_norm_filter_prune, please add it.")
            algo_config['prune_ratio'] = proto_float_to_python_float(
                filter_pruner.balanced_l2_norm_filter_prune.prune_ratio)
            if filter_pruner.balanced_l2_norm_filter_prune.HasField('ascend_optimized'):
                algo_config['ascend_optimized'] = \
                    filter_pruner.balanced_l2_norm_filter_prune.ascend_optimized
        elif hasattr(config_item, 'n_out_of_m_pruner') and config_item.HasField('n_out_of_m_pruner'):
            algo_config['prune_type'] = "selective_prune"
            n_out_of_m_pruner = config_item.n_out_of_m_pruner
            if not n_out_of_m_pruner.HasField('l1_selective_prune'):
                raise ValueError('only l1_selective_prune is support, '
                                ' but l1_selective_prune is not found.')
            algo_config[ALGO] = "l1_selective_prune"
            algo_config['n_out_of_m_type'] = \
                RetrainProtoConfig._parse_n_out_of_m_type(n_out_of_m_pruner.l1_selective_prune.n_out_of_m_type)
            algo_config['update_freq'] = n_out_of_m_pruner.l1_selective_prune.update_freq
        prune_param['regular_prune_config'] = algo_config

        return prune_param

    def parse_proto_enable(self):
        """
        Function:
        get proto enable from config defination. Both, either, neither.
        Retrain with empty config defination will get config.
        Compressed with empty config defination will get error.
        So can not add attr in proto.has_retrain.
        """
        proto_enable_quant = True
        proto_enable_prune = True
        if not self.proto_config.HasField('prune_config'):
            _, proto_enable_prune = self.__check_proto_override_enable()
        if not self.proto_config.HasField('retrain_data_quant_config') and \
            not self.proto_config.HasField('retrain_weight_quant_config'):
            proto_enable_quant, _ = self.__check_proto_override_enable()
        # neither
        if not proto_enable_quant and not proto_enable_prune:
            raise ValueError("proto can not neither retrain nor prune.")
        return proto_enable_quant, proto_enable_prune

    def parse_prune_type(self):
        """
        Function:
        get prune type from config defination.
        """
        if not self.proto_config.HasField('prune_config'):
            return NO_PRUNE
        if self.proto_config.prune_config.HasField('filter_pruner'):
            prune_type = FILTER
        elif self.proto_config.prune_config.HasField('n_out_of_m_pruner'):
            prune_type = SELECTIVE
        return prune_type

    def get_proto_config(self):
        """parse proto config"""
        global_config = OrderedDict()
        if hasattr(self.proto_config, BATCH_NUM):
            global_config[BATCH_NUM] = self._get_batch_num()
        if hasattr(self.proto_config, FAKEQUANT_PRECISION_MODE):
            global_config[FAKEQUANT_PRECISION_MODE] = self._get_fakequant_mode()
        return global_config

    def get_retrain_data_quant_config(self):
        """ Get default data quant params """
        data_quant_config = self.proto_config.retrain_data_quant_config
        return self._get_retrain_data_config(data_quant_config)

    def get_retrain_weights_config(self):
        """get weights config"""
        retrain_weight_config = self.proto_config.retrain_weight_quant_config
        return self._get_retrain_weight_config(retrain_weight_config)

    def get_skip_layers(self):
        """ Get the list of global skip_layers. """
        skip_layers = list(self.proto_config.skip_layers)
        repeated_layers = find_repeated_items(skip_layers)
        if repeated_layers:
            LOGGER.logw("Please delete repeated items in skip_layers, "
                        "repeated items are %s " % (repeated_layers),
                        module_name="Configuration")
        skip_layers = list(set(skip_layers))
        return skip_layers

    def get_skip_layer_types(self):
        """
        Function: Get the list of global skip_layer_types.
        Params: None
        Returns:
            skip_layer_types, a list
        """
        skip_layer_types = list(self.proto_config.skip_layer_types)
        repeated_types = find_repeated_items(skip_layer_types)
        if repeated_types:
            LOGGER.logw("Please delete repeated items in skip_layer_types, "
                        "repeated items are %s " % (repeated_types),
                        module_name="Configuration")
        skip_layer_types = list(set(skip_layer_types))

        # handle ms if RETRAIN_MAP_TYPES exists
        if self.capacity.get_value('RETRAIN_MAP_TYPES') is None:
            return skip_layer_types
        return self._transform_layer_types(skip_layer_types)

    def get_quant_skip_layers(self):
        """
        Function:
        Get the list of quant skip layers.
        Params:
        None
        Return:
        a list, contain the global skip layers and its own skip layers.
        """
        # quant's own skip layers.
        quant_skip_layers = list(self.proto_config.quant_skip_layers)
        repeated_layers = find_repeated_items(quant_skip_layers)
        if repeated_layers:
            LOGGER.logw("Please delete repeated items in quant_skip_layers, "
                        "repeated items are %s " % (repeated_layers),
                        module_name="Configuration")
        quant_skip_layers = list(set(quant_skip_layers))
        # extend the global skip layers
        quant_skip_layers.extend(self.get_skip_layers())
        # remove the redundant layers
        quant_skip_layers = list(set(quant_skip_layers))
        return quant_skip_layers

    def get_quant_skip_types(self):
        """
        Function:
        Get the list of quant skip types.
        Params:
        None.
        Return:
        a list, contain the gloabl skip types and its own skip types.
        """
        # quant's own skip types.
        quant_skip_types = list(self.proto_config.quant_skip_types)
        repeated_types = find_repeated_items(quant_skip_types)
        if repeated_types:
            LOGGER.logw("Please delete repeated items in quant_skip_types, "
                        "repeated items are %s " % (repeated_types),
                        module_name="Configuration")
        quant_skip_types = list(set(quant_skip_types))
        # handle ms if RETRAIN_MAP_TYPES exists
        if self.capacity.get_value('RETRAIN_MAP_TYPES') is not None:
            quant_skip_types = self._transform_layer_types(quant_skip_types)
        # extend the global skip types
        quant_skip_types.extend(self.get_skip_layer_types())
        # remove the redundant types
        quant_skip_types = list(set(quant_skip_types))
        return quant_skip_types

    def get_override_layers(self):
        """
        Get the list of override_layers.
        Params: None
        Returns:
            override_layers, a list for override
            retrain_override_layers, a list for override retrain
            regular_prune_override_layers, a list for override prune
        """
        # get retrain_override_layers
        retrain_override_layers = []
        prune_override_layers = {
            FILTER: [],
            SELECTIVE: []
        }
        override_layers = []
        for config in self.proto_config.override_layer_configs:
            override_layers.append(config.layer_name)
            if config.HasField('retrain_data_quant_config') or \
                config.HasField('retrain_weight_quant_config'):
                retrain_override_layers.append(config.layer_name)
            if self.has_prune and config.HasField('prune_config'):
                if config.prune_config.HasField('filter_pruner'):
                    prune_override_layers.get(FILTER).append(config.layer_name)
                elif config.prune_config.HasField('n_out_of_m_pruner'):
                    prune_override_layers.get(SELECTIVE).append(config.layer_name)
        # check repated items in retrain_override_layers
        repeated = find_repeated_items(override_layers)
        if repeated:
            raise ValueError(
                "Please delete repeated items in retrain_override_layers,  "
                "repeated items are %s " % (repeated))
        for config in self.proto_config.override_layer_configs:
            self.override_config_proto[config.layer_name] = config

        return override_layers, retrain_override_layers, prune_override_layers

    def read_override_config(self, override_layer):
        """ Read the config of one override_layer. """
        layer = self.override_config_proto.get(override_layer)
        retrain_data_params = self._get_retrain_data_config(
            layer.retrain_data_quant_config)
        retrain_weight_params = self._get_retrain_weight_config(
            layer.retrain_weight_quant_config)
        if self.has_prune:
            prune_params = self._get_prune_config(layer.prune_config)
        else:
            prune_params = {}
        return retrain_data_params, retrain_weight_params, prune_params

    def get_override_layer_types(self):
        """ Get the list of override_layer_types. """
        # get override_layer_types
        retrain_override_types = []
        regular_prune_override_types = {
            FILTER: [],
            SELECTIVE: []
        }
        override_types = []
        for config in self.proto_config.override_layer_types:
            override_types.append(config.layer_type)
            if config.HasField('retrain_data_quant_config') or \
                config.HasField('retrain_weight_quant_config'):
                retrain_override_types.append(config.layer_type)
            if self.has_prune and config.HasField('prune_config'):
                if config.prune_config.HasField('filter_pruner'):
                    regular_prune_override_types.get(FILTER).append(config.layer_type)
                else:
                    regular_prune_override_types.get(SELECTIVE).append(config.layer_type)
        # check repated items in override_types
        repeated_types = find_repeated_items(override_types)
        if repeated_types:
            raise ValueError(
                "Please delete repeated items in override_layer_types,  "
                "repeated items are %s " % (repeated_types))

        # handle if RETRAIN_MAP_TYPES not exists
        if self.capacity.get_value('RETRAIN_MAP_TYPES') is None:
            # match name and config
            for config in self.proto_config.override_layer_types:
                self.override_type_proto[config.layer_type] = config
            return override_types, retrain_override_types, \
                regular_prune_override_types

        # handle ms if RETRAIN_MAP_TYPES exists
        return self.get_trans_override_layer_types(override_types, retrain_override_types, \
            regular_prune_override_types)

    def get_trans_override_layer_types(self, override_types, retrain_override_types, regular_prune_override_types):
        """ Transform the list of override_layer_types according to RETRAIN_MAP_TYPES. """
        for config in self.proto_config.override_layer_types:
            layer_type = config.layer_type
            layer_type_list = self._transform_layer_types(layer_type)
            for layer_type in layer_type_list:
                if layer_type not in self.quantizable_type:
                    raise ValueError("Layer type {} does not support "
                                     "quantize.".format(layer_type))
                self.override_type_proto[layer_type] = config

        trans_retrain_override_types = []
        trans_regular_prune_override_types = {
            FILTER: [],
            SELECTIVE: []
        }
        trans_override_types = []
        for layer_type in override_types:
            trans_override_types.extend(self._transform_layer_types(layer_type))
        for layer_type in retrain_override_types:
            trans_retrain_override_types.extend(self._transform_layer_types(layer_type))
        for layer_type in regular_prune_override_types.get(FILTER):
            trans_regular_prune_override_types.get(FILTER).extend(self._transform_layer_types(layer_type))
        for layer_type in regular_prune_override_types.get(SELECTIVE):
            trans_regular_prune_override_types.get(SELECTIVE).extend(self._transform_layer_types(layer_type))

        return trans_override_types, trans_retrain_override_types, trans_regular_prune_override_types

    def read_override_type_config(self, override_layer_type):
        """ Read the config of one override_layer. """
        layer_type = self.override_type_proto.get(override_layer_type)
        retrain_data_params = self._get_retrain_data_config(
            layer_type.retrain_data_quant_config)
        retrain_weight_params = self._get_retrain_weight_config(
            layer_type.retrain_weight_quant_config)
        if self.has_prune:
            prune_params = self._get_prune_config(layer_type.prune_config)
        else:
            prune_params = {}
        return retrain_data_params, retrain_weight_params, prune_params

    def get_prune_config(self):
        """
        Get prune config params
        Params: None
        Return: a dict
        """
        if self.has_prune:
            return self._get_prune_config(self.proto_config.prune_config)

        return {}

    def get_regular_prune_skip_layers(self):
        """
        Get the list of prune_skip_layers.
        Params: None
        Return: a list, contain the global skip layers and its own skip layers.
        """
        # regular prune's own skip layers.
        prune_skip_layers = list(self.proto_config.regular_prune_skip_layers)
        repeated_layers = find_repeated_items(prune_skip_layers)
        if repeated_layers:
            LOGGER.logw("Please delete repeated items in regular_prune_skip_layers, "
                        "repeated items are %s " % (repeated_layers),
                        module_name="Configuration")
        prune_skip_layers = list(set(prune_skip_layers))
        # extend the global skip layers.
        prune_skip_layers.extend(self.get_skip_layers())
        # remove the redundant layers.
        prune_skip_layers = list(set(prune_skip_layers))
        return prune_skip_layers

    def get_regular_prune_skip_types(self):
        """
        Get the list of regular_prune_skip_types.
        Params: None
        Return: a list, contain the global skip types and its own skip types.
        """
        # regular prune's own skip types.
        prune_skip_types = list(self.proto_config.regular_prune_skip_types)
        repeated_types = find_repeated_items(prune_skip_types)
        if repeated_types:
            LOGGER.logw("Please delete repeated items in regular_prune_skip_types, "
                        "repeated items are %s " % (repeated_types),
                        module_name="Configuration")
        prune_skip_types = list(set(prune_skip_types))
        # handle ms if RETRAIN_MAP_TYPES exists
        if self.capacity.get_value('RETRAIN_MAP_TYPES') is not None:
            prune_skip_types = self._transform_layer_types(prune_skip_types)
        # extend the global skip types
        prune_skip_types.extend(self.get_skip_layer_types())
        # remove the redundant types
        prune_skip_types = list(set(prune_skip_types))
        return prune_skip_types

    def check_field(self, enable_retrain=True, enable_prune=True):
        """
        Check whether proto field is expected or not
        Params: enable_retrain, a bool
        Return: None
        """
        self._check_retrain_field(enable_retrain)
        self._check_prune_field(enable_prune)
        if enable_prune and not self.has_prune:
            raise ValueError("config proto file has no prune to enable prune.")

    def _check_retrain_field(self, enable_retrain):
        """ Check whether retrain field is expected or not
        Params: enable_retrain, a bool
        Params: enable_prune, a bool
        """
        if not enable_retrain:
            # batch_num has default value, so do not check
            retrain_fields = ['retrain_data_quant_config',
                             'retrain_weight_quant_config',
                             'quant_skip_layers', 'quant_skip_types']
            for item in retrain_fields:
                if not is_empty(getattr(self.proto_config, item)):
                    raise ValueError("unexpected {} in config_defination if retrain is disable.".format(item))
            override_fields = ['retrain_data_quant_config',
                               'retrain_weight_quant_config']
            for config in self.proto_config.override_layer_configs:
                for item in override_fields:
                    if getattr(config, item).ListFields():
                        raise ValueError("unexpected {} for override_layer_configs in "
                            "config_defination if retrain is disable.".format(item))
            for config in self.proto_config.override_layer_types:
                for item in override_fields:
                    if getattr(config, item).ListFields():
                        raise ValueError("unexpected {} for override_layer_types in"
                            " config_defination if retrain is disable.".format(item))

    def _check_prune_field(self, enable_prune):
        """ Check whether prune field is expected or not
        Params: enable_prune, a bool
        """
        if not enable_prune and self.has_prune:
            retrain_fields = ['prune_config',
                             'regular_prune_skip_layers', 'regular_prune_skip_types']
            for item in retrain_fields:
                if not is_empty(getattr(self.proto_config, item)):
                    raise ValueError("unexpected {} in config_defination if prune is disable.".format(item))
            override_fields = ['prune_config']

            for config in self.proto_config.override_layer_configs:
                for item in override_fields:
                    if getattr(config, item).ListFields():
                        raise ValueError("unexpected {} for override_layer_configs in "
                            "config_defination if prune is disable.".format(item))
            for config in self.proto_config.override_layer_configs:
                for item in override_fields:
                    if getattr(config, item).ListFields():
                        raise ValueError("unexpected {} for override_layer_types in "
                            "config_defination if prune is disable.".format(item))

    def _get_batch_num(self):
        if self.proto_config.HasField(BATCH_NUM):
            return self.proto_config.batch_num
        return None

    def _get_fakequant_mode(self):
        precision_mode_define = {
            FakequantPrecisionMode.DEFAULT.value: 'DEFAULT',
            FakequantPrecisionMode.FORCE_FP16_QUANT.value: 'FORCE_FP16_QUANT'
        }
        if self.proto_config.HasField(FAKEQUANT_PRECISION_MODE):
            return precision_mode_define.get(self.proto_config.fakequant_precision_mode)
        return 'DEFAULT'

    def _get_mapped_type(self, layer_type, map_types):
        if layer_type not in map_types:
            raise ValueError(
                'Unrecognized layer type:{}, only support {}'.format(
                    layer_type, map_types))
        return self.quantizable_type[map_types.index(layer_type)]

    def _transform_layer_types(self, layer_types):
        map_types = self.capacity.get_value('RETRAIN_MAP_TYPES')
        if map_types is None:
            return layer_types

        ret = []
        if not isinstance(layer_types, list):
            if layer_types == 'nn.Conv2D':
                return ['Conv2D', 'DepthwiseConv2dNative']
            ret.append(self._get_mapped_type(layer_types, map_types))
            return ret

        for item in layer_types:
            if item == 'nn.Conv2D':
                ret += ['Conv2D', 'DepthwiseConv2dNative']
            else:
                ret.append(self._get_mapped_type(item, map_types))
        return ret
    
    def _check_retrain_data_type(self, data_type):
        """ check int4 retrain capacity and config"""
        int4_retrain_enable = False
        if self.capacity.get_value('INT4_RETRAIN') is not None:
            int4_retrain_enable = self.capacity.get_value(
                'INT4_RETRAIN')
        if not int4_retrain_enable and data_type == INT4:
            raise ValueError(
                "Int4 quantization aware training is not supported.")

    def _get_retrain_data_config(self, config_item):
        retrain_data_params = OrderedDict()
        retrain_data_params[ALGO] = 'ulq_quantize'
        retrain_data_config = config_item.ulq_quantize
        if retrain_data_config.HasField('clip_max_min'):
            if not retrain_data_config.clip_max_min.HasField('clip_max') or \
                not retrain_data_config.clip_max_min.HasField('clip_min'):
                raise ValueError(
                "clip_max and clip_min are both required.")
            clip_max = retrain_data_config.clip_max_min.clip_max
            clip_min = retrain_data_config.clip_max_min.clip_min
            retrain_data_params['clip_max'] = clip_max
            retrain_data_params['clip_min'] = clip_min
        if retrain_data_config.HasField('fixed_min'):
            retrain_data_params['fixed_min'] = retrain_data_config.fixed_min
        if hasattr(retrain_data_config, DST_TYPE):
            if retrain_data_config.HasField(DST_TYPE):
                retrain_data_params[
                    DST_TYPE] = RetrainProtoConfig._parse_data_type(
                        retrain_data_config.dst_type)
            else:
                # default value is INT8
                retrain_data_params[DST_TYPE] = INT8
            self._check_retrain_data_type(retrain_data_params[DST_TYPE])
        return retrain_data_params

    def _get_retrain_weight_config(self, config_item):
        retrain_weight_params = OrderedDict()
        if config_item.HasField('arq_retrain'):
            retrain_weight_params[ALGO] = 'arq_retrain'
            retrain_weight_config = config_item.arq_retrain
            if retrain_weight_config.HasField(CHANNEL_WISE):
                retrain_weight_params[
                    CHANNEL_WISE] = retrain_weight_config.channel_wise
            if hasattr(retrain_weight_config, DST_TYPE):
                if retrain_weight_config.HasField(DST_TYPE):
                    retrain_weight_params[
                        DST_TYPE] = RetrainProtoConfig._parse_data_type(
                            retrain_weight_config.dst_type)
                else:
                    retrain_weight_params[DST_TYPE] = INT8
                self._check_retrain_data_type(retrain_weight_params[DST_TYPE])
        elif hasattr(config_item, ULQ_RETRAIN) and config_item.HasField(ULQ_RETRAIN):
            retrain_weight_params[ALGO] = ULQ_RETRAIN
            retrain_weight_config = config_item.ulq_retrain

            if retrain_weight_config.HasField(CHANNEL_WISE):
                retrain_weight_params[
                    CHANNEL_WISE] = retrain_weight_config.channel_wise

            if retrain_weight_config.HasField(DST_TYPE):
                retrain_weight_params[
                    DST_TYPE] = RetrainProtoConfig._parse_data_type(
                        retrain_weight_config.dst_type)
            else:
                retrain_weight_params[DST_TYPE] = INT8
            self._check_retrain_data_type(retrain_weight_params[DST_TYPE])

        return retrain_weight_params

    def _read(self):
        """ Read config from config_file. """
        proto_config = retrain_config_pb2.AMCTRetrainConfig()
        with open(self.config_file, 'rb') as cfg_file:
            pbtxt_string = cfg_file.read()
            text_format.Merge(pbtxt_string, proto_config)

        return proto_config

    def __check_proto_override_enable(self):
        """
        Function:
        check proto override enable.
        only both False in layer config and layer type config return False.
        Return: (proto_enable_retrain, proto_enable_prune)
        proto_enable_retrain: bool, T / F
        True: proto has config about retrain.
        False: proto has no config about retrain.
        proto_enable_prune: bool, T / F
        True: proto has config about prune.
        False: proto has no config about prune.
        """
        override_layer_configs = self.proto_config.override_layer_configs
        override_layer_types = self.proto_config.override_layer_types

        layer_retrain, layer_prune = self.__check_proto_override_configs_enable(override_layer_configs)
        layer_types_retrain, layer_types_prune = self.__check_proto_override_types_enable(override_layer_types)

        return (layer_retrain or layer_types_retrain, layer_prune or layer_types_prune)


def find_repeated_items(item_list):
    '''find repeated items in a list '''
    repeat_items = set()
    for item in item_list:
        count = item_list.count(item)
        if count > 1:
            repeat_items.add(item)

    return list(repeat_items)


def is_empty(proto_field):
    """
    Check whether a proto field is empty
    Params: proto_field, a field to check
    Returns: bool, empty or not
    """
    try:
        fields = proto_field.ListFields()
        return not fields
    except AttributeError:
        if proto_field:
            return False
        return True
