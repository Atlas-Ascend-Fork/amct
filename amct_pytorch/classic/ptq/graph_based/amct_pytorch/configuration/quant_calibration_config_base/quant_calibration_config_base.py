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
import json
import copy
from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict

import torch
from ....amct_pytorch.common.utils.files import save_to_json
from ....amct_pytorch.common.utils.util import check_no_repeated
from ....amct_pytorch.common.utils.util import find_repeated_items
from ....amct_pytorch.utils.log import LOGGER
from ....amct_pytorch.capacity import CAPACITY
from ....amct_pytorch.utils.model_util import ModuleHelper
from ....amct_pytorch.common.config.config_base import ConfigBase
from ....amct_pytorch.common.utils.check_params import check_params

from ....amct_pytorch.configuration.quant_calibration_config_base.quant_calibration_proto \
    import QuantCalibrationProtoConfig, QUANT_METHOD_INFO_MAP
from ....amct_pytorch.configuration.quant_calibration_config_base.quant_calibration_field \
    import QuantCalibrationConfigRoot
from ....amct_pytorch.utils.vars import ASYMMETRIC, SEARCH_RANGE, BATCH_NUM

LAYER_CHECK_TYPE = {'kv_cache_quant_layers': CAPACITY.get_value('KV_CACHE_QUANTIZE_TYPES')}
MODULE_NAME = 'QuantCalibrationConfigBase'


class QuantCalibrationConfigBase():
    '''kv_cache config base'''

    def __init__(self):
        ''' inner method '''
        self.config_root = QuantCalibrationConfigRoot(ModuleHelper, CAPACITY)

    @staticmethod
    def check_quant_layers(model, quant_layers):
        ''' check wheter user's quant_layers input is valid'''
        for quant_method, layer_names in quant_layers.items():
            QuantCalibrationConfigBase.check_quant_layer_info_legality(
                model, quant_method, layer_names)

    @staticmethod
    def check_quant_layer_info_legality(model, quant_method, layer_names):
        '''check quant layers'''
        repeated_items = find_repeated_items(layer_names)
        check_no_repeated(repeated_items, quant_method)
        layer_type = ModuleHelper.get_name_type_dict(model)
        for layer in layer_names:
            if layer not in layer_type:
                raise ValueError(
                    "Layer {} does not exist in the model.".format(layer))
            if layer_type.get(layer) not in LAYER_CHECK_TYPE.get(quant_method):
                raise ValueError('The type of Layer {} is {}, it does not support in {}.'.format(
                    layer, layer_type.get(layer), quant_method))

    @staticmethod
    def add_global_to_layer(quant_config):
        '''add global quantize parameter to each layer'''
        for layer_name in quant_config:
            if not isinstance(quant_config.get(layer_name), dict):
                continue
            # 1.1 split search_range
            config_key = 'kv_data_quant_config'
            act_config = quant_config.get(layer_name).get(config_key)
            # default act calibration algo is ifmr
            act_algo = act_config.get('act_algo', 'ifmr')
            if act_algo == 'ifmr':
                act_config["search_range_start"] = \
                    act_config.get(SEARCH_RANGE)[0]
                act_config["search_range_end"] = \
                    act_config.get(SEARCH_RANGE)[1]
                del act_config[SEARCH_RANGE]
            # 1.2 add activation_offset
            if quant_config.get(layer_name).get(config_key).get(ASYMMETRIC) is None:
                with_offset = quant_config.get('activation_offset')
            else:
                with_offset = quant_config.get(
                    layer_name).get(config_key).get(ASYMMETRIC)
            act_config['with_offset'] = with_offset
            if quant_config.get(BATCH_NUM) is not None:
                act_config[BATCH_NUM] = quant_config.get(BATCH_NUM)

    @staticmethod
    def get_quant_layer_config(layer_name, quant_calibration_config):
        ''' get quant detail config on layer name '''
        if layer_name not in quant_calibration_config:
            LOGGER.logd(
                "layer {} is disabled for quantization.".format(layer_name))
            return None
        layer_config = quant_calibration_config.get(layer_name)
        return layer_config

    @staticmethod
    def convert_quant_layer_format(quant_layers, model):
        '''
        add layer type info for quant_layers map. 
        target format is {quant_method: ((layer_name, layer_type),)}
        '''
        quant_layer_type_map = defaultdict(dict)
        layer_type_map = ModuleHelper.get_name_type_dict(model)
        for quant_method, layer_list in quant_layers.items():
            for layer in layer_list:
                if layer not in layer_type_map:
                    raise RuntimeError(
                        'Layer {} is not in quantized model. Please check your configuration'.format(layer))
                quant_layer_type_map[quant_method][layer] = layer_type_map.get(
                    layer)
        return quant_layer_type_map
    
    @staticmethod
    def _del_reduant_config(ordered_config):
        ''' del no-key config '''
        check_list = list(ordered_config.keys())
        for key in check_list:
            if isinstance(ordered_config[key], dict):
                QuantCalibrationConfigBase._del_reduant_config(
                    ordered_config[key])
            if ordered_config[key] is None:
                del ordered_config[key]

    @staticmethod
    def _generate_layer_config(model, proto):
        """
        Function: Generate layer config according to proto
        Params:
            model: nn.module
            proto: QuantCalibrationProto
        Return: config, a dict
        """
        for quant_method in QUANT_METHOD_INFO_MAP.keys():
            config, quant_layers, override_layers = QuantCalibrationConfigBase._generate_layer_config_on_quant_method(
                model, proto, quant_method)

        merged_dict = {}
        for key in set(quant_layers.keys()).union(override_layers.keys()):
            if key in quant_layers and key in override_layers:
                merged_dict[key] = list(
                    set(quant_layers.get(key) + override_layers.get(key)))
            elif key in quant_layers:
                merged_dict[key] = quant_layers.get(key)
            else:
                merged_dict[key] = override_layers.get(key)
        return config, merged_dict

    @staticmethod
    def _generate_layer_config_on_quant_method(model, proto, quant_method):
        """
        Function: Generate layer config according to proto on method
        Params:
            model: nn.module
            proto: QuantCalibrationProto
            quant_method: the method of doing quantization
            
        Return: 
            config, a dict
            quant_layers: layers to be quantized
            override_layers: quantized layer overrided
        """
        quant_layers = dict()
        layer_key, common_config_key, config_key = QUANT_METHOD_INFO_MAP.get(quant_method)
        specific_quant_layers = proto.get_quant_layers(quant_method)
        quant_layers[layer_key] = specific_quant_layers
        QuantCalibrationConfigBase.check_quant_layers(model, quant_layers)

        override_layers = proto.get_override_layers()
        QuantCalibrationConfigBase.check_quant_layers(model, override_layers)

        calibration_common_config = proto.get_quant_config(common_config_key)
        config = OrderedDict()

        if layer_key in quant_layers:
            for layer in quant_layers.get(layer_key):
                if layer not in config:
                    config[layer] = OrderedDict()
                config[layer][config_key] = calibration_common_config.copy()

        if layer_key in override_layers:
            for layer in override_layers.get(layer_key):
                if layer not in config:
                    config[layer] = OrderedDict()
                quant_params = proto.read_override_layer_config(
                    layer, config_key)
                config[layer][config_key] = quant_params
        return config, quant_layers, override_layers

    @check_params(config_file=str, model=torch.nn.Module, quant_layers=dict)
    def create_default_config(self, config_file, model, quant_layers):
        '''
        Function: create default quant calibration config by graph
        Parameter: config_file: the path of json file to save quant calibration config
                   model: torch.nn.Module
                   quant_layers: a dict to indicate quantized layers
        Return: None
        '''
        self._clear_config_tree()
        QuantCalibrationConfigBase.check_quant_layers(model, quant_layers)
        quant_layer_type_map = QuantCalibrationConfigBase.convert_quant_layer_format(
            quant_layers, model)
        if not quant_layer_type_map:
            raise RuntimeError(
                "No quant enable layer in quant config file, "
                "please check the quant_layers input.")
        self.config_root.build_default(quant_layer_type_map)
        ordered_config = self.config_root.dump()
        QuantCalibrationConfigBase._del_reduant_config(ordered_config)
        save_to_json(config_file, ordered_config)

    def create_config_from_proto(self, config_file, model, config_proto_file):
        '''
        Function: create quant calibration config by graph and simple config file
        Parameter: config_file: the path of json file to save quant calibration config
                   model: torch.nn.Module
                   config_proto_file: the path of user's simple config file
        Return: None
        '''
        proto = QuantCalibrationProtoConfig(config_proto_file, model)
        config, quant_layers = QuantCalibrationConfigBase._generate_layer_config(
            model, proto)

        QuantCalibrationConfigBase.check_quant_layers(model, quant_layers)
        global_config = proto.get_proto_global_config()
        for item in global_config.keys():
            if global_config[item] is not None:
                config[item] = global_config[item]

        self._clear_config_tree()
        quant_layer_type_map = QuantCalibrationConfigBase.convert_quant_layer_format(
            quant_layers, model)
        if not quant_layer_type_map:
            raise RuntimeError(
                "No quant enable layer in quant config file, "
                "please check the quant config file.")
        self.config_root.set_strong_check(False)
        self.config_root.build(config, quant_layer_type_map)
        ordered_config = self.config_root.dump()
        QuantCalibrationConfigBase._del_reduant_config(ordered_config)
        save_to_json(config_file, ordered_config)

    def parse_quant_config(self, config_file, model):
        '''
        Function: parse quant calibration config
            if graph is None, will not check op specification
        Parameter: config_file: the path of config json file
                   model: nn.module
        Return: dict, quant calibration config
        '''
        def _detect_repetitive_key_hook(lst):
            '''
            a hook function for detect repeated key in config file.
            '''
            keys = [key for key, val in lst]
            repeat_keys = find_repeated_items(keys)
            check_no_repeated(repeat_keys, config_file)
            result = {}
            for key, val in lst:
                result[key] = val
            return result

        with open(config_file, 'r') as fid:
            try:
                quant_config = json.load(
                    fid, object_pairs_hook=_detect_repetitive_key_hook)
            except json.decoder.JSONDecodeError as e:
                raise ValueError(
                    "config_file {} is invalid, please check.".format(config_file)) from e

        quant_layers = defaultdict(list)
        layer_type = ModuleHelper.get_name_type_dict(model)
        for item in quant_config:
            if item not in layer_type or not isinstance(quant_config.get(item), dict):
                continue
            for quant_info in QUANT_METHOD_INFO_MAP.values():
                if quant_info.config_key in quant_config.get(item):
                    quant_layers[quant_info.layer_key].append(item)

        quant_layer_type_map = QuantCalibrationConfigBase.convert_quant_layer_format(
            quant_layers, model)
        if not quant_layer_type_map:
            raise RuntimeError(
                "No quant enable layer in quant config file, "
                "please check the quant config file.")
        self._clear_config_tree()
        self.config_root.set_strong_check(False)
        self.config_root.build(quant_config, quant_layer_type_map)
        ordered_config = self.config_root.dump()

        QuantCalibrationConfigBase._del_reduant_config(ordered_config)
        QuantCalibrationConfigBase.add_global_to_layer(ordered_config)
        LOGGER.logd('kv_cache config is {}'.format(
            ordered_config), module_name=MODULE_NAME)
        return ordered_config

    def get_quant_layers(self, quant_calibration_config, quant_method):
        '''
        get quant layer based on quant method
        quant_calibration_config(dict): parsed quantization configuration
        quant_method(str): method of doing quantization
        '''
        quant_layers = list()
        for key, value in quant_calibration_config.items():
            if key in self.config_root.get_global_keys():
                continue
            if not isinstance(value, dict):
                raise RuntimeError(
                    'Layer {} config should be a dict, but not it is {}.'
                    ' Please check your config file'.format(key, value))
            if QUANT_METHOD_INFO_MAP.get(quant_method).config_key not in value:
                continue
            quant_layers.append(key)
        return quant_layers

    def _clear_config_tree(self):
        ''' inner method '''
        self.config_root = QuantCalibrationConfigRoot(ModuleHelper, CAPACITY)