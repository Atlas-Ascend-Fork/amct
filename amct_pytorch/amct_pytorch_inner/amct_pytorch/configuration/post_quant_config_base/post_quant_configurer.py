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

import torch

from amct_pytorch.common.utils.files import save_to_json
from amct_pytorch.common.utils.util import check_no_repeated
from amct_pytorch.common.utils.util import find_repeated_items
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.utils.model_util import ModuleHelper

from amct_pytorch.configuration.post_quant_config_base.post_quant_proto_config \
    import PostQuantProtoConfig, DEFAULT_GROUP_SIZE_MAP, GROUP_SIZE_SUPPORTED_MAP, GROUP_SIZE_SUPPORTED_DTYPE
from amct_pytorch.configuration.post_quant_config_base.post_quant_field \
    import RootConfig
from amct_pytorch.utils.vars import BATCH_NUM, QUANT_COMBINATION_MAP, POST_QUANTIZE_A8W8_TYPES, \
    ALLOWED_ROUND_MODE_MAP, WTS_GRANULARITY_SUPPORT_MAP, ALLOWED_WEIGHT_DTYPES, ACT_GRANULARITY_SUPPORT_MAP
from amct_pytorch.utils.vars import MXFP8_E4M3FN, MXFP4_E2M1, FLOAT8_E4M3FN, FLOAT4_E2M1, INT8

MODULE_NAME = 'PostQuantConfigurer'


def _check_no_skip_module(skip_layers, layer_name):
    """
    Check whether the current layer is a effective layers
    Params:
    skip_layers: list, Contains layers that need to skip quantization
    layer_name: string, Specifies whether to skip the quantized layer name
    Return: bool, Indicates whether to skip quantization at the current layer
    """
    # If Skip_layers is not defined, the default value is None.
    if not skip_layers:
        return True

    for skip_layer in skip_layers:
        if skip_layer in layer_name:
            LOGGER.logd('The {} quantifiable module is skipped'.format(layer_name))
            return False

    return True


def _check_support_int4_ori_dtype(layer_name, mod, quant_data_comb, quant_config):
    """
    Check whether the current layer supports quantization
    Params:
    layer_name: string, layer name
    mod: torch.nn.module
    quant_data_comb: str, current quant dtype comb
    quant_config: dict, quant config information
    Return: bool, Indicates whether the current layer supports quantization
    """
    if not quant_config:
        return True
    quant_param = quant_config.get('weight_quant_algo_params', None)
    if quant_data_comb != 'NOT_QUANTIZE INT4':
        return True

    if quant_param is not None and quant_param.get("lut_quantize") is not None:
        if mod.weight.dtype != torch.float32:
            LOGGER.logd('layer:{} cannot be quantized, quant act_dtype and wts dtype {} with lut_quantize'
                        ' only support ori_dtype float32'.format(layer_name, quant_data_comb))
            return False
    else:
        if mod.weight.dtype == torch.float32:
            LOGGER.logd('layer:{} cannot be quantized, quant act_dtype and wts dtype {} without lut_quantize'
                        ' do not support ori_dtype float32'.format(layer_name, quant_data_comb))
            return False
    return True


def _check_support_module(layer_name, mod, supported_type, allowed_weight_dtypes):
    """
    Check whether the current layer supports quantization
    Params:
    layer_name: string, layer name
    mod: torch.nn.module
    supported_type: list, Supported Quantization Operator Types
    allowed_weight_dtypes: list, Supported Weight Types
    Return: bool, Indicates whether the current layer supports quantization
    """
    mod_type = type(mod).__name__
    if mod_type not in supported_type:
        return False

    if mod_type == 'Conv2d' and mod.padding_mode != 'zeros':
        return False

    if mod.weight.dtype not in allowed_weight_dtypes:
        return False

    # not support meta device
    if mod.weight.device == torch.device('meta'):
        LOGGER.logd('quantifiable module not support meta device, the {} module is skipped'.format(layer_name))
        return False

    return True


def _check_group_size_for_npu(layer_name, mod, quant_data_comb, quant_config):
    """
    Check whether the current layer supports quantization
    Params:
    mod: torch.nn.module
    quant_data_comb: activation quant dtype and weight quant dtype
    act_granularity: activation quant granularity
    Return: bool, Indicates whether the current layer supports quantization
    """
    if not quant_config:
        return True
    if quant_config.get('weight_granularity', None) != "PER_GROUP":
        return True
    mod_type = type(mod).__name__
    if mod_type != 'Linear':
        return True
    if quant_data_comb not in ['NOT_QUANTIZE INT8', 'NOT_QUANTIZE INT4']:
        return True
   
    quant_params = quant_config.get('weight_quant_algo_params', None)
    if quant_params is not None and quant_params.get("lut_quantize") is not None:
        return True

    group_size = quant_config.get('group_size')
    if group_size < 32 or group_size >= mod.weight.shape[1] or group_size % 32 != 0:
        raise ValueError("act_type and wts_type {} only support group_size [32, {}] and integer multiple of 32, "
                "current is {}".format(quant_data_comb, mod.weight.shape[1] - 1, group_size))
    
    return True


def _check_linear_shape(layer_name, mod, quant_data_comb, quant_config):
    """
    Check whether the current layer supports quantization
    Params:
    mod: torch.nn.module
    quant_data_comb: activation quant dtype and weight quant dtype
    Return: bool, Indicates whether the current layer supports quantization
    """
    quant_params = quant_config.get('weight_quant_algo_params', None)
    mod_type = type(mod).__name__
    if mod_type != 'Linear':
        return True

    # npu op check cin length be integer multiple of 64
    support_quant_dtype_comb = [MXFP8_E4M3FN + ' ' + MXFP4_E2M1, FLOAT8_E4M3FN + ' ' + FLOAT4_E2M1]
    if quant_data_comb in support_quant_dtype_comb and mod.weight.shape[1] % 64 != 0:
        LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement'
                    'cin length should be integer multiple of 64'.format(layer_name, quant_data_comb))
        return False

    # npu op check cin length ceildiv(cin, 32) must be even number
    check_mxfp8_mxfp8_dtype = MXFP8_E4M3FN + ' ' + MXFP8_E4M3FN
    if quant_data_comb == check_mxfp8_mxfp8_dtype and ((mod.weight.shape[1] + 31) // 32) % 2 != 0:
        LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement'
            'cin length ceildiv(cin, 32) must be even number'.format(layer_name, quant_data_comb))
        return False

    # check per_token int8*int8, cout length be integer multiple of 8, cin length be integer multiple of 16
    check_per_token_dtype = [INT8 + ' ' + INT8,]
    act_granularity = quant_config.get('act_granularity', 'PER_TENSOR')
    if quant_data_comb in check_per_token_dtype and act_granularity == 'PER_TOKEN':
        if mod.weight.shape[0] % 8 != 0 or mod.weight.shape[1] % 16 != 0:
            LOGGER.logd('weight first dim should be integer multiple of 8, second dim should be integer multiple of 16')
            return False
        
    # npu op check no bias
    check_no_bias_dtype = ["FLOAT8_E4M3FN FLOAT4_E2M1"]
    if quant_data_comb in check_no_bias_dtype and mod.bias is not None:
        LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement'
            'bias is not supported'.format(layer_name, quant_data_comb))
        return False

    # npu op check cin length be integer multiple of 32b
    if quant_data_comb == 'NOT_QUANTIZE INT4':
        if quant_params is not None and quant_params.get("lut_quantize") is not None:
            return True
        if mod.weight.shape[1] % 64 != 0 or mod.weight.shape[0] % 64 != 0:
            LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement'
                'cin and cout length should be integer multiple of 64'.format(layer_name, quant_data_comb))
            return False
    
    if quant_data_comb == 'NOT_QUANTIZE INT8':
        if mod.weight.shape[1] % 32 != 0 or mod.weight.shape[0] % 32 != 0:
            LOGGER.logd('layer:{} cannot be quantized, act_dtype and wts dtype {} has shape requirement'
                'cin and cout length should be integer multiple of 32'.format(layer_name, quant_data_comb))
            return False

    return True


def _check_group_size(quant_config):
    """
    check group size whether support by wts_type
    Params:
    quant_config: quantize config in config dict
    """
    wts_type = quant_config.get('wts_type')
    if quant_config.get('weight_granularity') != 'PER_GROUP':
        raise ValueError("group_size only support weight granularity: PER_GROUP")
    if wts_type not in GROUP_SIZE_SUPPORTED_DTYPE:
        raise ValueError("group_size only support for weight dtype: {}".format(
            GROUP_SIZE_SUPPORTED_DTYPE))
    if wts_type in GROUP_SIZE_SUPPORTED_MAP.keys() and quant_config.get('group_size') not in \
        GROUP_SIZE_SUPPORTED_MAP.get(wts_type):
        raise ValueError("wts_type {} only support group_size value: {}".format(wts_type,
            GROUP_SIZE_SUPPORTED_MAP.get(wts_type)))


class PostQuantConfigurer():
    '''post quant config base'''
    def __init__(self, capacity):
        ''' inner method '''
        self.capacity = capacity
        self.config_tree = RootConfig(None, self.capacity)

    @staticmethod
    def add_global_to_layer(quant_config):
        """
        add global quantize parameter to each layer
        Params:
        quant_config: a dict, including layer detail quant config
        """
        for layer_name in quant_config:
            if not isinstance(quant_config.get(layer_name), dict):
                continue
            if quant_config.get(BATCH_NUM) is not None:
                quant_config[layer_name][BATCH_NUM] = quant_config.get(BATCH_NUM)

    @staticmethod
    def get_supported_layers(model, quant_config=None, skip_layers=None):
        """
        get support quant layers in model
        Params:
        model: torch.nn.Module, model to be quantized
        quant_config: a dict, quant config
        skip_layers: list, Contains layer name that need to skip quantization
        """
        supported_layers = dict()
        if quant_config:
            act_granularity = quant_config.get('act_granularity', None)
            quant_data_comb = quant_config.get('act_type', 'NOT_QUANTIZE') + ' ' + quant_config.get('wts_type', 'INT8')
            supported_type = QUANT_COMBINATION_MAP.get(quant_data_comb)
            wts_type = quant_config.get('wts_type')
            if not supported_type:
                raise ValueError("act_type {} and wts_type {} unsupport"
                    .format(quant_config.get('act_type', 'FLOAT16'), quant_config.get('wts_type', 'INT8')))
            if quant_config.get('round_mode') and \
                quant_config.get('round_mode') not in ALLOWED_ROUND_MODE_MAP.get(wts_type):
                raise ValueError("{} round_mode only support {}".format(wts_type, ALLOWED_ROUND_MODE_MAP.get(wts_type)))
            if quant_data_comb not in WTS_GRANULARITY_SUPPORT_MAP.get(quant_config.get('weight_granularity')):
                raise ValueError("wts_type {} unsupport weight_granularity {}".format(wts_type, 
                    quant_config.get('weight_granularity')))
            if quant_config.get('act_granularity') is not None and \
                quant_config.get('act_type') not in [MXFP8_E4M3FN, MXFP4_E2M1] and \
                quant_data_comb not in ACT_GRANULARITY_SUPPORT_MAP.get(quant_config.get('act_granularity')):
                raise ValueError("act_type {} is unsupported act_granularity {}".format(quant_config.get('act_type'), 
                    quant_config.get('act_granularity')))
            if quant_config.get('group_size'):
                _check_group_size(quant_config)
        else:
            act_granularity = 'PER_TENSOR'
            quant_data_comb = 'INT8 INT8'
            supported_type = POST_QUANTIZE_A8W8_TYPES

        # Obtains the model weight data type supported by the quantization bit width combination.
        allowed_weight_dtypes = ALLOWED_WEIGHT_DTYPES.get(quant_data_comb, [torch.bfloat16, torch.float16])

        # Skip the number of quantifiable layers
        skip_layer_counts = 0
        for name, mod in model.named_modules():
            if not _check_support_module(name, mod, supported_type, allowed_weight_dtypes):
                continue

            if not _check_support_int4_ori_dtype(name, mod, quant_data_comb, \
                quant_config):
                continue

            if quant_config:
                if not _check_linear_shape(name, mod, quant_data_comb, quant_config):
                    continue

            if not _check_no_skip_module(skip_layers, name):
                skip_layer_counts += 1
                continue

            if not _check_group_size_for_npu(name, mod, quant_data_comb, quant_config):
                continue

            supported_layers[name] = type(mod).__name__

        if skip_layers:
            if skip_layer_counts == 0:
                raise ValueError("skip_layers must match at least one quantifiable layer")
            else:
                LOGGER.logi('{} quantifiable modules are skipped'.format(skip_layer_counts))

        return supported_layers

    def create_default_config(self, config_file, model):
        """inner method, only generate pose quant config
        Params:
        config_file: a str, file path to save quant config
        model: torch.nn.Module, model to be quantized
        """
        supported_layers = self.get_supported_layers(model)
        if not supported_layers:
            raise ValueError("No supported layers")
        self._clear_config_tree()
        self.config_tree.build_default(supported_layers)
        ordered_config = self.config_tree.dump()
        save_to_json(config_file, ordered_config)

    def generate_layer_config(self, config, quant_config, quant_layers):
        """
        Generate layer config according to proto
        Params:
        config: a dict, contains global config
        quant_config: a dict, parsed proto config
        quant_layers: a dict, layers enable to do quant
        Return: config, a dict
        """
        for layer in quant_layers:
            config[layer] = quant_config.copy()
        return config

    def create_config_from_proto(self, config_file, model, config_defination):
        '''
        create json config file from proto config.
        Inputs:
        config_file: json config file path
        model: torch.nn.Module, model to be quantized
        config_defination: proto config path
        Returns:
        None
        '''
        config_defination = os.path.realpath(config_defination)
        proto = PostQuantProtoConfig(config_defination)
        global_config = proto.get_proto_global_config()
        config = dict()
        config["batch_num"] = global_config["batch_num"]

        quant_config = proto.get_common_config()

        supported_layers = self.get_supported_layers(model, quant_config, global_config.get("skip_layers"))
        if not supported_layers:
            raise ValueError("No supported layers")

        config = self.generate_layer_config(config, quant_config, supported_layers)

        self.config_tree.build(config, supported_layers)
        ordered_config = self.config_tree.dump()
        save_to_json(config_file, ordered_config)

    def parse_config_file(self, config_file, model):
        '''
        parse config file, check incorrect value, and fill default value.
        Inputs:
        config_file: json config file path
        model: torch.nn.Module, model to be quantized
        Returns:
        quant_config: quant config dict
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
            quant_config = json.load(
                fid, object_pairs_hook=_detect_repetitive_key_hook)
        model_helper = ModuleHelper(model)
        # check whether module in model
        valid_layers = dict()
        for key in quant_config:
            if key == BATCH_NUM:
                continue
            model_helper.get_module(key)
            valid_layers.update(self.get_supported_layers(model, quant_config.get(key)))

        self._check_reduant_config(quant_config)
        self.config_tree.build(quant_config, valid_layers)
        quant_config = self.config_tree.dump()
        PostQuantConfigurer.add_global_to_layer(quant_config)
        LOGGER.logd('post quant config is {}'.format(quant_config))
        return quant_config

    def _clear_config_tree(self):
        """
        replace config tree with empty ones
        """
        self.config_tree = RootConfig(None, self.capacity)

    def _check_reduant_config(self, quant_config):
        """
        check whether config have unexpected elem
        Params:
        quant_config: a dict, quant config
        """
        for key in quant_config:
            if key in self.config_tree.get_global_keys():
                continue
            if not isinstance(quant_config[key], dict):
                raise ValueError(
                    "{} is not valid config, its value should be a dict if it is a layer.".format(key))
