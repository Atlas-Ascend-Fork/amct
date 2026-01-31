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
from collections import OrderedDict
from enum import unique, Enum
from google.protobuf import text_format

from amct_pytorch.proto import post_quant_config_pb2
from amct_pytorch.utils.vars import BATCH_NUM
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.utils.vars import POST_QUANTIZE_ACT_TYPES, POST_QUANTIZE_WTS_TYPES, POST_QUANTIZE_WTS_ONLY_TYPES, \
    SMOOTH_QUANT_SUPPORT_COMBINATION, ACT_GRANULARITY_SUPPORT_MAP
from amct_pytorch.utils.vars import HIFLOAT8, FLOAT8_E4M3FN, MXFP8_E4M3FN, INT8, INT4, FLOAT4_E2M1, \
    FLOAT4_E1M2, MXFP4_E2M1
from amct_pytorch.utils.vars import WTS_ASYMMETRIC_DTYPE



@unique
class PostQuantDataType(Enum):
    HIFLOAT8 = 0
    FLOAT8_E4M3FN = 1
    INT8 = 10
    INT4 = 11
    MXFP4_E2M1 = 12
    MXFP8_E4M3FN = 13
    FLOAT4_E2M1 = 14
    FLOAT4_E1M2 = 15


@unique
class WtsQuantGranularity(Enum):
    PER_TENSOR = 0
    PER_CHANNEL = 1
    PER_GROUP = 2


@unique
class ActQuantGranularity(Enum):
    PER_TENSOR = 0
    PER_TOKEN = 1


@unique
class RoundMode(Enum):
    ROUND = 0
    HYBRID = 1
    RINT = 2


@unique
class LUTQuantAlgo(Enum):
    ARCTAN = 0
    CLUSTER = 1


DEFAULT_WEIGHT_GRANULARITY_MAP = {
    PostQuantDataType.HIFLOAT8.name: WtsQuantGranularity.PER_TENSOR,
    PostQuantDataType.FLOAT8_E4M3FN.name: WtsQuantGranularity.PER_TENSOR,
    PostQuantDataType.INT4.name: WtsQuantGranularity.PER_TENSOR,
    PostQuantDataType.INT8.name: WtsQuantGranularity.PER_TENSOR,
    PostQuantDataType.MXFP4_E2M1.name: WtsQuantGranularity.PER_GROUP,
    PostQuantDataType.MXFP8_E4M3FN.name: WtsQuantGranularity.PER_GROUP,
    PostQuantDataType.FLOAT4_E2M1.name: WtsQuantGranularity.PER_GROUP,
    PostQuantDataType.FLOAT4_E1M2.name: WtsQuantGranularity.PER_GROUP,
}


DEFAULT_ROUND_MODE_MAP = {
    PostQuantDataType.HIFLOAT8.name: RoundMode.ROUND,
    PostQuantDataType.FLOAT8_E4M3FN.name: RoundMode.RINT,
    PostQuantDataType.INT4.name: RoundMode.RINT,
    PostQuantDataType.INT8.name: RoundMode.RINT,
    PostQuantDataType.MXFP4_E2M1.name: RoundMode.RINT,
    PostQuantDataType.MXFP8_E4M3FN.name: RoundMode.RINT,
    PostQuantDataType.FLOAT4_E2M1.name: RoundMode.RINT,
    PostQuantDataType.FLOAT4_E1M2.name: RoundMode.RINT,
}


DEFAULT_LUT_MODE_MAP = {
    0: LUTQuantAlgo.ARCTAN,
    1: LUTQuantAlgo.CLUSTER
}


DEFAULT_GROUP_SIZE_MAP = {
    PostQuantDataType.FLOAT4_E2M1.name: 128,
    PostQuantDataType.FLOAT4_E1M2.name: 128,
    PostQuantDataType.MXFP4_E2M1.name: 32,
}


GROUP_SIZE_SUPPORTED_MAP = {
    PostQuantDataType.FLOAT4_E1M2.name: [32, 64, 128, 256],
    PostQuantDataType.FLOAT4_E2M1.name: [32, 64, 128, 256],
    PostQuantDataType.MXFP4_E2M1.name: [32]
}

GROUP_SIZE_SUPPORTED_DTYPE = [INT4, INT8, FLOAT4_E2M1, FLOAT4_E1M2, MXFP4_E2M1]

class PostQuantProtoConfig():
    """
    Function: Cope with simple config file from proto.
    APIs:
        get_proto_global_config
        get_common_config
    """
    def __init__(self, config_proto_file):
        self.proto_config = self.read(config_proto_file)

    @staticmethod
    def read(config_proto_file):
        """ Read config from config_proto_file. """
        proto_config = post_quant_config_pb2.AMCTPostQuantConfig()
        with open(config_proto_file, 'rb') as cfg_file:
            pbtxt_string = cfg_file.read()
            text_format.Merge(pbtxt_string, proto_config)
        return proto_config

    @staticmethod
    def get_calibration_config(config):
        """ generate layer config base on common config """
        layer_config = OrderedDict()
        layer_config['act_type'] = PostQuantDataType(config.act_type).name
        if layer_config['act_type'] not in POST_QUANTIZE_ACT_TYPES:
            raise ValueError("Not support act_type {}.".format(layer_config['act_type']))
        layer_config['wts_type'] = PostQuantDataType(config.wts_type).name
        if layer_config['wts_type'] not in POST_QUANTIZE_WTS_TYPES:
            raise ValueError("Not support wts_type {}.".format(layer_config['wts_type']))
        layer_config['act_asymmetric'] = config.act_asymmetric

        if layer_config.get('act_asymmetric') and \
            layer_config.get('act_type') in [HIFLOAT8, FLOAT8_E4M3FN, MXFP8_E4M3FN]:
            raise ValueError("act_asymmetric is unsupported to be true when act_type {}."
                .format(layer_config.get('act_type')))
        if hasattr(config, 'act_granularity') and config.HasField('act_granularity'):
            layer_config['act_granularity'] = ActQuantGranularity(config.act_granularity).name
        else:
            if layer_config['act_type'] not in [MXFP4_E2M1, MXFP8_E4M3FN]:
                layer_config['act_granularity'] = 'PER_TENSOR'
        # check act_asymmetric and act_granularity conflict
        if layer_config.get('act_asymmetric') and layer_config['act_granularity'] == 'PER_TOKEN':
            raise ValueError('act_asymmetric is unsupported to be true when act_granularity is PER_TOKEN')
            
        if hasattr(config, 'weight_granularity') and config.HasField('weight_granularity'):
            layer_config['weight_granularity'] = WtsQuantGranularity(config.weight_granularity).name
        else:
            layer_config['weight_granularity'] = DEFAULT_WEIGHT_GRANULARITY_MAP.get(
                layer_config['wts_type']).name

        if hasattr(config, 'round_mode') and config.HasField('round_mode'):
            layer_config['round_mode'] = RoundMode(config.round_mode).name
        else:
            layer_config['round_mode'] = DEFAULT_ROUND_MODE_MAP.get(layer_config['wts_type']).name

        if hasattr(config, 'act_smooth') and config.HasField('act_smooth'):
            quant_comb = layer_config['act_type'] + ' ' + layer_config['wts_type']
            if quant_comb not in SMOOTH_QUANT_SUPPORT_COMBINATION:
                raise ValueError("act_smooth only support act_type and wts_type in {}"
                    .format(SMOOTH_QUANT_SUPPORT_COMBINATION))
            act_smooth_params = OrderedDict()
            act_smooth_params['smooth_strength'] = config.act_smooth.smooth_strength 
            layer_config['act_smooth'] = act_smooth_params
        return layer_config

    @staticmethod
    def get_weight_only_config_wts_type(weight_only_config, layer_config):
        layer_config['wts_type'] = PostQuantDataType(weight_only_config.wts_type).name
        if layer_config['wts_type'] not in POST_QUANTIZE_WTS_ONLY_TYPES:
            raise ValueError("weight_compress_only only support wts_type {}.".format(POST_QUANTIZE_WTS_ONLY_TYPES))

    @staticmethod
    def get_weight_only_config_weight_compress_only(weight_only_config, layer_config):
        layer_config['weight_compress_only'] = weight_only_config.weight_compress_only

    @staticmethod
    def get_weight_only_config_weight_granularity(weight_only_config, layer_config):
        if weight_only_config.HasField('weight_granularity'):
            layer_config['weight_granularity'] = WtsQuantGranularity(weight_only_config.weight_granularity).name
        else:
            if hasattr(weight_only_config, 'lut_quantize') and weight_only_config.HasField('lut_quantize'):
                layer_config['weight_granularity'] = 'PER_GROUP'
            else:
                layer_config['weight_granularity'] = DEFAULT_WEIGHT_GRANULARITY_MAP.get(layer_config['wts_type']).name

    @staticmethod
    def get_weight_only_config_awq_quantize(weight_only_config, layer_config):
        if hasattr(weight_only_config, 'awq_quantize') and weight_only_config.HasField('awq_quantize'):
            if layer_config['wts_type'] not in (PostQuantDataType.MXFP4_E2M1.name, PostQuantDataType.FLOAT4_E1M2.name, 
                                                PostQuantDataType.FLOAT4_E2M1.name, PostQuantDataType.INT8.name, 
                                                PostQuantDataType.INT4.name):
                raise RuntimeError("awq_quantize only support wts_type MXFP4_E2M1, FLOAT4_E2M1, "
                                   "FLOAT4_E1M2, INT8, INT4")
            awq_quantize_params = OrderedDict()
            awq_quantize_params['grids_num'] = weight_only_config.awq_quantize.grids_num
            if 'weight_quant_algo_params' not in layer_config:
                layer_config['weight_quant_algo_params'] = OrderedDict()
            layer_config['weight_quant_algo_params']['awq_quantize'] = awq_quantize_params

    @staticmethod
    def get_weight_only_config_lut_quantize(weight_only_config, layer_config):
        if hasattr(weight_only_config, 'lut_quantize') and weight_only_config.HasField('lut_quantize'):
            if layer_config['wts_type'] not in (PostQuantDataType.INT4.name,):
                raise RuntimeError("lut_quantize only support wts_type INT4")
            lut_quantize_params = OrderedDict()
            lut_quantize_params['lut_algo'] = DEFAULT_LUT_MODE_MAP.get(weight_only_config.lut_quantize.lut_algo).name
            if 'weight_quant_algo_params' not in layer_config:
                layer_config['weight_quant_algo_params'] = OrderedDict()
            layer_config['weight_quant_algo_params']['lut_quantize'] = lut_quantize_params
            layer_config['group_size'] = 256

    @staticmethod
    def get_weight_only_config_group_size(weight_only_config, layer_config):
        if hasattr(weight_only_config, 'group_size'):
            if layer_config['weight_granularity'] != 'PER_GROUP':
                if weight_only_config.HasField('group_size'):
                    raise RuntimeError("group_size only support weight_granularity PER_GROUP")
                else:
                    return
            if weight_only_config.HasField('group_size'):
                layer_config['group_size'] = weight_only_config.group_size
            else:
                lut_flag = 'weight_quant_algo_params' in layer_config and \
                    'lut_quantize' in layer_config['weight_quant_algo_params']
                if layer_config['wts_type'] in DEFAULT_GROUP_SIZE_MAP.keys():
                    layer_config['group_size'] = DEFAULT_GROUP_SIZE_MAP.get(layer_config['wts_type'])
                elif layer_config['weight_compress_only'] and \
                    layer_config['wts_type'] in [INT4, INT8] and not lut_flag:
                    raise RuntimeError("group_size is necessary for wts_dtype {} weight only quantization, "
                                       "when weight_granularity is PER_GROUP".format(layer_config['wts_type']))
                

    @staticmethod
    def get_weight_only_config_round_mode(weight_only_config, layer_config):
        if hasattr(weight_only_config, 'round_mode') and weight_only_config.HasField('round_mode'):
            layer_config['round_mode'] = RoundMode(weight_only_config.round_mode).name
        else:
            layer_config['round_mode'] = DEFAULT_ROUND_MODE_MAP.get(layer_config['wts_type']).name

    @staticmethod
    def get_weight_only_config_gptq_quantize(weight_only_config, layer_config):
        if hasattr(weight_only_config, 'gptq_quantize') and weight_only_config.HasField('gptq_quantize'):
            if 'weight_quant_algo_params' not in layer_config:
                layer_config['weight_quant_algo_params'] = OrderedDict()
            layer_config['weight_quant_algo_params']['gptq_quantize'] = OrderedDict()

    @staticmethod
    def get_weight_only_config_wts_asymmetric(weight_only_config, layer_config):
        if hasattr(weight_only_config, 'wts_asymmetric'):
            if weight_only_config.HasField('wts_asymmetric'):
                layer_config['wts_asymmetric'] = weight_only_config.wts_asymmetric
                if weight_only_config.wts_asymmetric and layer_config['wts_type'] not in WTS_ASYMMETRIC_DTYPE:
                    raise ValueError("weight only compress with wts_asymmetric do not support wts_type {}.".format(
                        layer_config['wts_type']))
                if layer_config.get('weight_quant_algo_params') is not None and \
                    layer_config['weight_quant_algo_params'].get("lut_quantize") is not None:
                    raise ValueError("weight only compress with wts_asymmetric do not support lut_quantize.")
            else:
                layer_config['wts_asymmetric'] = False


    @staticmethod
    def get_weight_only_config(weight_only_config):
        """ generate layer config base on weight only config """
        layer_config = OrderedDict()
        PostQuantProtoConfig.get_weight_only_config_wts_type(weight_only_config, layer_config)
        PostQuantProtoConfig.get_weight_only_config_weight_compress_only(weight_only_config, layer_config)
        PostQuantProtoConfig.get_weight_only_config_weight_granularity(weight_only_config, layer_config)
        PostQuantProtoConfig.get_weight_only_config_awq_quantize(weight_only_config, layer_config)
        PostQuantProtoConfig.get_weight_only_config_lut_quantize(weight_only_config, layer_config)
        PostQuantProtoConfig.get_weight_only_config_group_size(weight_only_config, layer_config)
        PostQuantProtoConfig.get_weight_only_config_round_mode(weight_only_config, layer_config)
        PostQuantProtoConfig.get_weight_only_config_gptq_quantize(weight_only_config, layer_config)
        PostQuantProtoConfig.get_weight_only_config_wts_asymmetric(weight_only_config, layer_config)
        return layer_config

    def get_proto_global_config(self):
        """parse global config"""
        global_config = OrderedDict()
        if hasattr(self.proto_config, BATCH_NUM) and self.proto_config.HasField(BATCH_NUM):
            global_config[BATCH_NUM] = self.proto_config.batch_num
            if global_config[BATCH_NUM] < 1:
                raise ValueError("batch_num({}) should be greater than zero".format(global_config[BATCH_NUM]))
        else:
            global_config[BATCH_NUM] = 1

        if hasattr(self.proto_config, "skip_layers"):
            global_config["skip_layers"] = self._get_skip_layers()
        return global_config

    def _get_skip_layers(self):
        """
        Deduplicate and check whether skip_layers are special characters
        """
        skip_layers = list(set(self.proto_config.skip_layers))
        for skip_layer in skip_layers:
            if not any(c.isalnum() for c in skip_layer):
                raise ValueError("skip layer {} must contain letters or digits".format(skip_layer))

        return skip_layers

    def get_weight_config(self):
        """get weight config"""
        if self.proto_config.HasField('weight_only_config'):
            weight_only_config = getattr(self.proto_config, 'weight_only_config')
            return weight_only_config
        else:
            return None

    def get_common_config(self):
        """get calibration config"""
        weight_only_config = self.get_weight_config()
        config = OrderedDict()
        if weight_only_config and getattr(weight_only_config, 'weight_compress_only'):
            if self.proto_config.HasField('common_config'):
                LOGGER.logw("weight_compress_only is True, common config is ignored.")
            config = PostQuantProtoConfig.get_weight_only_config(weight_only_config)
        else:
            LOGGER.logd("weight_compress_only is False, weight only config is ignored.")
            if self.proto_config.HasField('common_config'):
                common_config = getattr(self.proto_config, 'common_config')
                config = PostQuantProtoConfig.get_calibration_config(common_config)
        return config
