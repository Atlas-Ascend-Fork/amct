#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Get config dict for quantization from .cfg file.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from collections import OrderedDict, namedtuple
from enum import IntEnum, unique, Enum

from google.protobuf import text_format # pylint: disable=E0401

from ..utils.util import find_repeated_items
from ..utils.util import check_no_repeated
from ..utils.util import proto_float_to_python_float
from .field import NUM_STEPS
from .field import NUM_OF_ITERATION
from ...utils.log import LOGGER # pylint: disable=relative-beyond-top-level
from ..utils.vars_util import WINOGRAD_NUM_BITS

_MODULE_NAME = 'Proto_config'
ASYMMETRIC = 'asymmetric'
CONV_CALIBRATION_CONFIG = 'conv_calibration_config'
FC_CALIBRATION_CONFIG = 'fc_calibration_config'
ACTIVATION_QUANT_PARAMS = 'activation_quant_params'
BATCH_NUM = 'batch_num'
CHANNEL_WISE = 'channel_wise'
WEIGHT_QUANT_PARAMS = 'weight_quant_params'


ProtoConfRet = namedtuple('ProtoConfRet', ['global_config', 'common_config', 'type_config', 'layer_config'])


# enum value of DataType in proto
@unique
class ProtoDataType(IntEnum):
    INT4 = 0
    INT8 = 1
    INT16 = 2


@unique
class FakequantPrecisionMode(Enum):
    DEFAULT = 0
    FORCE_FP16_QUANT = 1


class ProtoConfig(): # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Function: Cope with simple config file from proto.
    APIs:
    """
    def __init__(self, config_file, # pylint: disable=R0913
                 capacity, graph_querier, graph, proto_config):
        self.config_file = config_file
        self.proto_config = proto_config
        with open(self.config_file, 'rb') as cfg_file:
            pbtxt_string = cfg_file.read()
            text_format.Merge(pbtxt_string, self.proto_config)
        self.channel_wise_types = \
            capacity.get_value('CHANNEL_WISE_TYPES')

        self.capacity = capacity
        self.quantizable_type = capacity.get_value('QUANTIZABLE_TYPES')
        self.int16_quantizable_type = capacity.get_value('INT16_QUANTIZABLE_TYPES')
        self.ada_round_quantizable_type = capacity.get_value('ADA_ROUND_TYPES')
        self.graph_querier = graph_querier
        self.graph = graph
        self.layer_type = graph_querier.get_name_type_dict(graph)

    @staticmethod
    def _parse_data_type(data_type_proto):
        dtype_define = {
            ProtoDataType.INT8.value: 8,
            ProtoDataType.INT16.value: 16
        }
        if data_type_proto not in dtype_define:
            raise ValueError('Not support dtype of {}.'.format(data_type_proto))
        return dtype_define[data_type_proto]

    @staticmethod
    def _get_ada_quantize(ada_quantize, weight_params):
        weight_params['num_iteration'] = ada_quantize.num_iteration
        weight_params['reg_param'] = proto_float_to_python_float(ada_quantize.reg_param)
        weight_params['beta_range'] = [proto_float_to_python_float(ada_quantize.beta_range_start),
                      proto_float_to_python_float(ada_quantize.beta_range_end)]
        weight_params['warm_start'] = proto_float_to_python_float(ada_quantize.warm_start)
        if ada_quantize.HasField(CHANNEL_WISE):
            weight_params[CHANNEL_WISE] = ada_quantize.channel_wise

    @staticmethod
    def _get_hfmg_config(config, act_params):
        '''extract hfmg configs'''
        act_params['act_algo'] = 'hfmg'
        hfmg_quantize = config.hfmg_quantize
        num_of_bins = hfmg_quantize.num_of_bins
        act_params['num_of_bins'] = num_of_bins
        if hasattr(hfmg_quantize, 'dst_type'):
            act_params['num_bits'] = ProtoConfig._parse_data_type(hfmg_quantize.dst_type)
        if hfmg_quantize.HasField(ASYMMETRIC):
            act_params[ASYMMETRIC] = hfmg_quantize.asymmetric
        else:
            act_params[ASYMMETRIC] = None

    @staticmethod
    def _channel_wise_is_true(config):
        if config.get(WEIGHT_QUANT_PARAMS) is None:
            return False

        weight_params = config[WEIGHT_QUANT_PARAMS]
        if weight_params.get('wts_algo') in ('arq_quantize', None) and \
            weight_params.get(CHANNEL_WISE, False):
            return True
        return False

    @staticmethod
    def _get_ifmr_config(config, act_params):
        '''extract ifmr configs'''
        ifmr_quantize = config.ifmr_quantize
        if hasattr(ifmr_quantize, 'dst_type'):
            act_params['num_bits'] = ProtoConfig._parse_data_type(ifmr_quantize.dst_type)
        act_params['act_algo'] = 'ifmr'
        act_params['max_percentile'] = \
            proto_float_to_python_float(ifmr_quantize.max_percentile)
        act_params['min_percentile'] = \
            proto_float_to_python_float(ifmr_quantize.min_percentile)
        act_params['search_range'] = \
            [proto_float_to_python_float(ifmr_quantize.search_range_start),
             proto_float_to_python_float(ifmr_quantize.search_range_end)]
        act_params['search_step'] = proto_float_to_python_float(
            ifmr_quantize.search_step)
        if ifmr_quantize.HasField(ASYMMETRIC):
            act_params[ASYMMETRIC] = ifmr_quantize.asymmetric
        else:
            act_params[ASYMMETRIC] = None


    def get_proto_config(self, enable_quant=True, enable_approximate=False):
        """parse proto config"""
        if enable_approximate:
            global_config = self._get_approx_global_config()
            return ProtoConfRet(global_config, {}, {}, {})
        elif enable_quant:
            global_config = self._get_global_config()

            layer_config = self._get_override_layer_configs()
            type_config = self._get_override_layer_types()
            common_config = self._get_common_config()

            if not common_config and not type_config:
                if hasattr(self.proto_config, CONV_CALIBRATION_CONFIG):
                    common_config = self._get_conv_calibration_config()
                if hasattr(self.proto_config, FC_CALIBRATION_CONFIG):
                    fc_config = self._get_fc_calibration_config()
                    for item in (set(self.quantizable_type) -
                                set(self.channel_wise_types)):
                        type_config[item] = copy.deepcopy(fc_config)
            else:
                if hasattr(self.proto_config, CONV_CALIBRATION_CONFIG):
                    self._raise_ignore_info(CONV_CALIBRATION_CONFIG)
                if hasattr(self.proto_config, FC_CALIBRATION_CONFIG):
                    self._raise_ignore_info(FC_CALIBRATION_CONFIG)

            if hasattr(self.proto_config, 'tensor_quantize'):
                global_config['tensor_quantize'] = self._get_tensor_quantize_config(
                    self.proto_config.tensor_quantize)

            LOGGER.logd('global_config is {}'.format(global_config))
            LOGGER.logd('common_config is {}'.format(common_config))
            LOGGER.logd('type_config is {}'.format(type_config))
            LOGGER.logd('layer_config is {}'.format(layer_config))
            return ProtoConfRet(global_config, common_config, type_config, layer_config)
        return ProtoConfRet({}, {}, {}, {})

    def _get_batch_num(self):
        if self.proto_config.HasField(BATCH_NUM):
            return self.proto_config.batch_num
        return None

    def _get_fakequant_precision_mode(self):
        precision_mode_define = {
            FakequantPrecisionMode.DEFAULT.value: 'DEFAULT',
            FakequantPrecisionMode.FORCE_FP16_QUANT.value: 'FORCE_FP16_QUANT'
        }
        if self.proto_config.HasField('fakequant_precision_mode'):
            return precision_mode_define.get(self.proto_config.fakequant_precision_mode)
        return None

    def _get_activation_offset(self):
        if self.proto_config.HasField('activation_offset'):
            return self.proto_config.activation_offset
        return None

    def _get_joint_quant(self):
        if self.proto_config.HasField('joint_quant'):
            return self.proto_config.joint_quant
        return None

    def _get_weight_offset(self):
        if self.proto_config.HasField('weight_offset'):
            return self.proto_config.weight_offset
        return None

    def _get_skip_layers(self):
        skip_layers = list(self.proto_config.skip_layers)
        repeated_layers = find_repeated_items(skip_layers)
        check_no_repeated(repeated_layers, 'skip_layers')
        return skip_layers

    def _get_skip_approx_layers(self):
        skip_layers = list(self.proto_config.skip_approximation_layers)
        repeated_layers = find_repeated_items(skip_layers)
        check_no_repeated(repeated_layers, 'skip_approximation_layers')
        return skip_layers

    def _get_mapped_type(self, layer_type, map_types):
        if layer_type not in map_types:
            raise ValueError(
                'Unrecognized layer type:{}, only support {}'.format(
                    layer_type, map_types))
        return self.quantizable_type[map_types.index(layer_type)]

    def _transform_layer_types(self, layer_types):
        map_types = self.capacity.get_value('QUANTIZABLE_MAP_TYPES')
        if map_types is None:
            return layer_types

        if not isinstance(layer_types, list):
            return self._get_mapped_type(layer_types, map_types)

        ret = []
        for item in layer_types:
            ret.append(self._get_mapped_type(item, map_types))
        return ret

    def _get_skip_layer_types(self):
        skip_layer_types = list(self.proto_config.skip_layer_types)
        repeated_items = find_repeated_items(skip_layer_types)
        check_no_repeated(repeated_items, 'skip_layer_types')
        return self._transform_layer_types(skip_layer_types)

    def _get_do_fusion(self):
        if hasattr(self.proto_config, 'do_fusion'):
            return self.proto_config.do_fusion
        return True

    def _get_skip_fusion_layers(self):
        if not hasattr(self.proto_config, 'skip_fusion_layers'):
            return []
        skip_fusion_layers = list(self.proto_config.skip_fusion_layers)
        repeated_layers = find_repeated_items(skip_fusion_layers)
        check_no_repeated(repeated_layers, 'skip_fusion_layers')
        return skip_fusion_layers

    def _parser_calibration_config(self, config):
        layer_config = OrderedDict()
        act_params = OrderedDict()
        weight_params = OrderedDict()

        if config.HasField('ifmr_quantize'):
            self._get_ifmr_config(config, act_params)
        elif hasattr(config, 'hfmg_quantize') and config.HasField('hfmg_quantize'):
            self._get_hfmg_config(config, act_params)

        if config.HasField('arq_quantize'):
            arq_quantize = config.arq_quantize
            if arq_quantize.HasField(CHANNEL_WISE):
                weight_params[CHANNEL_WISE] = arq_quantize.channel_wise
            if hasattr(arq_quantize, 'quant_bits') and arq_quantize.HasField('quant_bits'):
                weight_params['num_bits'] = arq_quantize.quant_bits
        elif hasattr(config, 'ada_quantize') and config.HasField('ada_quantize'):
            weight_params['wts_algo'] = 'ada_quantize'
            self._get_ada_quantize(config.ada_quantize, weight_params)
        layer_config[ACTIVATION_QUANT_PARAMS] = act_params
        layer_config[WEIGHT_QUANT_PARAMS] = weight_params

        if hasattr(config, 'dmq_balancer') and config.HasField('dmq_balancer'):
            migration_strength = proto_float_to_python_float(config.dmq_balancer.migration_strength)
            layer_config['dmq_balancer_param'] = migration_strength

        return layer_config

    def _get_common_config(self):
        if self.proto_config.HasField('common_config'):
            common_config = self.proto_config.common_config
            return self._parser_calibration_config(common_config)
        return OrderedDict()

    def _get_override_layer_types(self):
        types = [item.layer_type for item in self.proto_config.override_layer_types]
        repeated_items = find_repeated_items(types)
        check_no_repeated(repeated_items, 'override_layer_types')

        override_types = OrderedDict()
        for item in self.proto_config.override_layer_types:
            layer_type = item.layer_type
            layer_type = self._transform_layer_types(layer_type)
            if layer_type not in self.quantizable_type:
                raise ValueError("Layer type {} does not support "
                                 "quantize.".format(layer_type))

            override_types[layer_type] = self._parser_calibration_config(item.calibration_config)
            symmetric_limit_types = self.graph_querier.get_act_symmetric_limit_types()
            if layer_type in symmetric_limit_types:
                if override_types.get(layer_type).get(ACTIVATION_QUANT_PARAMS).get(ASYMMETRIC):
                    raise ValueError('Layer type {} can only set asymmetric be False '.format(layer_type))
            if layer_type not in self.channel_wise_types and \
                self._channel_wise_is_true(override_types.get(layer_type)):
                raise ValueError('channel_wise can only be False '
                                 'for {} type'.format(layer_type))
            if override_types.get(layer_type).get('dmq_balancer_param') and \
                layer_type not in self.graph_querier.get_support_dmq_balancer_types():
                raise ValueError('dmq_balancer not support '
                                 'for {} type'.format(layer_type))
            if override_types.get(layer_type).get(ACTIVATION_QUANT_PARAMS).get('num_bits') == 16 and \
                layer_type not in self.int16_quantizable_type:
                raise ValueError('int16_quantizable not support '
                                 'for {} type'.format(layer_type))
            if override_types.get(layer_type).get(WEIGHT_QUANT_PARAMS).get('wts_algo') == 'ada_quantize' and \
                layer_type not in self.ada_round_quantizable_type:
                raise ValueError('ada_quantizable not support '
                                 'for {} layer'.format(layer_type))
            if hasattr(self.graph_querier, 'get_support_winograd_layer_types'):
                winograd_support_layer_type = self.graph_querier.get_support_winograd_layer_types()
            else:
                winograd_support_layer_type = list()
            wts_num_bits = override_types.get(layer_type).get(WEIGHT_QUANT_PARAMS).get('num_bits')
            if wts_num_bits in WINOGRAD_NUM_BITS and \
                layer_type not in winograd_support_layer_type:
                raise ValueError('quant_bits {} not support '
                                 'for {} type'.format(wts_num_bits, layer_type))
        return override_types

    def _get_override_layer_configs(self):
        names = [item.layer_name for item in self.proto_config.override_layer_configs]
        repeated_items = find_repeated_items(names)
        check_no_repeated(repeated_items, 'override_layer_configs')
        override_layers = OrderedDict()
        for item in self.proto_config.override_layer_configs:
            layer_name = item.layer_name
            if layer_name not in self.layer_type:
                raise ValueError('Layer {} does not exist in '
                                 'the graph.'.format(layer_name))

            override_layers[layer_name] = self._parser_calibration_config(item.calibration_config)
            symmetric_limit_layers = self.graph_querier.get_act_symmetric_limit_layers(self.graph)
            if layer_name in symmetric_limit_layers:
                if override_layers.get(layer_name).get(ACTIVATION_QUANT_PARAMS).get(ASYMMETRIC):
                    raise ValueError('Layer {} can only set asymmetric be False '.format(layer_name))
            if self.layer_type.get(layer_name) not in self.channel_wise_types and \
                self._channel_wise_is_true(override_layers.get(layer_name)):
                raise ValueError('channel_wise can only be False '
                                 'for {} layer.'.format(layer_name))
            if override_layers.get(layer_name).get('dmq_balancer_param') and \
                layer_name not in self.graph_querier.get_support_dmq_balancer_layers(self.graph):
                raise ValueError('dmq_balancer not support '
                                 'for {} layer'.format(layer_name))
            if override_layers.get(layer_name).get(ACTIVATION_QUANT_PARAMS).get('num_bits') == 16 and \
                layer_name not in self.graph_querier.get_support_int16_quantizable_layers(self.graph):
                raise ValueError('int16_quantizable not support '
                                 'for {} layer'.format(layer_name))
            if override_layers.get(layer_name).get(WEIGHT_QUANT_PARAMS).get('wts_algo') == 'ada_quantize' and \
                layer_name not in self.graph_querier.get_ada_quant_layers(self.graph):
                raise ValueError('ada_quantizable not support '
                                 'for {} layer'.format(layer_name))
            wts_quant_bits = override_layers.get(layer_name).get(WEIGHT_QUANT_PARAMS).get('num_bits')
            if wts_quant_bits in WINOGRAD_NUM_BITS and \
                layer_name not in self.graph_querier.get_support_winograd_quant_layers(self.graph):
                raise ValueError('quant_bits {} not support '
                                 'for {} layer'.format(wts_quant_bits, layer_name))
        return override_layers

    def _parse_deprecated_config(self, name, config):
        LOGGER.logi('{} field has been deprecated, use common_config and '
                    'override_layer_types instead.'.format(name))
        return self._parser_calibration_config(config)

    def _get_conv_calibration_config(self):
        name = CONV_CALIBRATION_CONFIG
        if not self.proto_config.HasField(name):
            return OrderedDict()
        config = self.proto_config.conv_calibration_config
        return self._parse_deprecated_config(name, config)

    def _get_fc_calibration_config(self):
        name = FC_CALIBRATION_CONFIG
        if not self.proto_config.HasField(name):
            return OrderedDict()
        config = self.proto_config.fc_calibration_config
        fc_config = self._parse_deprecated_config(name, config)
        if self._channel_wise_is_true(fc_config):
            raise ValueError('channel_wise can only be False '
                             'for fc_calibration_config field')
        return fc_config

    def _raise_ignore_info(self, name):
        if self.proto_config.HasField(name):
            LOGGER.logw('{} field would be ignored when common_config '
                        'or override_layer_types exists'.format(name))

    def _get_global_config(self):
        """parse global config"""
        global_config = OrderedDict()
        if hasattr(self.proto_config, BATCH_NUM):
            global_config[BATCH_NUM] = self._get_batch_num()
        global_config['activation_offset'] = self._get_activation_offset()
        if hasattr(self.proto_config, 'joint_quant'):
            global_config['joint_quant'] = self._get_joint_quant()
        if hasattr(self.proto_config, 'weight_offset'):
            global_config['weight_offset'] = self._get_weight_offset()
        if hasattr(self.proto_config, 'fakequant_precision_mode'):
            global_config['fakequant_precision_mode'] = self._get_fakequant_precision_mode()
        global_config['skip_layers'] = self._get_skip_layers()
        global_config['skip_layer_types'] = self._get_skip_layer_types()
        global_config['do_fusion'] = self._get_do_fusion()
        global_config['skip_fusion_layers'] = self._get_skip_fusion_layers()
        return global_config

    def _get_approx_global_config(self):
        """parse global config for op approximation"""
        global_config = OrderedDict()
        if hasattr(self.proto_config, BATCH_NUM):
            global_config[BATCH_NUM] = self._get_batch_num()
        global_config['skip_approximation_layers'] = self._get_skip_approx_layers()
        return global_config

    def _get_tensor_quantize_config(self, tensor_configs):
        '''parse config of tensor_quantize'''
        tensor_quantizes = []
        for layer_config in tensor_configs:
            tensor_quantize = {}
            if not layer_config.HasField('layer_name'):
                raise RuntimeError('To quantize tensor, must set "layer_name" at first.')
            if not layer_config.HasField('input_index'):
                raise RuntimeError('To quantize tensor, must set "input_index" at first.')
            tensor_quantize['layer_name'] = layer_config.layer_name
            tensor_quantize['input_index'] = layer_config.input_index
            quantize_params = {}
            if hasattr(layer_config, 'hfmg_quantize') and layer_config.HasField('hfmg_quantize'): # default set to ifmr
                self._get_hfmg_config(layer_config, quantize_params)
            else:
                self._get_ifmr_config(layer_config, quantize_params)
            tensor_quantize[ACTIVATION_QUANT_PARAMS] = quantize_params
            tensor_quantizes.append(tensor_quantize)
        return tensor_quantizes
