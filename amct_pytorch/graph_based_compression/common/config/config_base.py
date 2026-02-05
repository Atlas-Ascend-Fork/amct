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
import json
import copy
from collections import namedtuple
from collections import OrderedDict

from ..utils.util import find_repeated_items
from ..utils.util import check_no_repeated
from ..utils.vars_util import WINOGRAD_NUM_BITS, DEFAULT_NUM_BITS
from .proto_config import ProtoConfig
from .field import VersionField
from .field import BatchNumField
from .field import ActOffsetField
from .field import JointQuantField
from .field import WtsOffsetField
from .field import DoFusionField
from .field import SkipFusionLayersField
from .field import ContainerField
from .field import LayerPlhField
from .field import QuantEnableField
from .field import ActQuantParamsField
from .field import MaxPercentileField
from .field import MinPercentileField
from .field import SearchRangeField
from .field import SearchStepField
from .field import WgtQuantParamsField
from .field import WtsAlgoField
from .field import ActAlgoField
from .field import ChannelWiseField
from .field import NumOfBinsField
from .field import TensorQuantizeField
from .field import LayerNameField
from .field import InputIndexField
from .field import ActNumBitsField
from .field import WtsNumBitsField
from .field import PARAM_POOL
from .field import ApproximateAlgoField
from .field import LayerActOffsetField
from .field import DMQBalancerParamField
from .field import FakequantPrecisionModeField
from .field import RegParamField
from .field import NumIterationField
from .field import WarmStartField
from .field import BetaRangeField
from ...utils.log import LOGGER # pylint: disable=relative-beyond-top-level
from ..utils.files import create_empty_file
from ...proto import calibration_config_pb2  # pylint: disable=import-error, relative-beyond-top-level
try:
    from ...proto import calibration_config_ascend_pb2  # pylint: disable=import-error, relative-beyond-top-level
except ImportError as ex:
    pass
from ..capacity.query_capacity import ASCEND_CAPACITY_TYPE


_MODULE_NAME = 'Config'
GraphObjects = namedtuple('GraphObjects', ['graph_querier', 'graph_checker'])
ACTIVATION_QUANT_PARAMS = 'activation_quant_params'
WEIGHT_QUANT_PARAMS = 'weight_quant_params'
SEARCH_RANGE = 'search_range'
BATCH_NUM = 'batch_num'
ASYMMETRIC = 'asymmetric'
BETA_RANGE = 'beta_range'
FAKEQUANT_PRECISION_MODE = 'fakequant_precision_mode'


class ConfigBase():
    """
    Function:
    APIs:
    """
    def __init__(self, graph_objects, capacity, enable_quant=True, enable_approximate=False):
        self.enable_quant = enable_quant
        self.enable_approximate = enable_approximate
        if self.enable_quant and self.enable_approximate:
            raise RuntimeError(
                "Do not support activation of quantization and"
                "approximation at the same time")
        self.root = ContainerField(capacity)
        root = self.root
        root.add_child('version', VersionField(capacity))
        if capacity.is_enable('BATCH_NUM'):
            root.add_child(BATCH_NUM, BatchNumField(capacity))
        if self.enable_quant:
            self._init_quant_tree(root, capacity)
        elif self.enable_approximate:
            self._init_approximate_tree(capacity)
        self.capacity = capacity
        self.quantizable_type = capacity.get_value('QUANTIZABLE_TYPES')
        self.approximate_type = capacity.get_value('OP_APPROXIMATION_TYPES')
        self.graph_querier = graph_objects.graph_querier
        self.graph_checker = graph_objects.graph_checker

    @staticmethod
    def get_common_activation_quant_config(common_config):
        '''get activation quantize type and asymmetric'''
        act_quantize = None
        act_asymmetric = None
        if not common_config:
            return [act_quantize, act_asymmetric]
        act_quantize = common_config.get(ACTIVATION_QUANT_PARAMS).get('act_algo')
        act_asymmetric = common_config.get(ACTIVATION_QUANT_PARAMS).get(ASYMMETRIC)

        return [act_quantize, act_asymmetric]

    @staticmethod
    def set_global_asymmetric(config, act_common_config, activation_offset):
        '''set current global asymmetric for config'''
        act_algo, asymmetric = act_common_config
        if config.get('act_algo') is act_algo and asymmetric is not None:
            config[ASYMMETRIC] = asymmetric
        else:
            config[ASYMMETRIC] = activation_offset

    @staticmethod
    def add_global_to_layer(quant_config, num_bits=None, wts_algo=None):
        """add global quantize parameter to each layer"""
        for item in quant_config:
            if isinstance(quant_config.get(item), dict):
                # 1. adjust act_quant_params
                # 1.1 split search_range
                act_config = quant_config.get(item).get(ACTIVATION_QUANT_PARAMS)
                # default act calibration algo is ifmr
                act_algo = act_config.get('act_algo', 'ifmr')
                if act_algo == 'ifmr':
                    act_config["search_range_start"] = \
                        act_config.get(SEARCH_RANGE)[0]
                    act_config["search_range_end"] = \
                        act_config.get(SEARCH_RANGE)[1]
                    del act_config[SEARCH_RANGE]
                # 1.2 add activation_offset
                if quant_config.get(item).get(ACTIVATION_QUANT_PARAMS).get(ASYMMETRIC) is None:
                    with_offset = quant_config.get('activation_offset')
                else:
                    with_offset = quant_config.get(item).get(ACTIVATION_QUANT_PARAMS).get(ASYMMETRIC)
                act_config['with_offset'] = with_offset
                if quant_config.get(BATCH_NUM) is not None:
                    act_config[BATCH_NUM] = quant_config.get(BATCH_NUM)
                if quant_config.get(FAKEQUANT_PRECISION_MODE) is not None:
                    act_config[FAKEQUANT_PRECISION_MODE] = quant_config.get(FAKEQUANT_PRECISION_MODE)

                # 2. adjust weight_quant_params
                weight_config = quant_config.get(item).get("weight_quant_params")
                weight_config["with_offset"] = \
                    quant_config.get("weight_offset", False)
                if num_bits is not None:
                    weight_config["num_bits"] = num_bits
                if wts_algo is not None:
                    weight_config["wts_algo"] = wts_algo

    @staticmethod
    def _add_act_param(container, capacity):
        act_param = \
            container.add_child(ACTIVATION_QUANT_PARAMS,
                                ActQuantParamsField(capacity))
        act_param.add_child('num_bits', ActNumBitsField(capacity))
        act_param.add_child('max_percentile', MaxPercentileField(capacity))
        act_param.add_child('min_percentile', MinPercentileField(capacity))
        act_param.add_child(SEARCH_RANGE, SearchRangeField(capacity))
        act_param.add_child('search_step', SearchStepField(capacity))
        act_param.add_child('act_algo', ActAlgoField(capacity))
        act_param.add_child(ASYMMETRIC, LayerActOffsetField(capacity))
        if capacity.is_enable('HFMG'):
            act_param.add_child('num_of_bins', NumOfBinsField(capacity))

    @staticmethod
    def _add_wts_param(wts_param, capacity):
        wts_param.add_child('reg_param', RegParamField(capacity))
        wts_param.add_child('warm_start', WarmStartField(capacity))
        wts_param.add_child('num_iteration', NumIterationField(capacity))
        wts_param.add_child(BETA_RANGE, BetaRangeField(capacity))

    def create_config_from_proto(self, config_file, graph, config_defination):
        '''
        create json config file from proto config.
        Inputs:
            config_file: json config file path
            graph: a graph
            config_defination: proto config path
        Returns:
            None
        '''
        proto_config = ProtoConfig(os.path.realpath(config_defination),
                                   self.capacity,
                                   self.graph_querier,
                                   graph,
                                   self._get_proto_config())
        global_config, common_config, type_config, layer_config = \
            proto_config.get_proto_config(self.enable_quant, self.enable_approximate)

        if self.enable_approximate:
            return self._create_approx_config_from_proto(global_config, graph)

        supported_layers, layer_type = self._find_quant_layers(global_config, graph)
        config = OrderedDict()
        for item in self.root.get_keys():
            if item in global_config and global_config.get(item) is not None:
                config[item] = global_config.get(item)

        for item in supported_layers:
            if item in layer_config:
                config[item] = layer_config.get(item)
            elif layer_type.get(item) in type_config:
                config[item] = copy.deepcopy(type_config.get(layer_type.get(item)))
            else:
                config[item] = copy.deepcopy(common_config)
        self.root.fill_default(config)

        if common_config.get('dmq_balancer_param'):
            self.check_common_config_dmq_layers(graph)
        self.remove_config_not_support_dmq_balancer(graph, config, supported_layers)
        if common_config.get(ACTIVATION_QUANT_PARAMS) and \
            common_config.get(ACTIVATION_QUANT_PARAMS).get('num_bits', 8) == 16:
            self.check_int16_quantize_layers(graph, config, supported_layers)

        self.check_and_down_grade_winograd_num_bits(graph, config, supported_layers)
        act_common_config = self.get_common_activation_quant_config(common_config)
        self._fill_default_activation_asymmetric(graph, config, supported_layers, act_common_config)
        self.root.check(None, config)
        if config.get(FAKEQUANT_PRECISION_MODE) == 'DEFAULT':
            del config[FAKEQUANT_PRECISION_MODE]
        if common_config.get(WEIGHT_QUANT_PARAMS) and \
            common_config.get(WEIGHT_QUANT_PARAMS).get('wts_algo') == 'ada_quantize':
            self.check_ada_quantize_layers(graph, config, supported_layers)
        ordered_config = self.root.sort(config)
        config_file = create_empty_file(config_file, check_exist=True)
        with open(config_file, 'w') as fid:
            json.dump(ordered_config, fid, indent=4, separators=(',', ':'))

        LOGGER.logi('Generate config file:{} success!'.format(
            config_file), module_name=_MODULE_NAME)

    def check_ada_quantize_layers(self, graph, config, supported_layers):
        for layer in supported_layers:
            if config.get(layer).get(WEIGHT_QUANT_PARAMS).get('wts_algo') == 'ada_quantize' and \
                layer not in self.graph_querier.get_ada_quant_layers(graph):
                del config[layer][WEIGHT_QUANT_PARAMS]

    def check_common_config_dmq_layers(self, graph):
        ''' check common_config layer support dmq_balancer '''
        quant_layers = PARAM_POOL.get_quant_layers()
        if not [layer for layer in quant_layers if layer in self.graph_querier.get_support_dmq_balancer_layers(graph)]:
            LOGGER.logw("Except skip_layers or skip_types, no layer in graph support dmq_balancer, "
                "please remove dmq_balancer in config file.")

    def check_int16_quantize_layers(self, graph, config, supported_layers):
        ''' check layer support int 16 ptq quantize '''
        for layer in supported_layers:
            if config.get(layer).get(ACTIVATION_QUANT_PARAMS).get('num_bits') == 16 and \
                layer not in self.graph_querier.get_support_int16_quantizable_layers(graph):
                config.get(layer).get(ACTIVATION_QUANT_PARAMS)['num_bits'] = 8
                LOGGER.logw("Layer {} does not support int16_quantizable, set int8".format(layer))

    def check_and_down_grade_winograd_num_bits(self, graph, config, supported_layers):
        '''check quant num bits and turn it into 8 if it not support int6 int7 quant'''
        for layer in supported_layers:
            wts_quant_bits = config.get(layer).get('weight_quant_params').get('num_bits')
            if wts_quant_bits in WINOGRAD_NUM_BITS and \
                layer not in self.graph_querier.get_support_winograd_quant_layers(graph):
                config.get(layer).get('weight_quant_params')['num_bits'] = DEFAULT_NUM_BITS
                LOGGER.logi("Layer {} does not support weight num bits {}, set int8".format(layer, wts_quant_bits))


    def remove_config_not_support_dmq_balancer(self, graph, config, supported_layers):
        ''' remove not support dmq_balancer layer in config'''
        for layer in supported_layers:
            if config.get(layer).get('dmq_balancer_param') and \
                layer not in self.graph_querier.get_support_dmq_balancer_layers(graph):
                config.get(layer).pop('dmq_balancer_param')
                LOGGER.logw("Layer {} does not support dmq_balancer, " \
                    "remove dmq_balancer in config file".format(layer))

    def check_skip_layers(self, graph, skip_layers):
        '''check skip layers'''
        repeated_items = find_repeated_items(skip_layers)
        check_no_repeated(repeated_items, 'skip_layers')
        layer_type = self.graph_querier.get_name_type_dict(graph)
        for layer in skip_layers:
            if layer not in layer_type:
                raise ValueError(
                    "Layer {} does not exist in the graph.".format(layer))
        if self.enable_quant:
            target_type = self.quantizable_type
            feature = "quantization"
        elif self.enable_approximate:
            target_type = self.approximate_type
            feature = "approximation"
        for layer in skip_layers:
            if layer_type.get(layer) not in target_type:
                raise ValueError("Layer {} does not support {}.".format(layer, feature))

    def check_skip_types(self, skip_types):
        '''check skip types'''
        repeated_items = find_repeated_items(skip_types)
        check_no_repeated(repeated_items, 'skip_types')

        for item in skip_types:
            if item not in self.quantizable_type:
                raise ValueError(
                    "Layer type {} does not support quantize.".format(item))

        quant_type = set(self.quantizable_type) - set(skip_types)
        if not quant_type:
            raise ValueError('Except skip_types, no type need to quantize.')

    def get_supported_layers(self, graph, tensor_quant_valid=False):
        '''get supported layers'''
        if self.enable_quant:
            supported_layers = self.graph_querier.get_support_quant_layers(graph)
            feature = 'quantization'
        elif self.enable_approximate:
            supported_layers = self.graph_querier.get_support_approximate_layers(graph)
            feature = 'approximation'
        if not supported_layers and not tensor_quant_valid:
            raise ValueError('No layer support {} in the graph.'.format(feature))
        for item in self.root.get_keys():
            if item in supported_layers:
                raise ValueError('{} is a global parameter, '
                                 'can not be a layer name'.format(item))
        return supported_layers

    def check_quant_tensor_valid(self, graph, quant_tensors):
        '''
        Function: check quant tensor's validation by the graph, note that
        tensor quantization is only valid for tensorflow network
        '''
        if hasattr(self.graph_checker, 'check_tensor_quant'):
            self.graph_checker.check_tensor_quant(graph, quant_tensors)

    def set_param_pool(self, quant_layers, graph):
        '''set params used by field'''
        if self.enable_quant:
            supported_layers = self.graph_querier.get_support_quant_layers(graph)
        elif self.enable_approximate:
            supported_layers = self.graph_querier.get_support_approximate_layers(graph)
        layer_type = self.graph_querier.get_name_type_dict(graph)
        PARAM_POOL.clear()
        PARAM_POOL.set_quant_layers(quant_layers)
        PARAM_POOL.set_supported_layers(supported_layers)
        PARAM_POOL.set_layer_type(layer_type)

    def set_skip_layers(self, graph):
        """ set the skip layers for onnx"""
        skip_layers = self.graph_querier.get_skip_quant_layers(graph)
        PARAM_POOL.set_skip_layers(skip_layers)

    def create_quant_config(self, # pylint: disable=too-many-arguments
                            config_file,
                            graph,
                            skip_layers=None,
                            batch_num=1,
                            activation_offset=True):
        '''
        create json config file from parameters.
        Inputs:
            config_file: json config file path
            graph: a graph
            skip_layers: layers to skip quantize
            batch_num: the batch number used for calibration
            activation_offset: whether use offset for activation quant
        Returns:
            None
        '''
        self.check_skip_layers(graph, skip_layers)
        supported_layers = self.get_supported_layers(graph)
        quant_layers = set(supported_layers) - set(skip_layers)
        if not quant_layers:
            if self.enable_quant:
                raise ValueError('Except skip_layers, no layer needs to quantize.')
            elif self.enable_approximate:
                raise ValueError('Except skip_layers, no layer needs to approximate.')

        quant_config = {}
        # fill global parameters
        if self.capacity.is_enable('BATCH_NUM'):
            quant_config[BATCH_NUM] = batch_num

        self.set_param_pool(quant_layers, graph)

        if self.enable_quant:
            quant_config['activation_offset'] = activation_offset
            # set parameter
            self.root.check(None, quant_config)
            self.root.fill_default(quant_config)
            self._fill_default_activation_asymmetric(graph, quant_config,
                supported_layers, ['any', activation_offset])
            if quant_config.get(FAKEQUANT_PRECISION_MODE, None) == 'DEFAULT':
                del quant_config[FAKEQUANT_PRECISION_MODE]
            ordered_config = self.root.sort(quant_config)

            config_file = create_empty_file(config_file, check_exist=True)
            with open(config_file, 'w') as fid:
                json.dump(ordered_config, fid, indent=4, separators=(',', ':'))
            LOGGER.logi('Generate config file:{} success!'.format(
                config_file), module_name=_MODULE_NAME)
        elif self.enable_approximate:
            # set parameter
            self.root.check(None, quant_config)
            self.root.fill_default(quant_config)
            if quant_config.get(FAKEQUANT_PRECISION_MODE, None) == 'DEFAULT':
                del quant_config[FAKEQUANT_PRECISION_MODE]
            ordered_config = self.root.sort(quant_config)
            return ordered_config

    def check_activation_symmetric_valid(self, graph, config, layer_name):
        '''check current layer if support symmetric'''
        symmetric_limit_layers = self.graph_querier.get_act_symmetric_limit_layers(graph)
        if layer_name in symmetric_limit_layers and \
            config.get(ACTIVATION_QUANT_PARAMS).get(ASYMMETRIC) is True:
            raise ValueError('current layer {} only support act symmetric quant.'.format(layer_name))

    def check_int16_quantize_valid(self, graph, layer_name):
        '''check current layer if support int16 quantize'''
        int16_quantizable_layers = self.graph_querier.get_support_int16_quantizable_layers(graph)
        if layer_name not in int16_quantizable_layers:
            raise ValueError('current layer {} does not support int16 quant.'.format(layer_name))

    def parse_config_file(self, config_file, graph):
        '''
        parse config file, check incorrect value, and fill default value.
        Inputs:
            config_file: calibration json config file path
            graph: a graph
        Returns:
            quant_config: calibration quant config dict
        '''
        def _detect_repetitive_key_hook(lst):
            '''a hook function for detect repeated key in config file.'''
            keys = [key for key, value in lst]
            repeat_keys = find_repeated_items(keys)
            check_no_repeated(repeat_keys, config_file)
            result = {}
            for key, value in lst:
                result[key] = value
            return result

        with open(config_file, 'r') as fid:
            quant_config = json.load(
                fid, object_pairs_hook=_detect_repetitive_key_hook)

        quant_layers = []
        layer_type = self.graph_querier.get_name_type_dict(graph)
        for item in quant_config:
            if item in self.root.get_keys():
                continue
            if item in layer_type and isinstance(quant_config.get(item), dict) \
                and quant_config.get(item).get(ACTIVATION_QUANT_PARAMS).get('num_bits', 8) == 16:
                self.check_int16_quantize_valid(graph, item)
            if item in layer_type and isinstance(quant_config.get(item), dict) \
                and quant_config.get(item).get('quant_enable', False):
                self.check_activation_symmetric_valid(graph, quant_config.get(item), item)
                quant_layers.append(item)

        # check quant layer's validation by the graph
        if quant_config.get('tensor_quantize'):
            self.check_quant_tensor_valid(graph, quant_config.get('tensor_quantize'))
        self.set_param_pool(quant_layers, graph)
        self.root.check(None, quant_config)
        self.root.fill_default(quant_config)

        for item in list(quant_config):
            if item in self.root.get_keys():
                continue
            if item not in quant_layers:
                quant_config.pop(item)
        LOGGER.logd('quant_config:', module_name=_MODULE_NAME)
        for item in quant_config.items():
            LOGGER.logd(item, module_name=_MODULE_NAME)
        return quant_config

    def _init_quant_tree(self, root, capacity):
        '''init config tree for post training quantization'''
        root.add_child('activation_offset', ActOffsetField(capacity))
        if capacity.is_enable('JOINT_QUANT'):
            root.add_child('joint_quant', JointQuantField(capacity))
        if capacity.is_enable('WEIGHT_OFFSET'):
            root.add_child('weight_offset', WtsOffsetField(capacity))

        root.add_child('do_fusion', DoFusionField(capacity))
        root.add_child('fakequant_precision_mode', FakequantPrecisionModeField(capacity))
        root.add_child('skip_fusion_layers', SkipFusionLayersField(capacity))

        layer_plh = root.add_placeholder(LayerPlhField(capacity))
        layer_plh.add_child('quant_enable', QuantEnableField(capacity))
        layer_plh.add_child('dmq_balancer_param', DMQBalancerParamField(capacity))
        self._add_act_param(layer_plh, capacity)

        wgt_param = \
            layer_plh.add_child('weight_quant_params',
                                WgtQuantParamsField(capacity))

        wgt_param.add_child('num_bits', WtsNumBitsField(capacity))
        self._add_wts_param(wgt_param, capacity)
        wgt_param.add_child('channel_wise', ChannelWiseField(capacity))

        if capacity.is_enable('TENSOR_QUANTIZE'):
            tensor_container = root.add_child(
                'tensor_quantize', TensorQuantizeField(capacity))
            tensor_container.add_child('layer_name',
                LayerNameField(capacity))
            tensor_container.add_child('input_index',
                InputIndexField(capacity))
            self._add_act_param(tensor_container, capacity)

    def _init_approximate_tree(self, capacity):
        '''init config tree for post training op approximation'''
        layer_plh = self.root.add_placeholder(LayerPlhField(capacity))
        layer_plh.add_child('approximate_enable', QuantEnableField(capacity))
        layer_plh.add_child('approximate_algo', ApproximateAlgoField(capacity))

    def _get_proto_config(self):
        if self.capacity.get_capacity_type() == ASCEND_CAPACITY_TYPE:
            return calibration_config_ascend_pb2.AMCTConfig()
        return calibration_config_pb2.AMCTConfig()

    def _find_quant_layers(self, global_config, graph):
        '''find and check quantable layers in the graph'''
        skip_layers = global_config.get('skip_layers')
        self.check_skip_layers(graph, skip_layers)
        skip_types = global_config.get('skip_layer_types')
        self.check_skip_types(skip_types)
        quant_layers = []
        # step1: check tensor quant is valid, or raise Error in check
        tensor_quant_valid = False
        if global_config.get('tensor_quantize'):
            self.check_quant_tensor_valid(graph, global_config.get('tensor_quantize'))
            tensor_quant_valid = True
            LOGGER.logi("Check quant tensor success!")
        # step2: check layers in QUANTIZABLE_TYPES
        supported_layers = self.get_supported_layers(graph, tensor_quant_valid)
        # step3: remove skiped type and layer
        layer_type = self.graph_querier.get_name_type_dict(graph)
        for item in supported_layers:
            if layer_type.get(item) in skip_types:
                continue
            if item in skip_layers:
                continue
            quant_layers.append(item)
        # step4: check quant_layers empty and tensor quant invalid
        if not quant_layers and not tensor_quant_valid:
            raise ValueError('Except skip_layers or skip_types, no layer need to quantize.')

        self.set_param_pool(quant_layers, graph)
        return supported_layers, layer_type

    def _find_approx_layers(self, global_config, graph):
        '''find and check approximatable ops in the graph'''
        skip_approx_layers = global_config.get('skip_approximation_layers')
        self.check_skip_layers(graph, skip_approx_layers)
        approx_layers = []
        supported_layers = self.get_supported_layers(graph)
        layer_type = self.graph_querier.get_name_type_dict(graph)
        for item in supported_layers:
            if item in skip_approx_layers:
                continue
            approx_layers.append(item)
        if not approx_layers:
            raise ValueError('Except skip_layers or skip_types, '
                             'no layer needs to be approximated.')
        self.set_param_pool(approx_layers, graph)
        return supported_layers, layer_type

    def _create_approx_config_from_proto(self, global_config, graph):
        '''create config from proto for op approximation'''
        self._find_approx_layers(global_config, graph)
        config = OrderedDict()
        for item in self.root.get_keys():
            if item in global_config and global_config.get(item) is not None:
                config[item] = global_config.get(item)
        self.root.fill_default(config)
        self.root.check(None, config)
        ordered_config = self.root.sort(config)
        return ordered_config

    def _fill_default_activation_asymmetric(self, graph, config, supported_layers, act_common_config):
        '''fill default asymmetric value for activation_quant_params.'''
        symmetric_limit_layers = self.graph_querier.get_act_symmetric_limit_layers(graph)
        for item in supported_layers:
            if item in symmetric_limit_layers:
                config.get(item).get(ACTIVATION_QUANT_PARAMS)[ASYMMETRIC] = False
                LOGGER.logi("Layer {} does not support asymmetric quant, set symmetric".format(item))
            if config.get(item).get(ACTIVATION_QUANT_PARAMS).get(ASYMMETRIC) is None:
                self.set_global_asymmetric(config.get(item).get(ACTIVATION_QUANT_PARAMS),
                    act_common_config, config.get('activation_offset'))
        # fill asymmetric value for tensor_quantize
        for item in config.get('tensor_quantize', []):
            if item.get(ACTIVATION_QUANT_PARAMS).get(ASYMMETRIC) is None:
                self.set_global_asymmetric(item.get(ACTIVATION_QUANT_PARAMS),
                    act_common_config, config.get('activation_offset'))


def check_config_quant_enable(quant_config):
    """ check whether no quant enable layer"""
    quant_enable_layers = []
    for key, _ in quant_config.items():
        if not isinstance(quant_config[key], dict):
            continue
        if quant_config.get(key).get("quant_enable"):
            quant_enable_layers.append(key)
    if quant_config.get('tensor_quantize'):
        quant_enable_layers.append('tensor_quantize')
    if quant_enable_layers == []:
        LOGGER.loge(
            "No quant enable layer in quant config file, "
            "please check the quant config file.", module_name=_MODULE_NAME)
        raise RuntimeError(
            "No quant enable layer in quant config file, "
            "please check the quant config file.")


def check_config_dmq_balancer(quant_config):
    """ check whether no dmq_balancer layer """
    for key, _ in quant_config.items():
        if not isinstance(quant_config[key], dict):
            continue
        if quant_config[key].get("dmq_balancer_param"):
            return
    LOGGER.loge(
        "No dmq_balancer layer in quant config file, please check whether the quant config file matches "
        "or whether a layer in graph that supports dmq_balancer.", module_name=_MODULE_NAME)
    raise RuntimeError(
        "No dmq_balancer layer in quant config file, "
        "please check whether the quant config file matches or whether a layer in graph that supports dmq_balancer.")
