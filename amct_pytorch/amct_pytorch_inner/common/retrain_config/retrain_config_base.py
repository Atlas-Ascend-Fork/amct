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
from ..utils.util import find_repeated_items
from ..utils.util import check_no_repeated
from .retrain_field import RootConfig
from .retrain_field import LayerConfig
from .retrain_proto import RetrainProtoConfig
from ...utils.log import LOGGER
from ..utils.files import create_empty_file
from ..utils.vars_util import INT4, INT8
from ..utils.vars_util import RETRAIN_ACT_WTS_TYPES
from .retrain_proto import FILTER
from .retrain_proto import SELECTIVE
from .retrain_proto import NO_PRUNE

GraphObjects = namedtuple('GraphObjects', ['graph_querier', 'graph_checker'])

_MODULE_NAME = 'RetrainConfigBase'
DST_TYPE = 'dst_type'
RETRAIN_ENABLE = 'retrain_enable'
RETRAIN_DATA_CONFIG = 'retrain_data_config'
RETRAIN_WEIGHT_CONFIG = 'retrain_weight_config'
BATCH_NUM = 'batch_num'
FAKEQUANT_PRECISION_MODE = 'fakequant_precision_mode'


class RetrainConfigBase():
    '''retraon config base'''
    def __init__(self, graph_objects, capacity):
        ''' inner method '''
        self.graph_objects = graph_objects
        self.capacity = capacity
        self.graph_querier = graph_objects.graph_querier
        self.graph_checker = graph_objects.graph_checker
        self.config_tree = RootConfig(self.graph_querier, self.capacity)
        self.supported_types = self.capacity.get_value('RETRAIN_TYPES')
        self.prune_type = NO_PRUNE
        self.prunable_types = {}
        self.prunable_types[FILTER] = self.capacity.get_value('PRUNABLE_TYPES')
        self.prunable_types[SELECTIVE] = self.capacity.get_value('SELECTIVE_PRUNABLE_TYPES')
        self.prunable_types[NO_PRUNE] = []
        self.enable_retrain = True
        self.enable_prune = False
        self.valid_layers = {}

    def set_ability(self, enable_retrain, enable_prune):
        """
        set ability to support quant retrain and prune retrain
        Params: enable_retrain, a bool
            enable_prune, a bool
        Return: None
        """
        self.enable_retrain = enable_retrain
        self.enable_prune = enable_prune

    def get_supported_layers(self, graph):
        ''' inner method for retrain quant '''
        # get qat layer2type dict if has function
        if hasattr(self.graph_querier, 'get_support_qat_layer2type'):
            return self.graph_querier.get_support_qat_layer2type(graph)
        # from ptq layers, choice which type in retrain type
        layers_info = self.graph_querier.get_name_type_dict(graph)
        quant_layers = self.graph_querier.get_support_quant_layers(graph)

        supported_layers = {}
        for layer, layer_type in layers_info.items():
            if layer_type in self.supported_types and layer in quant_layers:
                supported_layers[layer] = layer_type

        return supported_layers

    def set_config_by_graph_construct(self, ordered_config, graph):
        '''set the config by the graph's construction'''
        # only tensorflow need to check softmax and set channelwise
        if hasattr(self.graph_checker, 'set_softmax_channelwise'):
            self.graph_checker.set_softmax_channelwise(
                ordered_config, graph, RETRAIN_WEIGHT_CONFIG)

    def check_quant_layers_valid(self, graph, quant_layers):
        '''check quant layer's validation by the graph'''
        # only tensorflow need to check matmul_transpose and placeholder and
        # gradient_op
        if hasattr(self.graph_checker, 'check_data_type'):
            quant_layers = self.graph_checker.check_data_type(
                graph, quant_layers)
        if hasattr(self.graph_checker, 'check_gradient_op'):
            quant_layers = self.graph_checker.check_gradient_op(
                graph, quant_layers)
        if hasattr(self.graph_checker, 'check_matmul_transpose'):
            quant_layers = self.graph_checker.check_matmul_transpose(
                graph, quant_layers)
        if hasattr(self.graph_checker, 'check_quantize_placeholder'):
            quant_layers = self.graph_checker.check_quantize_placeholder(
                graph, quant_layers)
        return quant_layers

    def create_default_config(self, config_file, graph):
        ''' inner method, only generate quant retrain config '''
        if self.enable_prune:
            raise RuntimeError("this function cannot support prune.")
        if not self.enable_retrain:
            raise RuntimeError("this function only support retrain, but enable_retrain si False.")

        # qat support layers
        supported_layers = self.get_supported_layers(graph)
        if not supported_layers:
            raise ValueError("No supported layers")
        self._clear_config_tree()
        self.config_tree.build_default(supported_layers)
        ordered_config = self.config_tree.dump()
        self.set_config_by_graph_construct(ordered_config, graph)

        self._del_reduant_config(ordered_config, supported_layers, prune_supported_layers={})

        config_file = create_empty_file(config_file, check_exist=True)
        with open(config_file, 'w') as fid:
            json.dump(ordered_config, fid, indent=4, separators=(',', ':'))

    def generate_layer_config(self, proto, retrain_layers, no_default):
        """
        Generate layer config according to proto
        Params:
        proto
        retrain_layers: a dict, recording support_layers, qat_layers, prunable_layers
            supported_layers, a dict
            qat_layers: a list, layers enable to do quant
            prunable_layers: a dict, layers enable to do prune
        Return: config, a dict
        """
        if self.enable_prune:
            self.prune_type = proto.parse_prune_type()
        self._check_proto(proto, retrain_layers)

        config = {}
        if self.enable_retrain:
            _generate_retrain_config(proto, retrain_layers.get('qat_layers'), config, no_default)
        if self.enable_prune:
            _generate_prune_config(proto, retrain_layers.get('prunable_layers'), no_default, config,
                self.prune_type)

        return config

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
        config_defination = os.path.realpath(config_defination)
        proto = RetrainProtoConfig(config_defination, self.capacity)
        # Only compresse feature is both True.
        no_default = False
        if self.enable_prune and self.enable_retrain:
            # parse the simple proto, reset_ability.
            no_default = True
            proto_enable_quant, proto_enable_prune = proto.parse_proto_enable()
            self.set_ability(proto_enable_quant, proto_enable_prune)

        supported_layers, retrain_supported_layers, prune_supported_layers \
            = self.get_support_layers(graph)
        self._check_layer_empty(retrain_supported_layers, prune_supported_layers)
        retrain_layers = {'support_layers': supported_layers,
                          'qat_layers': retrain_supported_layers,
                          'prunable_layers': prune_supported_layers
                         }
        config = self.generate_layer_config(proto, retrain_layers, no_default)
        global_config = proto.get_proto_config()
        for item in global_config.keys():
            if global_config[item] is not None:
                config[item] = global_config[item]
        self._clear_config_tree()
        self.config_tree.set_strong_check(False)
        self.config_tree.build(config, supported_layers)
        ordered_config = self.config_tree.dump()
        self.set_config_by_graph_construct(ordered_config, graph)

        self._del_reduant_config(ordered_config, retrain_supported_layers, prune_supported_layers)
        if hasattr(self.graph_checker, 'check_weights_shared'):
            layer_names = get_layers_from_config(
                ordered_config, self.config_tree.get_global_keys())
            self.graph_checker.check_weights_shared(graph, ordered_config,
                                                    layer_names)

        if isinstance(config_file, dict):
            config_file.clear()
            config_file.update(ordered_config)
        else:
            config_file = create_empty_file(config_file, check_exist=True)
            with open(config_file, 'w') as fid:
                json.dump(ordered_config, fid, indent=4, separators=(',', ':'))

    def get_support_layers(self, graph):
        """
        get support layers from graph
        Params: graph
        Return: supported_layers, a dict
            retrain_supported_layers, a dict
            prune_supported_layers, a dict
        """
        supported_layers = {}
        retrain_supported_layers = {}
        prune_supported_layers = {}
        prune_supported_layers[NO_PRUNE] = {}
        # qat support layers
        if self.enable_retrain:
            retrain_supported_layers = self.get_supported_layers(graph)
            supported_layers = {**retrain_supported_layers}
        # prune support layers
        if self.enable_prune:
            selective_prune_supported_layers = \
                self.graph_querier.get_support_selective_prune_layer2type(graph)
            filter_prune_supported_layers = self.graph_querier.get_support_prune_layer2type(graph)
            supported_layers.update(selective_prune_supported_layers)
            supported_layers.update(filter_prune_supported_layers)
            prune_supported_layers[FILTER] = filter_prune_supported_layers
            prune_supported_layers[SELECTIVE] = selective_prune_supported_layers

        return supported_layers, retrain_supported_layers, prune_supported_layers

    def parse_config_file(self, config_file, graph):
        '''
        parse config file, check incorrect value, and fill default value.
        Inputs:
        config_file: json config file path
        graph: a graph
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

        valid_layers = self.get_supported_layers(graph)

        self._check_reduant_config(quant_config)
        self.config_tree.build(quant_config, valid_layers)
        quant_config = self.config_tree.dump()
        valid_layers_refilled = self.get_supported_layers(graph)
        self._del_reduant_config(quant_config, valid_layers_refilled, dict())

        if hasattr(self.graph_checker, 'check_weights_shared'):
            layer_names = get_layers_from_config(
                quant_config, self.config_tree.get_global_keys())
            self.graph_checker.check_weights_shared(graph, quant_config,
                                                    layer_names)

        LOGGER.logd('quant_config is {}'.format(quant_config),
                    module_name=_MODULE_NAME)

        quant_config['support_types'] = self.supported_types

        return quant_config

    def _clear_config_tree(self):
        ''' inner method '''
        self.config_tree = RootConfig(self.graph_querier, self.capacity)

    def _del_reduant_config(self, ordered_config, retrain_supported_layers, prune_supported_layers):
        # clear reduant config
        if not self.enable_retrain:
            del ordered_config[BATCH_NUM]
        for key in ordered_config:
            if key in self.config_tree.get_global_keys():
                continue
            # del retrain config in layer config
            if not retrain_supported_layers or key not in retrain_supported_layers:
                for layer_key in LayerConfig.retrain_fields():
                    del ordered_config[key][layer_key]
            # del prune config in layer config
            prune_config = ordered_config[key].get('regular_prune_config')
            if prune_config and prune_config.get('prune_type'):
                prune_type = prune_config['prune_type']
                if prune_supported_layers.get(prune_type) and key in prune_supported_layers[prune_type]:
                    continue
            for layer_key in LayerConfig.prune_fields():
                del ordered_config[key][layer_key]

        # del fakequant_precision_mode if DEFAULT
        if ordered_config[FAKEQUANT_PRECISION_MODE] == 'DEFAULT':
            del ordered_config[FAKEQUANT_PRECISION_MODE]
        # del no-key config
        check_list = list(ordered_config.keys())
        for key in check_list:
            if not ordered_config[key]:
                del ordered_config[key]

    def _check_reduant_config(self, ordered_config):
        if not self.enable_retrain:
            if ordered_config.get(BATCH_NUM):
                raise ValueError("not expect {} in config file if retrain is disable.".format(BATCH_NUM))
        for key in ordered_config:
            if key in self.config_tree.get_global_keys():
                continue
            if not isinstance(ordered_config[key], dict):
                raise ValueError("{} is not valid config, its value should be a dict if it is a layer.".format(key))
            # check retrain config in layer config
            if not self.enable_retrain:
                for layer_key in ordered_config[key].keys():
                    if layer_key in LayerConfig.retrain_fields():
                        raise ValueError("not expect {} in config file if retrain is disable.".format(layer_key))
            # check prune config in layer config
            if not self.enable_prune:
                for layer_key in ordered_config[key].keys():
                    if layer_key in LayerConfig.prune_fields():
                        raise ValueError("not expect {} in config file if prune is disable.".format(layer_key))

    def _check_proto_prune(self, proto, prunable_layers):
        regular_prune_skip_layers = proto.get_regular_prune_skip_layers()
        regular_prune_skip_types = proto.get_regular_prune_skip_types()

        _, _, prune_override_layers = proto.get_override_layers()
        _, _, prune_override_types = proto.get_override_layer_types()

        if not set(prune_override_layers.get(FILTER)).issubset(set(prunable_layers.get(FILTER).keys())):
            raise ValueError("some override_layer not in valid_layers for filter prune")
        if not set(prune_override_layers.get(SELECTIVE)).issubset(set(prunable_layers.get(SELECTIVE).keys())):
            raise ValueError("some override_layer not in valid_layers for selective prune")
        if not set(regular_prune_skip_layers).issubset(set(prunable_layers.get(self.prune_type).keys())):
            raise ValueError("some regular_prune_skip_layers not in valid_layers")
        if not set(prune_override_types.get(FILTER)).issubset(set(self.prunable_types.get(FILTER))):
            raise ValueError("some override_types not supported for filter prune")
        if not set(prune_override_types.get(SELECTIVE)).issubset(set(self.prunable_types.get(SELECTIVE))):
            raise ValueError("some override_types not supported for selective prune")
        if not set(regular_prune_skip_types).issubset(set(self.prunable_types.get(self.prune_type))):
            raise ValueError("some regular_prune_skip_types not supported")

    def _check_proto_retrain(self, proto, qat_layers):
        skip_layers = proto.get_quant_skip_layers()
        skip_layer_types = proto.get_quant_skip_types()

        _, retrain_override_layers, _ = proto.get_override_layers()
        _, retrain_override_types, _ = proto.get_override_layer_types()

        if not set(retrain_override_layers).issubset(set(qat_layers.keys())):
            raise ValueError("some override_layer not in valid_layers for quant")
        if not set(skip_layers).issubset(set(qat_layers.keys())):
            raise ValueError("some quant_skip_layers not in valid_layers")
        if not set(retrain_override_types).issubset(set(self.supported_types)):
            raise ValueError("some override layers types not supported for quant")
        if not set(skip_layer_types).issubset(set(self.supported_types)):
            raise ValueError("some quant_skip_types not supported")

    def _check_proto_global(self, proto, support_layers):
        global_skip_layers = proto.get_skip_layers()
        global_skip_types = proto.get_skip_layer_types()
        if not set(global_skip_layers).issubset(set(support_layers.keys())):
            raise ValueError("some skip_layers not in valid_layers")
        if not set(global_skip_types).issubset(set(self.supported_types)) and \
            not set(global_skip_types).issubset(set(self.prunable_types.get(self.prune_type))):
            raise ValueError("some skip_layer_types not supported")

    def _check_proto(self, proto, retrain_layers):
        # Step1: check whether field is excepted or not
        proto.check_field(self.enable_retrain, self.enable_prune)

        # Step2: check not support layers and not support types
        # check gloabl
        self._check_proto_global(proto, retrain_layers.get('support_layers'))
        # check retrain
        if self.enable_retrain:
            self._check_proto_retrain(proto, retrain_layers.get('qat_layers'))
        # check prune
        if self.enable_prune:
            self._check_proto_prune(proto, retrain_layers.get('prunable_layers'))

    def _check_layer_empty(self, retrain_layers, prune_layers):
        """
        Function:
        check retrain or prune layers is empty or not.
        raise ValueError when layer is empty and enable is True.
        Parameters:
        retrain_layers: dict, key: layer_name; value: layer_type
        prune_layers: dict, key: layer_name; value: layer_type
        Returns:
        None.
        """
        # prune enable is True but prune_layers is empty
        if self.enable_prune and prune_layers and \
            not prune_layers[FILTER] and not prune_layers[SELECTIVE]:
            raise ValueError("Prune supported layer cannot be empty.")
        if self.enable_retrain and not retrain_layers:
            raise ValueError("Quant supported layer cannot be empty.")


def get_layers_from_config(quant_config, global_keys):
    '''Get all layer names form quant_config '''
    layer_names = []
    for key in quant_config:
        if key not in global_keys:
            layer_names.append(key)
    return layer_names


def check_dst_type_legal(layer_data_config, layer_weight_config):
    '''check if data/weight config has the same dst_type'''
    data_dst_type = layer_data_config.get(DST_TYPE, INT8)
    weights_dst_type = layer_weight_config.get(DST_TYPE, INT8)
    act_wts_type = 'A{}W{}'.format(data_dst_type.split('INT')[-1], weights_dst_type.split('INT')[-1])

    if act_wts_type not in RETRAIN_ACT_WTS_TYPES:
        if not layer_data_config.get(DST_TYPE):
            LOGGER.logw("dst_type of RetrainDataQuantConfig was not given in config, "
                        "and was set to 'INT8' by defualt!", module_name=_MODULE_NAME)
        if not layer_weight_config.get(DST_TYPE):
            LOGGER.logw("dst_type of RetrainWeightQuantConfig was not given in config, "
                        "and was set to 'INT8' by defualt!", module_name=_MODULE_NAME)
        error_info = "Activation and weights data_type are not supported for now. Note, " \
            "activation is {} and weight is {}.".format(data_dst_type, weights_dst_type)
        LOGGER.loge(error_info, module_name=_MODULE_NAME)
        raise ValueError(error_info)


def _generate_retrain_config(proto, qat_layers, config, no_default):
    """Generate retrain config according to proto and add it to config
    Params: qat_layers, a dict
    Params: config, a dict, to contain the retrain config
    """
    data_config = proto.get_retrain_data_quant_config()
    weights_config = proto.get_retrain_weights_config()
    override_layers, _, _ = proto.get_override_layers()
    override_types, _, _ = proto.get_override_layer_types()
    skip_layers = proto.get_quant_skip_layers()
    skip_layer_types = proto.get_quant_skip_types()

    # if enable quant retrain, fill params for qat
    for layer, layer_type in qat_layers.items():
        config[layer] = OrderedDict()
        config[layer]['regular_prune_enable'] = False
        # set quant_enable
        if layer in skip_layers or layer_type in skip_layer_types:
            config[layer][RETRAIN_ENABLE] = False
        elif no_default and not proto.proto_config.HasField('retrain_data_quant_config') and \
            not proto.proto_config.HasField('retrain_weight_quant_config'):
            config[layer][RETRAIN_ENABLE] = False
        else:
            config[layer][RETRAIN_ENABLE] = True
        # set activation_quant_params and weight_quant_params
        if layer in override_layers:
            retrain_data_params, retrain_weight_params, _ = \
                proto.read_override_config(layer)
            config[layer][RETRAIN_ENABLE] = True
            config[layer][RETRAIN_DATA_CONFIG] = retrain_data_params
            config[layer][RETRAIN_WEIGHT_CONFIG] = retrain_weight_params
        elif layer_type in override_types:
            retrain_data_params, retrain_weight_params, _ = \
                proto.read_override_type_config(layer_type)
            config[layer][RETRAIN_ENABLE] = True
            config[layer][RETRAIN_DATA_CONFIG] = retrain_data_params
            config[layer][RETRAIN_WEIGHT_CONFIG] = retrain_weight_params
        else:
            config[layer][RETRAIN_DATA_CONFIG] = data_config.copy()
            config[layer][RETRAIN_WEIGHT_CONFIG] = weights_config.copy()
        check_dst_type_legal(config[layer][RETRAIN_DATA_CONFIG],
                                config[layer][RETRAIN_WEIGHT_CONFIG])
        if config[layer][RETRAIN_DATA_CONFIG].get(DST_TYPE, INT8) == INT4 \
            and layer_type == 'AvgPool':
            config[layer][RETRAIN_ENABLE] = False


def _generate_prune_config(proto, prunable_layers, no_default, config, prune_type):
    """Generate prune config according to proto and add it to config
    Params: prunable_layers, a dict
    Params: config, a dict, to contain the retrain config
    Params: no_default, bool
    Params: prune_type, FILTER/SELECTIVE/NO_PRUNE
    """
    prune_config = proto.get_prune_config()
    prune_skip_layers = proto.get_regular_prune_skip_layers()
    prune_skip_layer_types = proto.get_regular_prune_skip_types()
    override_layers, _, _ = proto.get_override_layers()
    override_types, _, _ = proto.get_override_layer_types()
    if not no_default and not prune_config:
        raise RuntimeError("the prune_config cannot be empty if do prune.")

    # if enable prune, fill params for prune
    for layer, layer_type in {**prunable_layers[FILTER], **prunable_layers[SELECTIVE]}.items():
        override_prune_params = None
        if layer in override_layers:
            _, _, override_prune_params = proto.read_override_config(layer)
        elif layer_type in override_types:
            _, _, override_prune_params = proto.read_override_type_config(layer_type)
        elif layer not in prunable_layers[prune_type]:
            continue

        # set quant_enable
        if config.get(layer) is None:
            config[layer] = OrderedDict()
            config[layer][RETRAIN_ENABLE] = False
        config[layer]['regular_prune_enable'] = True

        if layer in prune_skip_layers or layer_type in prune_skip_layer_types:
            config[layer]['regular_prune_enable'] = False

        if override_prune_params:
            config[layer].update(override_prune_params)
        else:
            config[layer].update(copy.deepcopy(prune_config))
