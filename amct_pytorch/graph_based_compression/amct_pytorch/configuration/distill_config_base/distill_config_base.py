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

from ....amct_pytorch.common.utils.files import save_to_json
from ....amct_pytorch.common.utils.util import check_no_repeated
from ....amct_pytorch.common.utils.util import find_repeated_items
from ....amct_pytorch.common.utils.vars_util import INT8
from ....amct_pytorch.configuration.distill_config_base.distill_field import LayerConfig
from ....amct_pytorch.configuration.distill_config_base.distill_field import DistillRootConfig
from ....amct_pytorch.configuration.distill_config_base.distill_proto import DistillProtoConfig
from ....amct_pytorch.utils.log import LOGGER
from ....amct_pytorch.utils.model_util import ModuleHelper
from ....amct_pytorch.utils.vars import DST_TYPE

DISTILL_DATA_CONFIG = 'distill_data_config'
DISTILL_WEIGHT_CONFIG = 'distill_weight_config'

GraphObjects = namedtuple('GraphObjects', ['graph_querier', 'graph_checker'])


class DistillConfigBase():
    '''distill config base'''
    def __init__(self, graph_objects, capacity):
        ''' inner method '''
        self.capacity = capacity
        self.graph_querier = graph_objects.graph_querier
        self.graph_checker = graph_objects.graph_checker
        self.config_tree = DistillRootConfig(self.graph_querier, self.capacity)
        self.supported_distill_types = self.capacity.get_value('DISTILL_TYPES')
        self.supported_bn_types = self.capacity.get_value('DISTILL_BN_TYPES')
        self.supported_bn_onnx_types = self.capacity.get_value('DISTILL_BN_ONNX_TYPES')
        self.supported_activation_types = self.capacity.get_value('DISTILL_ACTIVATION_ONNX_TYPES')
        self.supported_module_type = self.capacity.get_value('DISTILL_MODULE_TYPES')

    @staticmethod
    def check_dst_type_legal(layer_data_config, layer_weight_config):
        '''check if data/weight config has the same dst_type'''
        data_dst_type = layer_data_config.get(DST_TYPE, INT8)
        weights_dst_type = layer_weight_config.get(DST_TYPE, INT8)

        if data_dst_type != weights_dst_type:
            if not layer_data_config.get(DST_TYPE):
                LOGGER.logw("dst_type of DistillDataQuantConfig was not given in config, "
                            "and was set to 'INT8' by defualt!", module_name='DistillConfig')
            if not layer_weight_config.get(DST_TYPE):
                LOGGER.logw("dst_type of RetrainWeightQuantConfig was not given in config, "
                            "and was set to 'INT8' by defualt!", module_name='DistillConfig')
            raise ValueError("Activation and weights with different data_type are not supported for now. Note, " \
                "activation is {} and weight is {}.".format(data_dst_type, weights_dst_type))

    @staticmethod
    def get_enable_quant_layers(config):
        '''
        Function: get enable quant layers
        Parameter: config: dict, distill config
        Return: list, enable quant layers
        '''
        enable_quant_layers = list()
        for layer_name, layer_config in config.items():
            if not isinstance(layer_config, dict):
                continue
            if layer_config.get('quant_enable', False):
                enable_quant_layers.append(layer_name)
        return enable_quant_layers

    @staticmethod
    def get_min_unit(node, distill_units):
        '''
        Function: find the distillation unit that the node belongs to
        Parameter: node: Node
                    distill_unit: distillation unit
        Return: distillation unit the node belongs to
                None, if the node cannot be distilled
        '''
        for unit in distill_units:
            if node.name in unit:
                return unit
        return None

    @staticmethod
    def is_node_in_units(node_name, cascade_units):
        '''
        Function: find if the node is in cascade units
        Parameter: node_name: node name
                cascade_units: cascade units
        Return: True/False
        '''
        for find_unit in cascade_units:
            if node_name in find_unit:
                return True
        return False

    @staticmethod
    def get_cascade_unit(graph, distill_unit, group_size, node, cascade_units):
        '''
        Function: find the neighboring casecade unit
        Parameter: graph: Graph
                distill_unit: min distillation unit
                group_size: max number of min distillation unit
                node: Node
        Return: cascade_units: cascade units, store node name
        '''
        unit_num = 0
        cascade = []
        while node is not None:
            # if node is not distill, stop searching
            unit = DistillConfigBase.get_min_unit(node, distill_unit)
            if unit is None:
                break

            # update casecade unit
            cascade.extend(unit)
            unit_num = unit_num + 1

            # group is full, save the last group
            if unit_num == group_size:
                break

            # get consumer for cascading
            end_layer_name = cascade[-1]
            end_node = graph.get_node_by_name(end_layer_name)
            consumers, _ = end_node.get_consumers(0)
            if len(consumers) == 1:
                node = consumers[0]
            else:
                break

        if len(cascade) != 0:
            cascade_units.append(cascade)
        return cascade_units

    @staticmethod
    def get_distill_cascade_unit(graph, distill_unit, group_size):
        '''
        Function: find the distillation casecade unit
        Parameter: graph: Graph
                    distill_unit: min distillation unit
        Return: casecade distillation units, store module name
        '''
        cascade_units = []
        for node in graph.nodes:
            if DistillConfigBase.is_node_in_units(node.name, cascade_units):
                continue
            cascade_units = DistillConfigBase.get_cascade_unit(graph, distill_unit, group_size, node, cascade_units)

        # get module name by AMCT IR node name
        cascade_module_units = []
        for cascade_unit in cascade_units:
            cascade_module_unit = []
            for node_name in cascade_unit:
                node = graph.get_node_by_name(node_name)
                cascade_module_unit.append(node.module_name)
            cascade_module_units.append(cascade_module_unit)

        return cascade_module_units

    @staticmethod
    def check_groups_intersection(graph, module_name_distill_groups):
        ''' check whether there is an intersection between groups '''
        # get AMCT IR node name
        node_name_distill_groups = list()
        for module_name_distill_group in module_name_distill_groups:
            node_name_distill_group = list()
            for module_name in module_name_distill_group:
                nodes = graph.get_node_by_module_name(module_name)
                # relu module is reused, no need to check whether intersect
                if len(nodes) > 1:
                    continue
                node_name_distill_group.append(nodes[0].name)
            node_name_distill_groups.append(node_name_distill_group)

        group_num = len(node_name_distill_groups)
        for i in range(group_num - 1):
            for j in range(i + 1, group_num):
                if set(node_name_distill_groups[i]) & set(node_name_distill_groups[j]):
                    raise ValueError("There is an intersection between "
                        "distill_group(start_layer_name: {}, end_layer_name: {}) and "
                        "distill_group(start_layer_name: {}, end_layer_name: {})".format(
                        node_name_distill_groups[i][0], node_name_distill_groups[i][-1],
                        node_name_distill_groups[j][0], node_name_distill_groups[j][-1]))

    @staticmethod
    def sort_distill_group(graph, distill_groups, cascade_unit):
        ''' inner method for sort distill group according to the topology order in graph '''
        all_distill_groups = distill_groups + cascade_unit
        if all_distill_groups == []:
            raise ValueError("Not found distill group in graph, please check the model or cfg file.")
        sorted_distill_groups = list()
        for node in graph.nodes:
            record_distill_layers = sum(sorted_distill_groups, [])
            if node.module_name in record_distill_layers:
                continue
            for distill_group in all_distill_groups:
                if node.module_name == distill_group[0]:
                    sorted_distill_groups.append(distill_group)
                    break
        return sorted_distill_groups

    @staticmethod
    def is_node_reused(node):
        '''check if module is reused'''
        if node.has_attr('is_reuse') and node.get_attr('is_reuse'):
            return True
        return False

    @staticmethod
    def check_proto(proto, supported_layer2type):
        ''' inner method for check not supported layers and types '''
        def _get_not_supported_names(be_checked_names, supported_names):
            not_supported_names = list()
            for name in be_checked_names:
                if name not in supported_names:
                    not_supported_names.append(name)
            return not_supported_names

        skip_layers = proto.get_quant_skip_layers()
        if not set(skip_layers).issubset(set(supported_layer2type.keys())):
            not_supported_layers = _get_not_supported_names(set(skip_layers), set(supported_layer2type.keys()))
            raise ValueError("quant_skip_layers{} not exist or supported".format(not_supported_layers))

        skip_layer_types = proto.get_quant_skip_layer_types()
        if not set(skip_layer_types).issubset(set(supported_layer2type.values())):
            not_supported_types = _get_not_supported_names(set(skip_layer_types), set(supported_layer2type.values()))
            raise ValueError("quant_skip_layer_types{} not exist or supported".format(not_supported_types))

        override_layers = proto.get_override_layers()
        if not set(override_layers).issubset(set(supported_layer2type.keys())):
            not_supported_layers = _get_not_supported_names(set(override_layers), set(supported_layer2type.keys()))
            raise ValueError("distill_override_layers{} not exist or supported for quant".format(
                not_supported_layers))

        override_layer_types = proto.get_override_layer_types()
        if not set(override_layer_types).issubset(set(supported_layer2type.values())):
            not_supported_types = _get_not_supported_names(
                set(override_layer_types), set(supported_layer2type.values()))
            raise ValueError("distill_override_layer_types{} not exist or supported for quant".format(
                not_supported_types))

    def create_default_config(self, config_file, graph):
        '''
        Function: create default distill config by graph
        Parameter: config_file: the path of json file to save distill config
                   graph: IR Graph
        Return: None
        '''
        supported_layer2type = self._get_supported_layers(graph.model, graph)
        # get distill group from graph
        distill_unit = self._get_distill_unit(graph, [], supported_layer2type.keys())
        # get cascade unit
        cascade_unit = self.get_distill_cascade_unit(graph, distill_unit, 1)
        # sort distill group according to the topology order in graph
        all_distill_groups = self.sort_distill_group(graph, [], cascade_unit)
        self._clear_config_tree()
        self.config_tree.build_default(all_distill_groups, supported_layer2type)
        ordered_config = self.config_tree.dump()

        save_to_json(config_file, ordered_config)
        LOGGER.logi("Create distill config file {} success!".format(config_file), module_name='DistillConfig')

    def create_config_from_proto(self, config_file, graph, config_proto_file):
        '''
        Function: create distill config by graph and simple config file
        Parameter: config_file: the path of json file to save distill config
                   graph: IR Graph
                   config_proto_file: the path of user's simple config file
        Return: None
        '''
        proto = DistillProtoConfig(config_proto_file, self.capacity)
        supported_layer2type = self._get_supported_layers(graph.model, graph)
        config = self._generate_layer_config(proto, supported_layer2type)

        global_config = proto.get_proto_global_config()
        for item in global_config.keys():
            if global_config[item] is not None:
                config[item] = global_config[item]

        # get distill group from simple config file
        distill_groups = proto.get_distill_groups()
        # get all layers in distill group
        detailed_distill_groups = self._get_layers_in_distill_group(distill_groups, graph)
        # check distill group from simple config file
        self._check_distill_group_type(detailed_distill_groups, graph.model)
        self.check_groups_intersection(graph, detailed_distill_groups)
        # get distill group from graph
        distill_unit = self._get_distill_unit(graph, detailed_distill_groups, supported_layer2type.keys())
        # get cascade unit
        cascade_unit = self.get_distill_cascade_unit(graph, distill_unit, config['group_size'])
        # sort distill group according to the topology order in graph
        all_distill_groups = self.sort_distill_group(graph, detailed_distill_groups, cascade_unit)
        config['distill_group'] = all_distill_groups

        self._clear_config_tree()
        # no need to check version field
        self.config_tree.set_strong_check(False)
        self.config_tree.build(config, supported_layer2type)
        ordered_config = self.config_tree.dump()

        save_to_json(config_file, ordered_config)
        LOGGER.logi("Create distill config file {} success!".format(config_file), module_name='DistillConfig')

    def parse_distill_config(self, config_file, model, graph=None):
        '''
        Function: parse distill config
            if graph is None, will not check op specification
        Parameter: config_file: the path of config json file
                   model: nn.module
                   graph: None, IR Graph
        Return: dict, distill config
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
                distill_config = json.load(fid, object_pairs_hook=_detect_repetitive_key_hook)
            except json.decoder.JSONDecodeError as e:
                raise ValueError("config_file {} is invalid, please check.".format(config_file)) from e

        supported_layers = self._get_supported_layers(model, graph)

        self._clear_config_tree()
        self.config_tree.build(distill_config, supported_layers.copy())
        ordered_config = self.config_tree.dump()
        distill_groups = ordered_config.get('distill_group')
        if distill_groups is None:
            raise ValueError("config file {} do not have distill group.".format(config_file))
        self._check_distill_group_type(distill_groups, model)

        LOGGER.logd('distill config is {}'.format(ordered_config), module_name='DistillConfig')
        return ordered_config

    def _get_supported_layers(self, model, graph=None):
        ''' inner method for distill quant '''
        supported_layer2type = self.graph_querier.get_support_distill_layer2type(model, graph)
        if not supported_layer2type:
            raise ValueError("Not found supported distill quant layers in model.")
        return supported_layer2type

    def _clear_config_tree(self):
        ''' inner method '''
        self.config_tree = DistillRootConfig(self.graph_querier, self.capacity)

    def _generate_layer_config(self, proto, supported_layer2type):
        """
        Function: Generate layer config according to proto
        Params:
            proto: DistillProtoConfig
            supported_layer2type: a dict, supported distill layer to type map
        Return: config, a dict
        """
        self.check_proto(proto, supported_layer2type)

        skip_layers = proto.get_quant_skip_layers()
        skip_layer_types = proto.get_quant_skip_layer_types()
        override_layers = proto.get_override_layers()
        override_layer_types = proto.get_override_layer_types()
        data_config = proto.get_distill_data_quant_config()
        weight_config = proto.get_distill_weight_quant_config()

        config = OrderedDict()
        for layer, layer_type in supported_layer2type.items():
            config[layer] = OrderedDict()
            config[layer]['quant_enable'] = True
            if layer in skip_layers or layer_type in skip_layer_types:
                config[layer]['quant_enable'] = False
                config[layer][DISTILL_DATA_CONFIG] = dict()
                config[layer][DISTILL_WEIGHT_CONFIG] = dict()
            elif layer in override_layers:
                distill_data_params, distill_weight_params = proto.read_override_layer_config(layer)
                config[layer][DISTILL_DATA_CONFIG] = distill_data_params
                config[layer][DISTILL_WEIGHT_CONFIG] = distill_weight_params
            elif layer_type in override_layer_types:
                distill_data_params, distill_weight_params = proto.read_override_type_config(layer_type)
                config[layer][DISTILL_DATA_CONFIG] = distill_data_params
                config[layer][DISTILL_WEIGHT_CONFIG] = distill_weight_params
            else:
                config[layer][DISTILL_DATA_CONFIG] = data_config.copy()
                config[layer][DISTILL_WEIGHT_CONFIG] = weight_config.copy()
            self.check_dst_type_legal(config[layer][DISTILL_DATA_CONFIG],
                config[layer][DISTILL_WEIGHT_CONFIG])

        return config

    def _get_distill_unit(self, graph, distill_groups, supported_distill_layers):
        '''
        Function: find the distillation unit
        Parameter: graph: Graph
                distill_groups: user define distillation groups
                supported_distill_layers: nodes that support distillation
        Return: list, distillation unit
        '''
        def _get_single_consumer(node):
            '''
            Function: get node consumer if the node has only one consumer
            Parameter: node: Node
            Return: None: if node has no consumer or has multi-consumer
                    consumer node: if node has only one consumer
            '''
            if len(node.output_anchors) == 0:
                return None

            consumers, _ = node.get_consumers(0)
            if len(consumers) != 1:
                return None
            return consumers[0]
        model_helper = ModuleHelper(graph.model)
        distill_unit = []
        user_distill_layers = sum(distill_groups, [])
        for node in graph.nodes:
            # skip user define units
            if node.module_name in user_distill_layers:
                continue
            if node.module_name not in supported_distill_layers:
                continue
            unit = [node.name]
            # the last node or multi-output node, save unit
            second_node = _get_single_consumer(node)
            if second_node is None:
                distill_unit.append(unit)
                continue

            # the second node is bn, update unit
            if second_node.type in self.supported_bn_onnx_types:
                bn_mod = model_helper.get_module(second_node.module_name)
                # only support bn2d
                if type(bn_mod).__name__ not in self.supported_bn_types:
                    continue
                if self.is_node_reused(second_node):
                    continue
                unit.append(second_node.name)

                # the second node is bn, check the third node
                # the third node is the last, save unit
                third_node = _get_single_consumer(second_node)
                if third_node is None:
                    distill_unit.append(unit)
                    continue

                # the third node is activation, update and save unit
                if third_node.type in self.supported_activation_types:
                    unit.append(third_node.name)

            # the second node is activation, update and save unit
            if second_node.type in self.supported_activation_types:
                unit.append(second_node.name)

            distill_unit.append(unit)

        return distill_unit

    def _check_distill_group_type(self, detailed_distill_groups, model):
        ''' inner method for check users's distill group module type '''
        # if user do not set distill group, no need to check
        if detailed_distill_groups == []:
            return

        model_helper = ModuleHelper(model)
        for detailed_distill_group in detailed_distill_groups:
            for layer_name in detailed_distill_group:
                try:
                    # check layer is nn.Module or not
                    mod = model_helper.get_module(layer_name)
                except RuntimeError as exception:
                    raise ValueError("layer [{}] in distill_group(start_layer_name: {}, end_layer_name: {}) "
                        "not exist or not a nn.Module object.".format(
                        layer_name, detailed_distill_group[0], detailed_distill_group[-1])) from exception

                # check module type whether supported
                mod_type = type(mod).__name__
                if mod_type not in self.supported_module_type:
                    raise ValueError("layer [{}] in distill_group(start_layer_name: {}, end_layer_name: {}) "
                        "is not a supported type{}.".format(layer_name, detailed_distill_group[0],
                        detailed_distill_group[-1], self.supported_module_type))

    def _dfs_search(self, node, end_layer, path):
        '''search all paths from start layer to end layer'''
        if node.type in self.supported_bn_onnx_types and self.is_node_reused(node):
            raise ValueError("Not support to distill reused module for {}".format(node.module_name))

        # save the path from start node to curent node
        cur_path = path + [node]
        if node.module_name == end_layer:
            return [cur_path]

        all_paths = list()
        for out_port in range(len(node.output_anchors)):
            consumers, _ = node.get_consumers(out_port)
            for consumer in consumers:
                if consumer not in cur_path:
                    newpaths = self._dfs_search(consumer, end_layer, cur_path)
                    all_paths.extend(newpaths)

        return all_paths

    def _get_all_nodes_between_two_layers(self, layers, graph):
        start_layer = layers.get('start_layer')
        end_layer = layers.get('end_layer')
        all_paths = list()
        for node in graph.nodes:
            if node.module_name == start_layer:
                if self.is_node_reused(node):
                    raise ValueError("Not support module {} be reused "
                        "as a start layer of distill_group".format(node.module_name))
                all_paths = self._dfs_search(node, end_layer, [])
                break
        if len(all_paths) != 1:
            raise ValueError("Can not find an only route from start_layer_name({}) to end_layer_name({}), "
                "please check whether layer in model and in right order.".format(start_layer, end_layer))
        # get module name from node
        all_nodes = list()
        for node in all_paths[0]:
            all_nodes.append(node.module_name)
        return all_nodes

    def _get_layers_in_distill_group(self, distill_groups, graph):
        ''' inner method to get users's distill group '''
        detailed_distill_groups = list()
        for distill_group in distill_groups:
            distill_group_nodes = self._get_all_nodes_between_two_layers(distill_group, graph)
            detailed_distill_groups.append(distill_group_nodes)
        return detailed_distill_groups
