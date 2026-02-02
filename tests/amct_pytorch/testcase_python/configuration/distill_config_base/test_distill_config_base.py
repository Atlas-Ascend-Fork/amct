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
import shutil
import sys
import numpy as np
import unittest
from unittest import mock
from unittest.mock import patch
from collections import OrderedDict

import torch
from onnx import onnx_pb

from amct_pytorch.amct_pytorch_inner.amct_pytorch.capacity import CAPACITY
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config_base.distill_config_base import DistillConfigBase
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config_base.distill_config_base import GraphObjects
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config_base.distill_proto import DistillProtoConfig
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.check import GraphQuerier
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.graph import Graph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.node import Node

from .utils import models

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestDistillConfigBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestDistillConfigBase start!')
        cls.temp_folder = os.path.join(CUR_DIR, 'test_distill_config_base')
        os.makedirs(cls.temp_folder, exist_ok=True)

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_001.onnx')
        Parser.export_onnx(cls.model_001, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)
        cls.graph.model = cls.model_001

        cls.distill_config_base = DistillConfigBase(
            GraphObjects(graph_querier=GraphQuerier, graph_checker=None), CAPACITY)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder)
        print('TestDistillConfigBase end!')

    def test_check_dst_type_legal_default_data(self):
        data_config = {
            'algo' : 'ulq_quantize'
        }
        weight_config = {
            'algo' : 'arq_distill',
            'channel_wise' : True,
            'dst_type' : 'INT4'
        }
        with self.assertRaises(ValueError):
            self.distill_config_base.check_dst_type_legal(data_config, weight_config)

    def test_check_dst_type_legal_default_weight(self):
        data_config = {
            'algo' : 'ulq_quantize',
            'dst_type' : 'INT4'
        }
        weight_config = {
            'algo' : 'arq_distill',
            'channel_wise' : True
        }
        with self.assertRaises(ValueError):
            self.distill_config_base.check_dst_type_legal(data_config, weight_config)

    def test_get_cascade_unit_not_in_unit(self):
        cascade_unit = self.distill_config_base.get_cascade_unit(self.graph, [], 1, self.graph.nodes[0], [])
        self.assertEqual(cascade_unit, [])

    def test_get_cascade_unit_equal_group_size(self):
        node_proto1 = onnx_pb.NodeProto()
        node_proto1.name = 'layer1'
        node_proto1.op_type = 'type1'

        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'

        graph = Graph(model_proto)
        node1 = graph.add_node(node_proto1)
        node1.set_module_name(node_proto1.name)

        distill_unit = [[node_proto1.name]]
        cascade_unit = self.distill_config_base.get_cascade_unit(
            graph, distill_unit, 1, node1, [])
        self.assertEqual(cascade_unit, distill_unit)

    def test_get_cascade_unit_two_groups(self):
        node_proto1 = onnx_pb.NodeProto()
        node_proto1.name = 'layer1'
        node_proto1.op_type = 'type1'

        node_proto2 = onnx_pb.NodeProto()
        node_proto2.name = 'layer2'
        node_proto2.op_type = 'type2'

        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'

        graph = Graph(model_proto)
        node1 = graph.add_node(node_proto1)
        node1.set_module_name(node_proto1.name)
        node2 = graph.add_node(node_proto2)
        node2.set_module_name(node_proto2.name)

        distill_unit = [[node_proto1.name], [node_proto2.name]]
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.common.graph_base.node_base.NodeBase.get_consumers') as mock_get_consumers:
            mock_get_consumers.return_value = [[node2], []]
            cascade_unit = self.distill_config_base.get_cascade_unit(
                graph, distill_unit, 2, node1, [])
            self.assertEqual(cascade_unit, [sum(distill_unit, [])])

    def test_get_distill_cascade_unit(self):
        node_proto1 = onnx_pb.NodeProto()
        node_proto1.name = 'layer1'
        node_proto1.op_type = 'type1'

        node_proto2 = onnx_pb.NodeProto()
        node_proto2.name = 'layer2'
        node_proto2.op_type = 'type2'

        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'

        graph = Graph(model_proto)
        node1 = graph.add_node(node_proto1)
        node1.set_module_name(node_proto1.name)
        node2 = graph.add_node(node_proto2)
        node2.set_module_name(node_proto2.name)

        distill_unit = [[node_proto1.name, node_proto2.name]]
        cascade_units = self.distill_config_base.get_distill_cascade_unit(graph, distill_unit, 1)
        self.assertEqual(cascade_units, distill_unit)

    def test_check_groups_intersection(self):
        node_proto1 = onnx_pb.NodeProto()
        node_proto1.name = 'layer1'
        node_proto1.op_type = 'type1'

        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'

        graph = Graph(model_proto)
        node1 = graph.add_node(node_proto1)
        node1.set_module_name(node_proto1.name)

        distill_groups = [[node_proto1.name], [node_proto1.name]]
        with self.assertRaises(ValueError):
            self.distill_config_base.check_groups_intersection(graph, distill_groups)

    def test_sort_distill_group_empty(self):
        with self.assertRaises(ValueError):
            self.distill_config_base.sort_distill_group(self.graph, [], [])

    def test_sort_distill_group_success(self):
        node_proto1 = onnx_pb.NodeProto()
        node_proto1.name = 'layer1'
        node_proto1.op_type = 'type1'

        node_proto2 = onnx_pb.NodeProto()
        node_proto2.name = 'layer2'
        node_proto2.op_type = 'type2'

        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'

        graph = Graph(model_proto)
        node1 = graph.add_node(node_proto1)
        node1.set_module_name(node_proto1.name)
        node2 = graph.add_node(node_proto2)
        node2.set_module_name(node_proto2.name)

        distill_groups = [[node_proto2.name]]
        cascade_unit = [[node_proto1.name]]
        sort_groups = self.distill_config_base.sort_distill_group(graph, distill_groups, cascade_unit)
        self.assertEqual(sort_groups, distill_groups + cascade_unit)

    def test_create_default_config_success(self):
        config_file = os.path.join(self.temp_folder, 'default_config.json')
        self.distill_config_base.create_default_config(config_file, self.graph)

        # check json
        config = self.distill_config_base.parse_distill_config(config_file, self.graph.model)
        self.assertEqual(config.get('batch_num'), 1)
        self.assertEqual(config.get('group_size'), 1)
        self.assertEqual(config.get('data_dump'), False)
        distill_groups = [['conv1', 'bn1'], ['conv2', 'relu1'], ['conv3', 'bn2', 'relu2']]
        self.assertEqual(config.get('distill_group'), distill_groups)
        layer_config = {
            "quant_enable":True,
            "distill_data_config":{
                "algo":"ulq_quantize",
                "dst_type":"INT8"
            },
            "distill_weight_config":{
                "algo":"arq_distill",
                "channel_wise":True,
                "dst_type":"INT8"
            }
        }
        self.assertEqual(config.get('conv1'), layer_config)
        self.assertEqual(config.get('conv2'), layer_config)
        self.assertEqual(config.get('conv3'), layer_config)

    def test_create_config_from_proto_success(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        config_proto_file = os.path.join(CUR_DIR, './utils/distill.cfg')
        self.distill_config_base.create_config_from_proto(config_file, self.graph, config_proto_file)

        # check json
        config = self.distill_config_base.parse_distill_config(config_file, self.graph.model)
        self.assertEqual(config.get('batch_num'), 2)
        self.assertEqual(config.get('group_size'), 2)
        self.assertEqual(config.get('data_dump'), True)
        distill_groups = [['conv1', 'bn1'], ['conv2', 'relu1'], ['conv3', 'bn2', 'relu2']]
        self.assertEqual(config.get('distill_group'), distill_groups)
        conv1_config = {
            "quant_enable":False,
            "distill_data_config":{
                "algo":"ulq_quantize",
                "dst_type":"INT8"
            },
            "distill_weight_config":{
                "algo":"arq_distill",
                "channel_wise":True,
                "dst_type":"INT8"
            }
        }
        self.assertEqual(config.get('conv1'), conv1_config)
        conv2_config = {
            "quant_enable":True,
            "distill_data_config":{
                "algo":"ulq_quantize",
                "clip_max":6.0,
                "clip_min":-6.0,
                "fixed_min":True,
                "dst_type":"INT8"
            },
            "distill_weight_config":{
                "algo":"arq_distill",
                "channel_wise":False,
                "dst_type":"INT8"
            }
        }
        self.assertEqual(config.get('conv2'), conv2_config)
        conv3_config = {
            "quant_enable":True,
            "distill_data_config":{
                "algo":"ulq_quantize",
                "clip_max":3.0,
                "clip_min":-3.0,
                "dst_type":"INT4"
            },
            "distill_weight_config":{
                "algo":"arq_distill",
                "channel_wise":False,
                "dst_type":"INT4"
            }
        }
        self.assertEqual(config.get('conv3'), conv3_config)

    def test_parse_distill_config_no_distill_group(self):
        config_file = os.path.join(CUR_DIR, './utils/no_group_config.json')
        with self.assertRaises(ValueError):
            self.distill_config_base.parse_distill_config(config_file, self.graph.model)

    def test_get_supported_layers_empty(self):
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.check.GraphQuerier.get_support_distill_layer2type', return_value=[]):
            with self.assertRaises(ValueError):
                self.distill_config_base._get_supported_layers(self.graph.model)

    def test_get_supported_layers_success(self):
        layer2type = self.distill_config_base._get_supported_layers(self.graph.model)
        self.assertEqual(layer2type, {'conv1':'Conv2d', 'conv2':'Conv2d', 'conv3':'Conv2d'})

    def test_check_proto_skip_layer(self):
        supported_layer2type = {'layer':'type'}
        skip_layers = ['layer1']
        config_proto_file = os.path.join(CUR_DIR, './utils/distill.cfg')
        proto = DistillProtoConfig(config_proto_file, CAPACITY)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config_base.distill_proto.DistillProtoConfig.get_quant_skip_layers',
            return_value=skip_layers):
            with self.assertRaises(ValueError):
                self.distill_config_base.check_proto(proto, supported_layer2type)

    def test_check_proto_skip_type(self):
        supported_layer2type = {'conv1':'Conv2d'}
        skip_types = ['type1']
        config_proto_file = os.path.join(CUR_DIR, './utils/distill.cfg')
        proto = DistillProtoConfig(config_proto_file, CAPACITY)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config_base.distill_proto.DistillProtoConfig.get_quant_skip_layer_types',
            return_value=skip_types):
            with self.assertRaises(ValueError):
                self.distill_config_base.check_proto(proto, supported_layer2type)

    def test_check_proto_override_layer(self):
        supported_layer2type = {'conv1':'Conv2d'}
        override_layers = ['layer1']
        config_proto_file = os.path.join(CUR_DIR, './utils/distill.cfg')
        proto = DistillProtoConfig(config_proto_file, CAPACITY)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config_base.distill_proto.DistillProtoConfig.get_override_layers',
            return_value=override_layers):
            with self.assertRaises(ValueError):
                self.distill_config_base.check_proto(proto, supported_layer2type)

    def test_check_proto_override_type(self):
        supported_layer2type = {'conv1':'Conv2d', 'conv3':'Conv2d'}
        override_types = ['type1']
        config_proto_file = os.path.join(CUR_DIR, './utils/distill.cfg')
        proto = DistillProtoConfig(config_proto_file, CAPACITY)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config_base.distill_proto.DistillProtoConfig.get_override_layer_types',
            return_value=override_types):
            with self.assertRaises(ValueError):
                self.distill_config_base.check_proto(proto, supported_layer2type)

    def test_get_distill_unit(self):
        distill_groups = [['conv1', 'bn1']]
        supported_distill_layers = {'conv1':'Conv2d', 'conv2':'Conv2d', 'conv3':'Conv2d'}
        distill_unit = self.distill_config_base._get_distill_unit(self.graph, distill_groups, supported_distill_layers)
        self.assertEqual(distill_unit, [['conv2', 'relu1'], ['conv3', 'bn2', 'relu2']])

    def test_check_distill_group_type_empty(self):
        distill_groups = []
        self.distill_config_base._check_distill_group_type(distill_groups, self.graph.model)

    def test_check_distill_group_type_not_module(self):
        distill_groups = [['not_module']]
        with self.assertRaises(ValueError):
            self.distill_config_base._check_distill_group_type(distill_groups, self.graph.model)

    def test_check_distill_group_type_not_supported(self):
        distill_groups = [['conv_transpose']]
        with self.assertRaises(ValueError):
            self.distill_config_base._check_distill_group_type(distill_groups, self.graph.model)

    def test_dfs_search_bn_reused(self):
        node_proto = onnx_pb.NodeProto()
        node_proto.name = 'bn'
        node_proto.op_type = 'BatchNormalization'
        node = Node(0, node_proto)
        node.set_attr('is_reuse', True)
        with self.assertRaises(ValueError):
            self.distill_config_base._dfs_search(node, '', [])

    def test_get_all_nodes_between_two_layers_start_layer_reused(self):
        node_proto = onnx_pb.NodeProto()
        node_proto.name = 'layer'
        node_proto.op_type = 'type'

        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'

        graph = Graph(model_proto)
        node = graph.add_node(node_proto)
        node.set_module_name(node_proto.name)
        node.set_attr('is_reuse', True)
        layers = {'start_layer':node_proto.name}
        with self.assertRaises(ValueError):
            self.distill_config_base._get_all_nodes_between_two_layers(layers, graph)

    def test_get_all_nodes_between_two_layers_no_start_layer(self):
        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'

        graph = Graph(model_proto)
        layers = {'start_layer':'layer'}
        with self.assertRaises(ValueError):
            self.distill_config_base._get_all_nodes_between_two_layers(layers, graph)
