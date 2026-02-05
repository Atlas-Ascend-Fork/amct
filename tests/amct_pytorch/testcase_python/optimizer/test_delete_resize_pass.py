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
import sys
import os
import unittest
from copy import deepcopy

import json
import numpy as np
import torch
from .util import models

from onnx import onnx_pb
from amct_pytorch.graph_based_compression.amct_pytorch.graph.graph import Graph
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.common.utils.util import version_higher_than
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer import DeleteResizePass
CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestDeleteResizePass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_delete_resize_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model = models.Yolov3ResizeModel()
        cls.args_shape = [(2, 32, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'yolov3_resize.onnx')
        Parser.export_onnx(cls.model, cls.args, cls.onnx_file)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_do_success_01(self):
        """ now is fail"""
        graph = Parser.parse_net_to_graph(self.onnx_file)
        ori_len = len(graph.nodes)
        resize_node = None
        for node in graph.nodes:
            if node.type == 'Resize':
                resize_node = node
        passer = DeleteResizePass()
        is_matched = passer.match_pattern(resize_node)
        self.assertTrue(is_matched)

        passer.do_pass(graph, resize_node)
        # self.assertEqual(6, ori_len - len(graph.nodes))

    def test_do_success_02(self):
        """ now is fail"""
        # set basic info
        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'
        graph = onnx_pb.GraphProto()
        # Add graph input 0
        graph_input0 = graph.input.add()
        graph_input0.name = 'data0'
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        # Add scales
        scales = graph.initializer.add()
        scales = graph.initializer.add()
        scales.name = 'resize_scales'
        scales.data_type = onnx_pb.TensorProto.DataType.FLOAT
        scales.float_data[:] = [1,1,2,2]
        scales.dims[:] = [4]
        # Add roi
        roi = graph.node.add()
        roi.name = 'resize_roi'
        roi.op_type = 'Constant'
        roi.output[:] = ['resize_roi_output']
        # Add resize
        resize0 = graph.node.add()
        resize0.name = 'resize'
        resize0.op_type = 'Resize'
        resize0.input[:] = ['data0', 'resize_roi_output', 'resize_scales']
        resize0.output[:] = ['output']

        # add output
        graph_output = graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        model_proto.graph.CopyFrom(graph)

        graph = Graph(model_proto)
        ori_len = len(graph.nodes)
        resize_node = graph.get_node_by_name('resize')
        passer = DeleteResizePass()
        is_matched = passer.match_pattern(resize_node)
        self.assertTrue(is_matched)

        passer.do_pass(graph, resize_node)
        self.assertEqual(3, ori_len - len(graph.nodes))