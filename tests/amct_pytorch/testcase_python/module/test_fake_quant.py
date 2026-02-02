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
import unittest
from unittest.mock import patch
import os
from io import BytesIO

import torch
from onnx import onnx_pb, AttributeProto

from .utils import models
from amct_pytorch.amct_pytorch_inner.amct_pytorch.module import quant_module
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.graph import Graph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.node import Node


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestQuantModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_check')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_model = BytesIO()
        Parser.export_onnx(cls.model_001, cls.args, cls.onnx_model)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.graph = Parser.parse_net_to_graph(self.onnx_model)

    def testDown(self):
        pass

    def test_float16_add_quant_module(self):
        node = self.graph.get_node_by_name("layer1.0")
        node.set_attr('op_data_type', 'float16')
        scale = 1.0
        offset = 0
        quant_bit = 8
        enter_node, out_node = quant_module.add_fakequant(self.graph, 'layer1.0.quant', scale, offset, quant_bit)
        self.assertEqual(enter_node.proto.op_type, "Cast")

    def test_float32_add_quant_module(self):
        node = self.graph.get_node_by_name("layer1.0")
        node.set_attr('op_data_type', 'float32')
        scale = 1.0
        offset = 0
        quant_bit = 8
        enter_node, out_node = quant_module.add_fakequant(self.graph, 'layer1.0.quant', scale, offset, quant_bit)
        self.assertEqual(enter_node.proto.op_type, "Mul")

    def test_construct_fakequant_cast_fp32(self):
        layer_name = "Conv_1"
        node_proto = quant_module.construct_fake_quant_dequant_cast_op(layer_name, "float32")
        self.assertEqual(node_proto.op_type, "Cast")
        self.assertEqual(node_proto.name, "Conv_1.cast")
        for attr in node_proto.attribute:
            if attr.name == "to":
                self.assertEqual(attr.i, onnx_pb.TensorProto.FLOAT)
