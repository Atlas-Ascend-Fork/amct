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
import numpy as np
import copy
# import amct_pytorch.amct_pytorch_inner.amct_pytorch
import shutil
import onnx

from onnx import onnx_pb
from unittest.mock import patch
from onnx.external_data_helper import convert_model_to_external_data
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.graph import Graph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.save import convert_external_data_format
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestSave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.temp_folder = os.path.join(CUR_DIR, 'test_convert_external_data_format')
        os.makedirs(self.temp_folder, exist_ok=True)
        # set basic info
        self.model_proto = onnx_pb.ModelProto()
        self.model_proto.producer_name = 'model'
        self.graph = onnx_pb.GraphProto()
        # Add graph input 0
        graph_input0 = self.graph.input.add()
        graph_input0.name = 'data0'
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        # Add graph input 1
        graph_input1 = self.graph.input.add()
        graph_input1.name = 'data1'
        graph_input1.type.tensor_type.shape.dim.add().dim_value = 1
        # Add conv1
        conv1 = self.graph.node.add()
        conv1.name = 'conv1'
        conv1.op_type = 'Conv'
        conv1.input[:] = ['data0', 'conv1.weights', 'conv1.bias']
        conv1.output[:] = ['conv1']
        # add attribute "kernel_shape"
        kernel_shape = conv1.attribute.add()
        kernel_shape.name = 'kernel_shape'
        kernel_shape.type = onnx_pb.AttributeProto.AttributeType.INTS
        kernel_shape.ints[:] = [64, 3, 3, 3]
        # add attribute "pads"
        pads = conv1.attribute.add()
        pads.name = 'pads'
        pads.type = onnx_pb.AttributeProto.AttributeType.INTS
        pads.ints[:] = [0, 0, 0, 0]
        # Add weights
        weights = self.graph.initializer.add()
        weights.name = 'conv1.weights'
        weights.data_type = 1
        weights.float_data[:] = [1, 2, 3, 4, 5, 6]
        weights.dims[:] = [1, 1, 2, 3]
        # Add bias
        bias = self.graph.initializer.add()
        bias.name = 'conv1.bias'
        bias.data_type = 1
        bias.float_data[:] = [0]
        bias.dims[:] = [1]
        # Add relu
        relu1 = self.graph.node.add()
        relu1.name = 'relu1'
        relu1.op_type = 'Relu'
        relu1.input[:] = ['conv1']
        relu1.output[:] = ['relu1']
        # Add add
        add1 = self.graph.node.add()
        add1.name = 'add1'
        add1.op_type = 'Add'
        add1.input[:] = ['relu1', 'data1']
        add1.output[:] = ['add1']
        # Add average_pool
        avg_pool1 = self.graph.node.add()
        avg_pool1.name = 'avg_pool1'
        avg_pool1.op_type = 'AveragePool'
        avg_pool1.input[:] = ['add1']
        avg_pool1.output[:] = ['output']
        # add output
        graph_output = self.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        self.model_proto.graph.CopyFrom(self.graph)

    def tearDown(self):
        shutil.rmtree(self.temp_folder)
        pass

    @patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.save.MAXIMUM_PROTOBUF', 1)
    def test_convert_external_data_format(self):
        file_name = os.path.join(self.temp_folder, 'test_external_data.pth')
        test_model = copy.deepcopy(self.model_proto)
        graph = Graph(copy.deepcopy(self.model_proto), model_path=self.temp_folder)
        for initial in test_model.graph.initializer:
            data_type = initial.data_type
            np_type = TensorProtoHelper.map_np_type(data_type)
            if initial.float_data:
                data = np.array(initial.float_data).astype(np_type).tobytes()
                initial.raw_data = data
                del initial.float_data[:]
            else:
                pass
        convert_external_data_format(test_model, graph, file_name, None)
        bias_path = os.path.join(self.temp_folder, 'conv1.bias.external')
        weight_path = os.path.join(self.temp_folder, 'conv1.weights.external')
        self.assertTrue(os.path.exists(bias_path), f"File {bias_path} does not exist.")
        self.assertTrue(os.path.exists(weight_path), f"File {weight_path} does not exist.")

    @patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.save.MAXIMUM_PROTOBUF', 1)
    def test_convert_external_data_format_fakequant(self):
        file_name = os.path.join(self.temp_folder, 'test_external_data_fake_quant.onnx')
        test_model = copy.deepcopy(self.model_proto)
        graph = Graph(copy.deepcopy(self.model_proto), model_path=self.temp_folder)
        convert_external_data_format(test_model, graph, file_name, 'Fakequant')
        bias_path = os.path.join(self.temp_folder, 'conv1.bias_fakequant.external')
        weight_path = os.path.join(self.temp_folder, 'conv1.weights_fakequant.external')
        self.assertTrue(os.path.exists(bias_path), f"File {bias_path} does not exist.")
        self.assertTrue(os.path.exists(weight_path), f"File {weight_path} does not exist.")

    @patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.save.MAXIMUM_PROTOBUF', 1)
    def test_convert_external_data_format_deploy(self):
        file_name = os.path.join(self.temp_folder, 'test_external_data_deploy.onnx')
        test_model = copy.deepcopy(self.model_proto)
        graph = Graph(copy.deepcopy(self.model_proto), model_path=self.temp_folder)
        convert_external_data_format(test_model, graph, file_name, 'Deploy')
        bias_path = os.path.join(self.temp_folder, 'conv1.bias_deploy.external')
        weight_path = os.path.join(self.temp_folder, 'conv1.weights_deploy.external')
        self.assertTrue(os.path.exists(bias_path), f"File {bias_path} does not exist.")
        self.assertTrue(os.path.exists(weight_path), f"File {weight_path} does not exist.")
    
    # def test_convert_external_data_format_with_external(self):
    #     file_name = os.path.join(self.temp_folder, 'test_external_data.pth')
    #     temp_model = copy.deepcopy(self.model_proto)
    #     graph = Graph(copy.deepcopy(self.model_proto), model_path=self.temp_folder)
    #     onnx_model_name = os.path.join(self.temp_folder, 'original_external_data.onnx')
    #     for initial in temp_model.graph.initializer:
    #         data_type = initial.data_type
    #         np_type = TensorProtoHelper.map_np_type(data_type)
    #         if initial.float_data:
    #             data = np.array(initial.float_data).astype(np_type).tobytes()
    #             initial.raw_data = data
    #             del initial.float_data[:]
    #         else:
    #             pass
    #     if onnx.__version__ in ['1.9.0', '1.14.0']:
    #         convert_model_to_external_data(temp_model, location="data.bin", size_threshold=0)
    #     else:
    #         convert_model_to_external_data(temp_model, location="data.bin")
    #     onnx.save_model(temp_model, onnx_model_name)
    #     test_model = Parser.parse_proto(onnx_model_name)
    #     convert_external_data_format(test_model, graph, file_name, None)
    #     bias_path = os.path.join(self.temp_folder, 'conv1.bias.external')
    #     weight_path = os.path.join(self.temp_folder, 'conv1.weights.external')
    #     self.assertTrue(os.path.exists(bias_path), f"File {bias_path} does not exist.")
    #     self.assertTrue(os.path.exists(weight_path), f"File {weight_path} does not exist.")

    # def test_convert_external_data_format_data_location_error(self):
    #     file_name = os.path.join(self.temp_folder, 'test_external_data.pth')
    #     temp_model = copy.deepcopy(self.model_proto)
    #     graph = Graph(copy.deepcopy(self.model_proto), model_path=self.temp_folder)
    #     onnx_model_name = os.path.join(self.temp_folder, 'original_external_data.onnx')
    #     for initial in temp_model.graph.initializer:
    #         data_type = initial.data_type
    #         np_type = TensorProtoHelper.map_np_type(data_type)
    #         if initial.float_data:
    #             data = np.array(initial.float_data).astype(np_type).tobytes()
    #             initial.raw_data = data
    #             del initial.float_data[:]
    #         else:
    #             pass
    #     if onnx.__version__ in ['1.9.0', '1.14.0']:
    #         convert_model_to_external_data(temp_model, location="data.bin", size_threshold=0)
    #     else:
    #         convert_model_to_external_data(temp_model, location="data.bin")
    #     onnx.save_model(temp_model, onnx_model_name)
    #     test_model = Parser.parse_proto(onnx_model_name)
    #     del test_model.graph.initializer[0].external_data[0]
    #     self.assertRaises(ValueError, convert_external_data_format, test_model, graph, file_name, None)

    # def test_convert_external_data_format_external_data_error(self):
    #     file_name = os.path.join(self.temp_folder, 'test_external_data.onnx')
    #     temp_model = copy.deepcopy(self.model_proto)
    #     graph = Graph(copy.deepcopy(self.model_proto), model_path=self.temp_folder)
    #     onnx_model_name = os.path.join(self.temp_folder, 'original_external_data.onnx')
    #     for initial in temp_model.graph.initializer:
    #         data_type = initial.data_type
    #         np_type = TensorProtoHelper.map_np_type(data_type)
    #         if initial.float_data:
    #             data = np.array(initial.float_data).astype(np_type).tobytes()
    #             initial.raw_data = data
    #             del initial.float_data[:]
    #         else:
    #             pass
    #     if onnx.__version__ in ['1.9.0', '1.14.0']:
    #         convert_model_to_external_data(temp_model, location="data.bin", size_threshold=0)
    #     else:
    #         convert_model_to_external_data(temp_model, location="data.bin")
    #     onnx.save_model(temp_model, onnx_model_name)
    #     test_model = Parser.parse_proto(onnx_model_name)
    #     del test_model.graph.initializer[0].external_data[2]
    #     self.assertRaises(ValueError, convert_external_data_format, test_model, graph, file_name, None)