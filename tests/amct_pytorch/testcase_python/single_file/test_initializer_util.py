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
import unittest
from unittest.mock import patch

from onnx import onnx_pb
import numpy as np

from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.graph import Graph

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestInitializerUtil(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_initializer_util')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_proto = onnx_pb.ModelProto()
        cls.model_proto.producer_name = 'model'
        graph = onnx_pb.GraphProto()
        graph_input0 = graph.input.add()
        graph_input0.name = 'data0'
        graph_output = graph.output.add()
        graph_output.name = 'output'

        cls.model_proto.graph.CopyFrom(graph)


    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_map_data_location(self):
        # TensorProtoHelper.map_data_location(-1)
        self.assertRaises(ValueError, TensorProtoHelper.map_data_location, -1)

    def test_map_np_type(self):
        # TensorProtoHelper.map_np_type(-1)
        self.assertRaises(ValueError, TensorProtoHelper.map_np_type, -1)

    def test_set_data(self):
        tensor_proto = onnx_pb.TensorProto()
        tensor_proto.name = 'initializer1'
        tensor_helper = TensorProtoHelper(tensor_proto)

        data = np.array([1, 3])
        tensor_helper.set_data(data, 'INT8')
        self.assertEqual(tensor_helper.tensor.data_type, tensor_proto.DataType.INT8)

    def test_set_data_python_backend(self):
        tensor_proto = onnx_pb.TensorProto()
        tensor_proto.name = 'initializer1'
        tensor_helper = TensorProtoHelper(tensor_proto)
        data = np.array([1, 3])
        with patch('google.protobuf.internal.api_implementation.Type', return_value='python'):
            tensor_helper.set_data(data, 'FLOAT')
            self.assertEqual(tensor_helper.tensor.data_type, tensor_proto.DataType.FLOAT)

    def test_cast_ori_data_fp16(self):
        tensor_proto = onnx_pb.TensorProto()
        tensor_proto.name = 'initializer1'
        tensor_helper = TensorProtoHelper(tensor_proto)

        data = np.array([1, 3])
        np_value = tensor_helper.cast_ori_data(data, 'float16')
        self.assertEqual(np_value.dtype, np.float16)
        
    def test_get_external_data(self):
        tensor_proto = onnx_pb.TensorProto()
        tensor_proto.name = 'initializer1'
        tensor_proto.data_type = onnx_pb.TensorProto().DataType.FLOAT16
        tensor_proto.dims.extend([20])
        tensor_proto.data_location = 1
        ex_data = tensor_proto.external_data.add()
        ex_data.key = "location"
        ex_data.value = "data.bin"
        ex_data = tensor_proto.external_data.add()
        ex_data.key = "offset"
        ex_data.value = str(0)
        ex_data = tensor_proto.external_data.add()
        ex_data.key = "length"
        ex_data.value = str(40)
 
        path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path, "data.bin")
        data = np.array([0.1]*20, np.float16)
        data.tofile(data_path)
 
        tensor_helper = TensorProtoHelper(tensor_proto, path)
        externel_data = tensor_helper.get_data()
       
        os.remove(data_path)
        self.assertEqual((data == externel_data).all(), True)
 
    def test_get_external_data_failed(self):
        tensor_proto = onnx_pb.TensorProto()
        tensor_proto.name = 'initializer1'
        tensor_proto.data_type = onnx_pb.TensorProto().DataType.FLOAT16
        tensor_proto.dims.extend([20])
        tensor_proto.data_location = 1
        ex_data = tensor_proto.external_data.add()
        ex_data.key = "offset"
        ex_data.value = str(0)
        ex_data = tensor_proto.external_data.add()
        ex_data.key = "length"
        ex_data.value = str(20)
 
        tensor_helper = TensorProtoHelper(tensor_proto)
       
        with self.assertRaises(ValueError):
            tensor_helper.get_data()
 
        ex_data = tensor_proto.external_data.add()
        ex_data.key = "location"
        ex_data.value = "data.bin"
 
        path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(path, "data.bin")
        data = np.array([0.1]*20, np.float16)
        data.tofile(data_path)
 
        tensor_helper = TensorProtoHelper(tensor_proto, path)
        with self.assertRaises(ValueError):
            tensor_helper.get_data()
        os.remove(data_path)
    
    def test_set_external_data(self):
        tensor_proto = onnx_pb.TensorProto()
        tensor_proto.name = 'initializer1'
        tensor_proto.data_type = onnx_pb.TensorProto().DataType.FLOAT16
        tensor_proto.dims.extend([20])
        tensor_proto.data_location = 1
        ex_data = tensor_proto.external_data.add()
        ex_data.key = "location"
        ex_data.value = "data.bin"
        ex_data = tensor_proto.external_data.add()
        ex_data.key = "offset"
        ex_data.value = str(0)
        ex_data = tensor_proto.external_data.add()
        ex_data.key = "length"
        ex_data.value = str(40)
 
        data = np.array([0.1]*20, np.float16)
        tensor_helper = TensorProtoHelper(tensor_proto)
        tensor_helper.set_external_data(data)
 
        tensor_np_type = tensor_helper.map_np_type(tensor_helper.tensor.data_type)
        np_value = np.frombuffer(tensor_helper.tensor.raw_data, getattr(np, tensor_np_type))
        np_value = np.array(np_value)
        self.assertEqual((data == np_value).all(), True)