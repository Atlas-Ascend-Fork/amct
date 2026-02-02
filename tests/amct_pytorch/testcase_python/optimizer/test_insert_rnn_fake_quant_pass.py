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
import numpy as np
from onnx import onnx_pb
from onnx.onnx_pb import AttributeProto

from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.graph import Graph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.insert_rnn_fake_quant_pass import InsertRNNFakeQuantPass
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.quant_node import QuantOpInfo
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper


class TestInsertRNNFakeQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestInsertRNNFakeQuantPass start!')

    @classmethod
    def tearDownClass(cls):
        print('TestInsertRNNFakeQuantPass end!')

    def setUp(self):
        graph_proto = onnx_pb.GraphProto()
        # Add graph input 0
        graph_input0 = graph_proto.input.add()
        graph_input0.name = 'data0'
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        # Add conv1
        conv1 = graph_proto.node.add()
        conv1.name = 'conv1'
        conv1.op_type = 'Conv'
        conv1.input[:] = ['data0', 'conv1.weights']
        conv1.output[:] = ['conv1']
        # Add weights
        weights = graph_proto.initializer.add()
        weights.name = 'conv1.weights'
        weights.data_type = 3
        weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
        weights.dims[:] = [1, 1, 2, 3]
        # Add graph input 1
        graph_input1 = graph_proto.input.add()
        graph_input1.name = 'X'
        graph_input1.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input1.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input1.type.tensor_type.shape.dim.add().dim_value = 3
        # Add graph input 2
        graph_input2 = graph_proto.input.add()
        graph_input2.name = 'h'
        graph_input2.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input2.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input2.type.tensor_type.shape.dim.add().dim_value = 3
        # Add graph input 3
        graph_input3 = graph_proto.input.add()
        graph_input3.name = 'c'
        graph_input3.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input3.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input3.type.tensor_type.shape.dim.add().dim_value = 3
        # Add W
        W = graph_proto.initializer.add()
        W.name = 'W'
        W.data_type = 3
        W.int32_data[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        W.dims[:] = [1, 3, 3]
        # Add R
        R = graph_proto.initializer.add()
        R.name = 'R'
        R.data_type = 3
        R.int32_data[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        R.dims[:] = [1, 3, 3]
        # add quantizable lstm_0
        lstm_0 = graph_proto.node.add()
        lstm_0.name = 'lstm_0'
        lstm_0.op_type = 'LSTM'
        lstm_0.input[:] = ['X', 'W', 'R', '', '', 'h', 'c']
        lstm_0.output[:] = ['Y0', 'Y0_h', 'Y0_c']
        # add attribute
        hidden_size = lstm_0.attribute.add()
        hidden_size.name = 'hidden_size'
        hidden_size.type = AttributeProto.AttributeType.INT
        hidden_size.i = 128
        # add unquantizable lstm_1
        lstm_1 = graph_proto.node.add()
        lstm_1.name = 'lstm_1'
        lstm_1.op_type = 'LSTM'
        lstm_1.input[:] = ['X', 'W', 'R', '', '', 'h', 'c']
        lstm_1.output[:] = ['Y1', 'Y1_h', 'Y1_c']
        # add attribute
        hidden_size = lstm_1.attribute.add()
        hidden_size.name = 'hidden_size'
        hidden_size.type = AttributeProto.AttributeType.INT
        hidden_size.i = 128
        # add attribute
        input_forget = lstm_1.attribute.add()
        input_forget.name = 'input_forget'
        input_forget.type = AttributeProto.AttributeType.INT
        input_forget.i = 1

        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'
        model_proto.graph.CopyFrom(graph_proto)
        self.graph = Graph(model_proto)

        self.records = {
            'lstm_0': {
                'data_scale': 1.0,
                'data_offset': 0,
                'h_scale': 1.0,
                'h_offset': 0,
                'act_type': 'INT8',
                'weight_scale': np.ones(4),
                'weight_offset': np.zeros(4),
                'recurrence_weight_scale': np.ones(4),
                'recurrence_weight_offset': np.zeros(4)
            }
        }

    def test_match_pattern_not_lstm(self):
        conv_node = self.graph.get_node_by_name('conv1')
        self.assertFalse(InsertRNNFakeQuantPass(self.records).match_pattern(conv_node))

    def test_match_pattern_not_in_records(self):
        lstm_1 = self.graph.get_node_by_name('lstm_1')
        self.assertFalse(InsertRNNFakeQuantPass(self.records).match_pattern(lstm_1))

    def test_match_pattern_success(self):
        lstm_0 = self.graph.get_node_by_name('lstm_0')
        self.assertTrue(InsertRNNFakeQuantPass(self.records).match_pattern(lstm_0))

    def test_do_pass_success(self):
        lstm_0 = self.graph.get_node_by_name('lstm_0')
        InsertRNNFakeQuantPass(self.records).do_pass(self.graph, lstm_0)

    def test_do_pass_fail(self):
        lstm_1 = self.graph.get_node_by_name('lstm_1')
        with self.assertRaises(RuntimeError):
            InsertRNNFakeQuantPass(self.records).do_pass(self.graph, lstm_1)

    def test_generate_fake_quant_node(self):
        lstm_0 = self.graph.get_node_by_name('lstm_0')
        quant_node, antiquant_node = InsertRNNFakeQuantPass(self.records).generate_fake_quant_node(self.graph, lstm_0, 0)
        self.assertEqual(quant_node.type, 'AscendQuant')
        self.assertEqual(antiquant_node.type, 'AscendAntiQuant')

class TestInsertRNNFakeQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestInsertRNNFakeQuantPass start!')

    @classmethod
    def tearDownClass(cls):
        print('TestInsertRNNFakeQuantPass end!')

    def setUp(self):
        graph_proto = onnx_pb.GraphProto()
        # Add graph input 0
        graph_input0 = graph_proto.input.add()
        graph_input0.name = 'data0'
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        # Add conv1
        conv1 = graph_proto.node.add()
        conv1.name = 'conv1'
        conv1.op_type = 'Conv'
        conv1.input[:] = ['data0', 'conv1.weights']
        conv1.output[:] = ['conv1']
        # Add weights
        weights = graph_proto.initializer.add()
        weights.name = 'conv1.weights'
        weights.data_type = 3
        weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
        weights.dims[:] = [1, 1, 2, 3]
        # Add graph input 1
        graph_input1 = graph_proto.input.add()
        graph_input1.name = 'X'
        graph_input1.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input1.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input1.type.tensor_type.shape.dim.add().dim_value = 3
        # Add graph input 2
        graph_input2 = graph_proto.input.add()
        graph_input2.name = 'h'
        graph_input2.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input2.type.tensor_type.shape.dim.add().dim_value = 1
        graph_input2.type.tensor_type.shape.dim.add().dim_value = 3
        # Add W
        W = graph_proto.initializer.add()
        W.name = 'W'
        W.data_type = 3
        W.int32_data[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        W.dims[:] = [1, 3, 3]
        # Add R
        R = graph_proto.initializer.add()
        R.name = 'R'
        R.data_type = 3
        R.int32_data[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        R.dims[:] = [1, 3, 3]
        # add quantizable gru_0
        gru_0 = graph_proto.node.add()
        gru_0.name = 'gru_0'
        gru_0.op_type = 'GRU'
        gru_0.input[:] = ['X', 'W', 'R', '', '', 'h']
        gru_0.output[:] = ['Y0', 'Y0_h']
        linear_before_reset = gru_0.attribute.add()
        linear_before_reset.name = 'linear_before_reset'
        linear_before_reset.type = AttributeProto.AttributeType.INT
        linear_before_reset.i = 1
        # add attribute
        hidden_size = gru_0.attribute.add()
        hidden_size.name = 'hidden_size'
        hidden_size.type = AttributeProto.AttributeType.INT
        hidden_size.i = 128

        # add unquantizable gru_1
        gru_1 = graph_proto.node.add()
        gru_1.name = 'gru_1'
        gru_1.op_type = 'GRU'
        gru_1.input[:] = ['X', 'W', 'R', '', '', 'h']
        gru_1.output[:] = ['Y1', 'Y1_h']
        # add attribute
        hidden_size = gru_1.attribute.add()
        hidden_size.name = 'hidden_size'
        hidden_size.type = AttributeProto.AttributeType.INT
        hidden_size.i = 128
        # add attribute
        linear_before_reset = gru_1.attribute.add()
        linear_before_reset.name = 'linear_before_reset'
        linear_before_reset.type = AttributeProto.AttributeType.INT
        linear_before_reset.i = 0

        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'
        model_proto.graph.CopyFrom(graph_proto)
        self.graph = Graph(model_proto)

        self.records = {
            'gru_0': {
                'data_scale': 1.0,
                'data_offset': 0,
                'h_scale': 1.0,
                'h_offset': 0,
                'act_type': 'INT8',
                'weight_scale': np.ones(3),
                'weight_offset': np.zeros(3),
                'recurrence_weight_scale': np.ones(3),
                'recurrence_weight_offset': np.zeros(3)
            }
        }

    def test_match_pattern_not_in_records(self):
        gru_1 = self.graph.get_node_by_name('gru_1')
        self.assertFalse(InsertRNNFakeQuantPass(self.records).match_pattern(gru_1))

    def test_match_pattern_success(self):
        gru_0 = self.graph.get_node_by_name('gru_0')
        self.assertTrue(InsertRNNFakeQuantPass(self.records).match_pattern(gru_0))

    def test_do_pass_success(self):
        gru_0 = self.graph.get_node_by_name('gru_0')
        InsertRNNFakeQuantPass(self.records).do_pass(self.graph, gru_0)

    def test_do_pass_fail(self):
        gru_1 = self.graph.get_node_by_name('gru_1')
        with self.assertRaises(RuntimeError):
            InsertRNNFakeQuantPass(self.records).do_pass(self.graph, gru_1)

    def test_generate_fake_quant_node(self):
        gru_0 = self.graph.get_node_by_name('gru_0')
        quant_node, antiquant_node = InsertRNNFakeQuantPass(self.records).generate_fake_quant_node(self.graph, gru_0, 0)
        self.assertEqual(quant_node.type, 'AscendQuant')
        self.assertEqual(antiquant_node.type, 'AscendAntiQuant')