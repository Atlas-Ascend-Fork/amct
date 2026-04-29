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
from unittest import mock
import torch


from .utils import models
from .utils import record_utils

DEVICE = 'cpu'
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.prune.filter_prune_helper import create_filter_prune_helper
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pytorch_pb2
from amct_pytorch.graph_based_compression.amct_pytorch.common.prune.prune_recorder_helper import PruneRecordHelper
from amct_pytorch.graph_based_compression.amct_pytorch.configuration import retrain_config
from amct_pytorch.graph_based_compression.amct_pytorch.utils.model_util import ModuleHelper

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestFilterPruneHelper(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_filter_prune_helper')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cascade(self,):
        """ test active, passive, disable"""
        model = models.SingleConv()
        args_shape = [(1, 16, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        onnx_file = os.path.join(self.temp_folder, 'SingleConv.onnx')
        Parser.export_onnx(model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        preprocess_graph(graph, model)

        record = scale_offset_record_pytorch_pb2.ScaleOffsetRecord()
        record_helper = PruneRecordHelper(record, graph)

        configer = FakeRetrainConfig()
        configer.set_prune_layers(['layer1.0', 'layer2.0'])
        with mock.patch.object(retrain_config, 'RetrainConfig', FakeRetrainConfig):
                for node in graph.nodes:
                    helper = create_filter_prune_helper(node)
                    helper.process(record_helper)

        producer_names, consumer_names = record_utils.get_producer(record.prune_record[0])
        self.assertEqual(producer_names, ['layer1.0'])
        self.assertEqual(consumer_names, ['layer1.1', 'layer2.0'])

    def test_eltwise(self,):
        """ test active, passive, disable"""
        model = models.EltwiseConv()
        args_shape = [(1, 16, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        onnx_file = os.path.join(self.temp_folder, 'EltwiseConv.onnx')
        Parser.export_onnx(model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        preprocess_graph(graph, model)

        records = scale_offset_record_pytorch_pb2.ScaleOffsetRecord()
        record_helper = PruneRecordHelper(records, graph)

        configer = FakeRetrainConfig()
        configer.set_prune_layers(['layer1.0', 'layer2.0', 'layer3.0', 'layer4.0', 'layer5.0'])
        with mock.patch.object(retrain_config, 'RetrainConfig', FakeRetrainConfig):
            for node in graph.nodes + graph._in_out_nodes:
                helper = create_filter_prune_helper(node)
                helper.process(record_helper)

        tar_records = record_utils.read_record_file(os.path.join(CUR_DIR, './utils/records/record_EltwiseConv.txt'))

        prune_records = records.prune_record[0]
        tar_prune_records = tar_records.prune_record[0]
        for idx, producer in enumerate(prune_records.producer):
            self.assertEqual(producer, tar_prune_records.producer[idx])
        # node name is different based on torch version
        # change name in record as name in file
        for idx, consumer in enumerate(prune_records.consumer):
            if 'add' in consumer.name.lower():
                consumer.name = tar_prune_records.consumer[idx].name
            self.assertEqual(consumer, tar_prune_records.consumer[idx])

    def test_splitconcat(self,):
        """ test active, passive, disable"""
        model = models.SplitConcatConv()
        args_shape = [(1, 16, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        onnx_file = os.path.join(self.temp_folder, 'SplitConcatConv.onnx')
        Parser.export_onnx(model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        preprocess_graph(graph, model)

        records = scale_offset_record_pytorch_pb2.ScaleOffsetRecord()
        record_helper = PruneRecordHelper(records, graph)

        configer = FakeRetrainConfig()

        configer.set_prune_layers(['layer1.0', 'layer2_1.0', 'layer2_2.0', 'layer3.0'])
        with mock.patch.object(retrain_config, 'RetrainConfig', FakeRetrainConfig):
            for node in graph.nodes + graph._in_out_nodes:
                helper = create_filter_prune_helper(node)
                helper.process(record_helper)
        tar_records = record_utils.read_record_file(os.path.join(CUR_DIR, './utils/records/record_SplitConcatConv.txt'))

        prune_records = records.prune_record[0]
        tar_prune_records = tar_records.prune_record[0]
        for idx, producer in enumerate(prune_records.producer):
            self.assertEqual(producer, tar_prune_records.producer[idx])
        # node name is different based on torch version
        # change name in record as name in file
        for idx, consumer in enumerate(prune_records.consumer):
            if 'concat' in consumer.name.lower():
                consumer.name = tar_prune_records.consumer[idx].name
            self.assertEqual(consumer, tar_prune_records.consumer[idx])

    def test_splitconcat_gconv(self,):
        """ test active, passive, disable"""
        model = models.SplitConcatGroupConv()
        args_shape = [(1, 16, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        onnx_file = os.path.join(self.temp_folder, 'SplitConcatGroupConv.onnx')
        Parser.export_onnx(model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        preprocess_graph(graph, model)

        record = scale_offset_record_pytorch_pb2.ScaleOffsetRecord()
        record_helper = PruneRecordHelper(record, graph)

        configer = FakeRetrainConfig()

        configer.set_prune_layers(['layer1.0', 'layer2_1.0', 'layer2_2.0', 'layer3.0'])
        with mock.patch.object(retrain_config, 'RetrainConfig', FakeRetrainConfig):
            for node in graph.nodes + graph._in_out_nodes:
                helper = create_filter_prune_helper(node)
                helper.process(record_helper)
        self.assertTrue(not record.ListFields())


class FakeRetrainConfig:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        super(FakeRetrainConfig, self).__init__()

    def set_prune_layers(self, prune_layers):
        self.prune_layers = prune_layers

    def prune_enable(self, name):
        if name in self.prune_layers:
            return True
        else:
            return False

    def filter_prune_enable(self, name):
        if name in self.prune_layers:
            return True
        else:
            return False


def preprocess_graph(graph, model):
    """
    Function: preprocess graph to add torch_type for node if it can be found in torch
    Param: None
    Returns: None
    """
    model_helper = ModuleHelper(model)
    for node in graph.nodes:
        try:
            mod = model_helper.get_module(node.name)
            node.set_attr('torch_type', type(mod).__name__)
        except RuntimeError:
            pass