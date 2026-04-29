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
from amct_pytorch.graph_based_compression.amct_pytorch.prune.pruner_helper import PruneHelper

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestFilterPruneHelper(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_prune_helper')
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

    def test_train_branch(self,):
        """ test active, passive, disable"""
        model = models.NetTrainBranch().to(torch.device("cpu"))
        args_shape = [(1, 2, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        ori_output = model.forward(args[0])

        onnx_file = os.path.join(self.temp_folder, 'NetTrainBranch.onnx')
        Parser.export_onnx(model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        Parser.export_onnx(model, args, onnx_file)
        graph.model = model

        record_file = os.path.join(CUR_DIR, 'utils/records/record_NetTrainBranch.txt')
        PruneHelper(graph, args, record_file).restore_prune_model()
        new_model = graph.model
        new_output = new_model.forward(args[0])

        self.assertEqual(ori_output.shape, new_output.shape)

        self.assertEqual(new_model.layer1[0].out_channels, 16)