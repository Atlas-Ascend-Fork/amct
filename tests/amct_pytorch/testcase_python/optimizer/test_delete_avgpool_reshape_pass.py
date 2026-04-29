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
import torch

from .util import models
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer import ReplaceAvgpoolReshapePass

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestReplaceAvgpoolReshapePass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_delete_avgpool_reshape_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model = models.MobilenetTailModel()
        cls.args_shape = [(2, 32, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'mobilenet_tail.onnx')
        Parser.export_onnx(cls.model, cls.args, cls.onnx_file)


    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_do_success(self):
        graph = Parser.parse_net_to_graph(self.onnx_file)
        ori_len = len(graph.nodes)
        avgpool_node = None
        for node in graph.nodes:
            if node.type == 'GlobalAveragePool':
                avgpool_node = node
                break
        passer = ReplaceAvgpoolReshapePass()
        is_matched = passer.match_pattern(avgpool_node)
        if '1.10' not in torch.__version__:
            self.assertTrue(is_matched)
            passer.do_pass(graph, avgpool_node)
            self.assertEqual(5, ori_len - len(graph.nodes))

