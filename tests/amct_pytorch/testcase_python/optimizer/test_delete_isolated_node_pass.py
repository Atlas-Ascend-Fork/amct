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
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer import DeleteIsolatedNodePass
from amct_pytorch.graph_based_compression.amct_pytorch.common.utils.util import version_higher_than

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestDeleteIsolatedNodePass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_delete_isolated_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model = models.ConvPadLong()
        cls.args_shape = [(2, 32, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'conv_pad_long.onnx')
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

        pad_node = graph.get_node_by_name('features.0_pad')
        cast_node = None
        for node in graph.nodes:
            if node.type == 'Cast':
                cast_node = node
        graph.remove_edge(cast_node, 0, pad_node, 1)

        passer = DeleteIsolatedNodePass()
        is_matched = passer.match_pattern(cast_node)
        self.assertTrue(is_matched)

        passer.do_pass(graph, cast_node)
        if version_higher_than(torch.__version__, '1.10.0'):
            self.assertEqual(15, ori_len - len(graph.nodes))
        else:
            self.assertEqual(20, ori_len - len(graph.nodes))

