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
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO

from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.common.auto_channel_prune.auto_channel_prune_config_helper import AutoChannelPruneConfigHelper
from amct_pytorch.graph_based_compression.amct_pytorch.auto_channel_prune_search import AutoChannelPruneSearch
from amct_pytorch.graph_based_compression.amct_pytorch.auto_channel_prune_search import TaylorLossSensitivity
from amct_pytorch.graph_based_compression.amct_pytorch.auto_channel_prune_search import auto_channel_prune_search
from amct_pytorch.graph_based_compression.amct_pytorch.capacity import CAPACITY
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.check import GraphQuerier


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class Net001(nn.Module):
    """ args_shape: [(1, 2, 28, 28)]
    conv + bn
    conv(with bias) + bn
    depthwise_conv + bn
    depthwise_conv(with bais) + bn
    group_conv + bn
    group_conv(bias) + bn
    fc + bn
    fc(bias) + bn
    """
    def __init__(self):
        super(Net001,self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        # depthwise_conv + bn
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm2d(16))
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        # group_conv + bn
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, groups=4),
            nn.BatchNorm2d(32))
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=3, groups=8),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        # fc
        self.fc = nn.Sequential(
            nn.Linear(8 * 16 * 16, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10, bias=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x


class TestAutoChannelPruneSearchTorch(unittest.TestCase):
    """
    The ST for TestAutoChannelPruneSearch
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_auto_channel_prune_search')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        config_defination = os.path.join(CUR_DIR, 'utils/sample.cfg')
        cls.output_cofig = os.path.join(cls.temp_folder, 'output.cfg')

        model = Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)
        model.eval()
        output = model.forward(cls.args[0])
        labels = torch.randn(output.size())
        cls.sample_data = [cls.args[0], labels]

         # parse to graph
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, cls.args, tmp_onnx)
        cls.graph = Parser.parse_net_to_graph(tmp_onnx)
        cls.graph.add_model(model)

        cls.config_helper = AutoChannelPruneConfigHelper(cls.graph, config_defination, GraphQuerier, CAPACITY)
        cls.auto_channel_prune_search = AutoChannelPruneSearch(cls.graph, cls.args, cls.config_helper, None, cls.output_cofig, None)
        print('AutoChannelPruneSearchTorch start!')


    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_search_ops(self):
        search_ops = self.auto_channel_prune_search.get_search_ops(self.graph, self.args)
        self.assertIsNotNone(search_ops)

    def test_get_graph_bitops(self):
        graph_bitops = self.auto_channel_prune_search.get_graph_bitops(self.graph, {})
        print(graph_bitops)
        self.assertIsNotNone(graph_bitops)

    def test_auto_channel_prune_search(self):
        config_defination = os.path.join(CUR_DIR, 'utils/sample.cfg')
        output_cofig = os.path.join(self.temp_folder, 'output.cfg')

        model = Net001().to(torch.device("cpu"))
        args_shape = [(1, 2, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        model.eval()
        output = model.forward(args[0])
        labels = torch.randn(output.size())
        sample_data = [args[0], labels]

        auto_channel_prune_search(model=model, config=config_defination, input_data=sample_data, output_cfg=output_cofig,
            sensitivity='TaylorLossSensitivity', search_alg='GreedySearch')
        print(output_cofig)
        self.assertTrue(output_cofig)