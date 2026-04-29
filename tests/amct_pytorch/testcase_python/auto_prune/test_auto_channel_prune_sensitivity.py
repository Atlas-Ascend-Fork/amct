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
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pb2
from amct_pytorch.graph_based_compression.amct_pytorch.common.utils.prune_record_attr_util import AttrProtoHelper

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


class TestTaylorLossSensitivity(unittest.TestCase):
    """
    The ST for TestAutoChannelPruneSearch
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_auto_channel_prune_search')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.config_defination = os.path.join(CUR_DIR, 'utils/sample.cfg')
        cls.output_cofig = os.path.join(cls.temp_folder, 'output.cfg')
        model = Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)
         # parse to graph
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, cls.args, tmp_onnx)
        cls.graph = Parser.parse_net_to_graph(tmp_onnx)
        cls.graph.add_model(model)

        cls.graph.model.eval()
        output = cls.graph.model.forward(cls.args[0])
        labels = torch.randn(output.size())
        cls.sample_data = [cls.args[0], labels]

        cls.config_helper = AutoChannelPruneConfigHelper(cls.graph, cls.config_defination, GraphQuerier, CAPACITY)
        cls.auto_channel_prune_search = AutoChannelPruneSearch(cls.graph, cls.args, cls.config_helper, None, cls.output_cofig, None)
        cls.graph_info = cls.auto_channel_prune_search.graph_info
        cls.test_iteration = 1

        print('TestTaylorLossSensitivity start!')


    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        print('TestTaylorLossSensitivity end!')
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_backward_grad(self):
        sentitvity_func = TaylorLossSensitivity()
        sentitvity_func.setup_initialization(graph_tuple=(self.graph, self.graph_info),
            input_data=self.sample_data, test_iteration=self.test_iteration)
        self.assertTrue(sentitvity_func.weights)
        self.assertTrue(sentitvity_func.grads)


    def test_compute_taylor_by_channel(self):
        sentitvity_func = TaylorLossSensitivity()
        sentitvity_func.setup_initialization(graph_tuple=(self.graph, self.graph_info),
            input_data=self.sample_data, test_iteration=self.test_iteration)

        print(self.graph_info)
        ch_info = {'begin': 0,  'end': self.graph_info.get('layer1.0').get('cout')}
        ch_score = sentitvity_func.compute_taylor_by_channel('layer1.0', ch_info)
        assert not torch.any(torch.isnan(ch_score))


    def test_sensitivity_param_exception(self):
        sentitvity_func = TaylorLossSensitivity()
        try:
            sentitvity_func.setup_initialization(graph_tuple=(self.graph, self.graph_info),
                input_data=self.sample_data, test_iteration=100)
        except Exception as e:
            print('[Exception]test_sensitivity_param_error:', e)
            self.assertTrue(True)


    def test_get_sensitivity(self):
        sentitvity_func = TaylorLossSensitivity()
        sentitvity_func.setup_initialization(graph_tuple=(self.graph, self.graph_info),
            input_data=self.sample_data, test_iteration=self.test_iteration)
        record = self.auto_channel_prune_search.search_records
        sentitvity_func.get_sensitivity(record)
        attr_helper = AttrProtoHelper(record[0].producer[0])
        sensitivity = attr_helper.get_attr_value('sensitivity')
        self.assertTrue(sensitivity)
        print(sensitivity)
