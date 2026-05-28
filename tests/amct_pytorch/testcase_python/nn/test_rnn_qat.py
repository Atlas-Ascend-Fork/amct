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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import copy
import os
import shutil
import unittest
from collections import defaultdict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.utils import (
    copy_tensor,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.gru import (
    GRUQAT,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.lstm import (
    LSTMQAT,
)
from amct_pytorch.classic.graph_based.amct_pytorch.utils.log import LOGGER

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class SingleGRUNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleGRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        return out


class SingleLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleLSTMNet, self).__init__()
        self.gru = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.gru(x, (h0, c0))
        return out


class TestGRUQAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LOGGER.logi(f'torch version: {torch.__version__}')
        cls.temp_folder = os.path.join(CUR_DIR, 'test_qat_rnn')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def test_gru_qat_convert_from_ori_op(self):
        net_gru = SingleGRUNet(input_size=10, hidden_size=20)
        for name, module in net_gru.named_modules():
            if isinstance(module, nn.GRU):
                qat_module = GRUQAT.from_float(
                    module)
                set_module(net_gru, name, qat_module)
        for _ in range(5):
            ret = net_gru.forward(torch.rand(1, 1, 10))
            self.assertIsNotNone(ret)

    def test_gru_qat_convert_from_ori_op_seq_n(self):
        net_gru = SingleGRUNet(input_size=10, hidden_size=20)
        for name, module in net_gru.named_modules():
            if isinstance(module, nn.GRU):
                qat_module = GRUQAT.from_float(
                    module)
                set_module(net_gru, name, qat_module)
        for _ in range(5):
            ret = net_gru.forward(torch.rand(1, 2, 10))
            self.assertIsNotNone(ret)


class TestLSTMQAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LOGGER.logi(f'torch version: {torch.__version__}')
        cls.temp_folder = os.path.join(CUR_DIR, 'test_qat_rnn')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def test_lstm_qat_convert_from_ori_op(self):
        net_lstm = SingleLSTMNet(input_size=10, hidden_size=20)
        for name, module in net_lstm.named_modules():
            if isinstance(module, nn.LSTM):
                qat_module = LSTMQAT.from_float(
                    module)
                set_module(net_lstm, name, qat_module)
        for _ in range(5):
            ret = net_lstm.forward(torch.rand(1, 1, 10))
            self.assertIsNotNone(ret)

    def test_lstm_qat_convert_from_ori_op_seq_n(self):
        net_lstm = SingleLSTMNet(input_size=10, hidden_size=20)
        for name, module in net_lstm.named_modules():
            if isinstance(module, nn.LSTM):
                qat_module = LSTMQAT.from_float(
                    module)
                set_module(net_lstm, name, qat_module)
        for _ in range(5):
            ret = net_lstm.forward(torch.rand(1, 5, 10))
            self.assertIsNotNone(ret)


def set_module(model, sub_module_name, module):
    tokens = sub_module_name.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)