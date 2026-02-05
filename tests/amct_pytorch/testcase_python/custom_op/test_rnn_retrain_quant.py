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
import numpy as np
from torch.nn.utils.rnn import pack_sequence
from unittest.mock import patch

from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.recorder.recorder import Recorder
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.ifmr.ifmr import IFMR
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.comp_module.comp_module_rnn import CompModuleRNN
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.rnn_retrain_quant import RNNRetrainQuant

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestRNNRetrainQuant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn_retrain_quant')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.module = torch.nn.LSTM(10, 20, 1, batch_first=True)
        cls.input = torch.randn(1, 1, 10)
        cls.h0 = torch.randn(1, 1, 20)
        cls.c0 = torch.randn(1, 1, 20)

        cls.act_config = {
            'num_bits': 8,
            'clip_max': 1.0,
            'clip_min': -1.0
        }
        cls.wts_config = {
            'num_bits': 8,
            'channel_wise': False,
            'algo': 'arq_retrain'
        }
        cls.comp_common_config = {
            'device': 'cpu',
            'need_sync': False,
            'process_group': None,
            'world_size': 1,
            'layers_name': ['lstm'],
            'batch_num': 1
        }
        cls.comp_args = {
            'module': cls.module,
            'act_config': cls.act_config,
            'wts_config': cls.wts_config,
            'common_config': cls.comp_common_config,
            'acts_comp_reuse': False
        }
        cls.quant_module = CompModuleRNN(**cls.comp_args)
        cls.quant_module.comp_algs.append('quant')

        # recorder
        cls.record_file = os.path.join(cls.temp_folder, 'record.txt')
        if not os.path.exists(cls.record_file):
            with open(cls.record_file, 'w') as f:
                f.write('')
        cls.record_module = Recorder(cls.record_file)

        cls.common_config = {
            'data_num_bits': 8,
            'wts_num_bits': 8,
            'layers_name': ['lstm'],
            'batch_num': 1
        }
        cls.retrain_quant = RNNRetrainQuant(cls.quant_module, cls.record_module)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def test_forward_input_type_error(self):
        input_data = pack_sequence([self.input])
        with self.assertRaises(ValueError):
            self.retrain_quant.forward(input_data)

    def test_forward_sequence_length_error(self):
        input = torch.randn(2, 1, 10)
        with self.assertRaises(ValueError):
            self.retrain_quant.forward(input)

    def test_forward_hx_none(self):
        with self.assertRaises(ValueError):
            self.retrain_quant.forward(self.input)

    def test_forward_success(self):
        self.retrain_quant.forward(self.input, (self.h0, self.c0))

    def test_reorganize_rnn_quant_factor(self):
        quant_factor = np.array([0, 1, 2, 3])
        reorganized_quant_factor = self.retrain_quant._reorganize_rnn_quant_factor(quant_factor, 'name', 'LSTM')
        self.assertEqual(reorganized_quant_factor, [0, 3, 1, 2])

        quant_factor = np.array([0, 1, 2])
        reorganized_quant_factor = self.retrain_quant._reorganize_rnn_quant_factor(quant_factor, 'name', 'GRU')
        self.assertEqual(reorganized_quant_factor, [1, 0, 2])

    def test_update_quant_factor(self):
        with patch('amct_pytorch.graph_based_compression.amct_pytorch.custom_op.utils.process_scale'):
            self.retrain_quant._update_quant_factor()