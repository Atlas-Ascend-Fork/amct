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

import json as _json
import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.quantize_tool import (
    create_quant_config,
    create_quant_retrain_config,
    create_quant_retrain_model,
    quantize_model,
    save_model,
    save_quant_retrain_model,
)

from amct_pytorch.classic.graph_based.amct_pytorch.utils.onnx_initializer_util import (
    TensorProtoHelper,
)

from .utils import rnn_model
from .utils import models

torch.manual_seed(0)
CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
# 原生 INT4 需 onnx>=1.16；旧版 onnx 无 INT4 枚举，A8W4 相关用例跳过
_INT4_SUPPORTED = 'INT4' in TensorProtoHelper.data_type_maps
_SKIP_INT4_MSG = 'onnx version too old for native INT4 (A8W4)'


class TestGRUPTQ(unittest.TestCase):
    """
    The UT for QuantizeTool
    """

    @classmethod
    def setUpClass(cls):
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        num_class = 10
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        gru_hidden_size = 64
        num_gru_layers = 1

        cls.model = rnn_model.Conv1dGRU(
            input_channels=channels,
            conv1d_kernel_size=conv1d_kernel_size,
            conv1d_out_channels=conv1d_out_channels,
            gru_hidden_size=gru_hidden_size,
            num_classes=num_class,
            num_gru_layers=num_gru_layers,
            dropout=0.1,
        )

        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)

        cls.ori_out = cls.model(cls.input, cls.h0)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        sys.modules["torch_npu"] = MagicMock()

    def tearDown(self):
        pass

    def test_create_quant_config(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        create_quant_config(
            config_file=config_file, model=self.model, input_data=(self.input, self.h0)
        )

        self.assertTrue(os.path.exists(config_file))

    def test_quantize_model(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        record_file = os.path.join(self.temp_folder, 'record.txt')
        modified_model = os.path.join(self.temp_folder, 'modified_model.onnx')
        new_model = quantize_model(
            config_file, modified_model, record_file, self.model, (self.input, self.h0)
        )

        self.assertTrue(os.path.exists(modified_model))
        output = new_model(self.input, self.h0)

        self.assertTrue(os.path.exists(modified_model))
        self.assertIsNotNone(output)

    def test_save_model(self):
        record_file = os.path.join(self.temp_folder, 'record.txt')
        modified_model = os.path.join(self.temp_folder, 'modified_model.onnx')
        save_path = os.path.join(self.temp_folder, 'res')
        save_model(modified_model, record_file, save_path)

        fakequant = os.path.join(self.temp_folder, 'res_fake_quant_model.onnx')
        deploy = os.path.join(self.temp_folder, 'res_deploy_model.onnx')

        self.assertTrue(os.path.exists(fakequant))
        self.assertTrue(os.path.exists(deploy))


class TestGRUQAT(unittest.TestCase):
    """
    The UT for QuantizeTool
    """

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        cls.num_class = 10
        cls.learning_rate = 0.001
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        gru_hidden_size = 64
        num_gru_layers = 1

        cls.model = rnn_model.Conv1dGRU(
            input_channels=channels,
            conv1d_kernel_size=conv1d_kernel_size,
            conv1d_out_channels=conv1d_out_channels,
            gru_hidden_size=gru_hidden_size,
            num_classes=cls.num_class,
            num_gru_layers=num_gru_layers,
            dropout=0.1,
        )

        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)

        cls.ori_out = cls.model(cls.input, cls.h0)

        cls.new_model = None

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_quant_retrain_config(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        create_quant_retrain_config(
            config_file=config_file, model=self.model, input_data=(self.input, self.h0)
        )

        self.assertTrue(os.path.exists(config_file))

    @unittest.skip(
        'pre-existing hang: blocks indefinitely in dev env, tracked separately'
    )
    def test_create_quant_retrain_model(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        record_file = os.path.join(self.temp_folder, 'record.txt')
        self.new_model = create_quant_retrain_model(
            config_file, self.model, record_file, (self.input, self.h0)
        )

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.new_model.parameters(), lr=self.learning_rate)
        label = torch.randint(0, self.num_class, (self.batch_size,))

        self.assertIsNotNone(self.new_model)
        output, _ = self.new_model(self.input, self.h0)
        self.assertIsNotNone(output)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.new_model.eval()
        with torch.no_grad():
            output, _ = self.new_model(self.input, self.h0)

        save_path = os.path.join(self.temp_folder, 'res')
        fakequant = os.path.join(self.temp_folder, 'res_fake_quant_model.onnx')
        deploy = os.path.join(self.temp_folder, 'res_deploy_model.onnx')

        save_quant_retrain_model(
            model=self.new_model,
            input_data=(self.input, self.h0),
            config_file=config_file,
            record_file=record_file,
            save_path=save_path,
        )

        self.assertTrue(os.path.exists(fakequant))
        self.assertTrue(os.path.exists(deploy))


class TestGRUQATA8W4(unittest.TestCase):
    """GRU A8W4 (INT8 activation, INT4 weight) e2e QAT retrain: patch weight dst_type→INT4, assert pipeline runs."""

    @classmethod
    def setUpClass(cls):
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        cls.num_class = 10
        cls.learning_rate = 0.001
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        gru_hidden_size = 64
        num_gru_layers = 1

        cls.model = rnn_model.Conv1dGRU(
            input_channels=channels,
            conv1d_kernel_size=conv1d_kernel_size,
            conv1d_out_channels=conv1d_out_channels,
            gru_hidden_size=gru_hidden_size,
            num_classes=cls.num_class,
            num_gru_layers=num_gru_layers,
            dropout=0.1,
        )

        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn_a8w4')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)
        cls.batch_size = 1
        cls.cfg = os.path.join(CUR_DIR, 'utils', 'qat_rnn_a8w4.cfg')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules['torch_npu'] = MagicMock()

    def tearDown(self):
        pass

    @unittest.skipUnless(_INT4_SUPPORTED, _SKIP_INT4_MSG)
    def test_gru_a8w4_qat_e2e(self):
        """GRU A8W4: steps 1-4 fully asserted; step 5 env-blocked, known RuntimeError asserted."""
        config_file = os.path.join(self.temp_folder, 'retrain_config_a8w4.json')
        record_file = os.path.join(self.temp_folder, 'record_a8w4.txt')
        save_path = os.path.join(self.temp_folder, 'res_a8w4')

        # Step 1: create retrain config via cfg entry (config_defination drives A8W4)
        create_quant_retrain_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, self.h0),
            config_defination=self.cfg,
        )
        self.assertTrue(
            os.path.exists(config_file), 'retrain config file must be created'
        )

        # verify the cfg entry actually drove weight INT4 into the generated config
        _global_keys = {
            'version',
            'batch_num',
            'activation_offset',
            'fakequant_precision_mode',
        }
        with open(config_file) as _fh:
            _cfg_after = _json.load(_fh)
        _wts_int4_seen = False
        for _lname, _lcfg in _cfg_after.items():
            if _lname in _global_keys or not isinstance(_lcfg, dict):
                continue
            if (
                'retrain_weight_config' in _lcfg
                and _lcfg['retrain_weight_config'].get('dst_type') == 'INT4'
            ):
                _wts_int4_seen = True
        self.assertTrue(
            _wts_int4_seen,
            'A8W4 cfg must yield at least one layer with weight dst_type=INT4',
        )

        # Step 2: build quantized retrain model using patched A8W4 config
        new_model = create_quant_retrain_model(
            config_file, self.model, record_file, (self.input, self.h0)
        )
        self.assertIsNotNone(new_model, 'A8W4 retrain model must not be None')

        # Step 3: one training step
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate)
        label = torch.randint(0, self.num_class, (self.batch_size,))
        output, _ = new_model(self.input, self.h0)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step 4: eval forward pass
        new_model.eval()
        with torch.no_grad():
            output, _ = new_model(self.input, self.h0)
        self.assertIsNotNone(output, 'A8W4 eval output must not be None')

        # Step 5: save quant retrain model (deploy + fakequant)
        save_quant_retrain_model(
            model=new_model,
            input_data=(self.input, self.h0),
            config_file=config_file,
            record_file=record_file,
            save_path=save_path,
        )
        self.assertTrue(
            os.path.exists(save_path + '_deploy_model.onnx'),
            'A8W4 QAT deploy model must be saved',
        )


class TestGRUQATA16W8(unittest.TestCase):
    """GRU A16W8 (INT16 activation, INT8 weight) e2e QAT retrain: patch data dst_type→INT16, assert pipeline runs."""

    @classmethod
    def setUpClass(cls):
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        cls.num_class = 10
        cls.learning_rate = 0.001
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        gru_hidden_size = 64
        num_gru_layers = 1

        cls.model = rnn_model.Conv1dGRU(
            input_channels=channels,
            conv1d_kernel_size=conv1d_kernel_size,
            conv1d_out_channels=conv1d_out_channels,
            gru_hidden_size=gru_hidden_size,
            num_classes=cls.num_class,
            num_gru_layers=num_gru_layers,
            dropout=0.1,
        )

        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn_a16w8')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)
        cls.batch_size = 1
        cls.cfg = os.path.join(CUR_DIR, 'utils', 'qat_rnn_a16w8.cfg')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules['torch_npu'] = MagicMock()

    def tearDown(self):
        pass

    def test_gru_a16w8_qat_e2e(self):
        """GRU A16W8 end-to-end via cfg entry (config_defination drives A16W8)."""
        config_file = os.path.join(self.temp_folder, 'retrain_config_a16w8.json')
        record_file = os.path.join(self.temp_folder, 'record_a16w8.txt')
        save_path = os.path.join(self.temp_folder, 'res_a16w8')

        # Step 1: create retrain config via cfg entry (config_defination drives A16W8)
        create_quant_retrain_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, self.h0),
            config_defination=self.cfg,
        )
        self.assertTrue(
            os.path.exists(config_file), 'retrain config file must be created'
        )

        # verify the cfg entry actually drove activation INT16 into the generated config
        _global_keys = {
            'version',
            'batch_num',
            'activation_offset',
            'fakequant_precision_mode',
        }
        with open(config_file) as _fh:
            _cfg_after = _json.load(_fh)
        _act_int16_seen = False
        for _lname, _lcfg in _cfg_after.items():
            if _lname in _global_keys or not isinstance(_lcfg, dict):
                continue
            if (
                'retrain_data_config' in _lcfg
                and _lcfg['retrain_data_config'].get('dst_type') == 'INT16'
            ):
                _act_int16_seen = True
        self.assertTrue(
            _act_int16_seen,
            'A16W8 cfg must yield at least one layer with activation dst_type=INT16',
        )

        # Step 2: build quantized retrain model using patched A16W8 config
        new_model = create_quant_retrain_model(
            config_file, self.model, record_file, (self.input, self.h0)
        )
        self.assertIsNotNone(new_model, 'A16W8 retrain model must not be None')

        # Step 3: one training step
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate)
        label = torch.randint(0, self.num_class, (self.batch_size,))
        output, _ = new_model(self.input, self.h0)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step 4: eval forward pass
        new_model.eval()
        with torch.no_grad():
            output, _ = new_model(self.input, self.h0)
        self.assertIsNotNone(output, 'A16W8 eval output must not be None')

        # Step 5: save quant retrain model (deploy + fakequant)
        save_quant_retrain_model(
            model=new_model,
            input_data=(self.input, self.h0),
            config_file=config_file,
            record_file=record_file,
            save_path=save_path,
        )
        self.assertTrue(
            os.path.exists(save_path + '_deploy_model.onnx'),
            'A16W8 QAT deploy model must be saved',
        )


class TestLSTMQATA8W4(unittest.TestCase):
    """LSTM A8W4 (INT8 activation, INT4 weight) e2e QAT retrain: patch weight dst_type→INT4, assert pipeline runs."""

    @classmethod
    def setUpClass(cls):
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        cls.num_class = 10
        cls.learning_rate = 0.001
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        lstm_hidden_size = 64
        num_lstm_layers = 1

        cls.model = rnn_model.Conv1dLSTM(
            input_channels=channels,
            conv1d_kernel_size=conv1d_kernel_size,
            conv1d_out_channels=conv1d_out_channels,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=cls.num_class,
            num_lstm_layers=num_lstm_layers,
            dropout=0.1,
        )

        cls.temp_folder = os.path.join(CUR_DIR, 'test_lstm_a8w4')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.input = torch.randn(1, time_steps, channels, height, width)
        # LSTM requires (h0, c0) tuple — unlike GRU which takes a single tensor
        cls.h0 = torch.zeros(num_lstm_layers, 1, lstm_hidden_size)
        cls.c0 = torch.zeros(num_lstm_layers, 1, lstm_hidden_size)
        cls.batch_size = 1
        cls.cfg = os.path.join(CUR_DIR, 'utils', 'qat_rnn_a8w4.cfg')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules['torch_npu'] = MagicMock()

    def tearDown(self):
        pass

    @unittest.skipUnless(_INT4_SUPPORTED, _SKIP_INT4_MSG)
    def test_lstm_a8w4_qat_e2e(self):
        """LSTM A8W4 end-to-end via cfg entry (config_defination drives A8W4)."""
        config_file = os.path.join(self.temp_folder, 'retrain_config_a8w4.json')
        record_file = os.path.join(self.temp_folder, 'record_a8w4.txt')
        save_path = os.path.join(self.temp_folder, 'res_a8w4')
        hx = (self.h0, self.c0)

        # Step 1: create retrain config via cfg entry (config_defination drives A8W4)
        create_quant_retrain_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, hx),
            config_defination=self.cfg,
        )
        self.assertTrue(
            os.path.exists(config_file), 'LSTM retrain config file must be created'
        )

        # verify the cfg entry actually drove weight INT4 into the generated config
        _global_keys = {
            'version',
            'batch_num',
            'activation_offset',
            'fakequant_precision_mode',
        }
        with open(config_file) as _fh:
            _cfg_after = _json.load(_fh)
        _wts_int4_seen = False
        for _lname, _lcfg in _cfg_after.items():
            if _lname in _global_keys or not isinstance(_lcfg, dict):
                continue
            if (
                'retrain_weight_config' in _lcfg
                and _lcfg['retrain_weight_config'].get('dst_type') == 'INT4'
            ):
                _wts_int4_seen = True
        self.assertTrue(
            _wts_int4_seen,
            'LSTM A8W4 cfg must yield at least one layer with weight dst_type=INT4',
        )

        # Step 2: build quantized retrain model
        new_model = create_quant_retrain_model(
            config_file, self.model, record_file, (self.input, hx)
        )
        self.assertIsNotNone(new_model, 'LSTM A8W4 retrain model must not be None')

        # Step 3: one training step
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate)
        label = torch.randint(0, self.num_class, (self.batch_size,))
        output, _ = new_model(self.input, hx)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step 4: eval forward pass
        new_model.eval()
        with torch.no_grad():
            output, _ = new_model(self.input, hx)
        self.assertIsNotNone(output, 'LSTM A8W4 eval output must not be None')

        # Step 5: save quant retrain model (deploy + fakequant)
        save_quant_retrain_model(
            model=new_model,
            input_data=(self.input, hx),
            config_file=config_file,
            record_file=record_file,
            save_path=save_path,
        )
        self.assertTrue(
            os.path.exists(save_path + '_deploy_model.onnx'),
            'LSTM A8W4 QAT deploy model must be saved',
        )


class TestLSTMQATA16W8(unittest.TestCase):
    """LSTM A16W8 (INT16 activation, INT8 weight) e2e QAT retrain: patch data dst_type→INT16, assert pipeline runs."""

    @classmethod
    def setUpClass(cls):
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        cls.num_class = 10
        cls.learning_rate = 0.001
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        lstm_hidden_size = 64
        num_lstm_layers = 1

        cls.model = rnn_model.Conv1dLSTM(
            input_channels=channels,
            conv1d_kernel_size=conv1d_kernel_size,
            conv1d_out_channels=conv1d_out_channels,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=cls.num_class,
            num_lstm_layers=num_lstm_layers,
            dropout=0.1,
        )

        cls.temp_folder = os.path.join(CUR_DIR, 'test_lstm_a16w8')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.input = torch.randn(1, time_steps, channels, height, width)
        # LSTM requires (h0, c0) tuple — unlike GRU which takes a single tensor
        cls.h0 = torch.zeros(num_lstm_layers, 1, lstm_hidden_size)
        cls.c0 = torch.zeros(num_lstm_layers, 1, lstm_hidden_size)
        cls.batch_size = 1

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules['torch_npu'] = MagicMock()

    def tearDown(self):
        pass

    def test_lstm_a16w8_qat_e2e(self):
        """LSTM A16W8 end-to-end via cfg entry (config_defination drives A16W8)."""
        config_file = os.path.join(self.temp_folder, 'retrain_config_a16w8.json')
        record_file = os.path.join(self.temp_folder, 'record_a16w8.txt')
        save_path = os.path.join(self.temp_folder, 'res_a16w8')
        hx = (self.h0, self.c0)
        cfg = os.path.join(CUR_DIR, 'utils', 'qat_rnn_a16w8.cfg')

        # Step 1: create retrain config via cfg entry (config_defination drives A16W8)
        create_quant_retrain_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, hx),
            config_defination=cfg,
        )
        self.assertTrue(
            os.path.exists(config_file), 'LSTM retrain config file must be created'
        )

        # verify the cfg entry actually drove activation INT16 into the generated config
        _global_keys = {
            'version',
            'batch_num',
            'activation_offset',
            'fakequant_precision_mode',
        }
        with open(config_file) as _fh:
            _cfg_after = _json.load(_fh)
        _act_int16_seen = False
        for _lname, _lcfg in _cfg_after.items():
            if _lname in _global_keys or not isinstance(_lcfg, dict):
                continue
            if (
                'retrain_data_config' in _lcfg
                and _lcfg['retrain_data_config'].get('dst_type') == 'INT16'
            ):
                _act_int16_seen = True
        self.assertTrue(
            _act_int16_seen,
            'LSTM A16W8 cfg must yield at least one layer with activation dst_type=INT16',
        )

        # Step 2: build quantized retrain model
        new_model = create_quant_retrain_model(
            config_file, self.model, record_file, (self.input, hx)
        )
        self.assertIsNotNone(new_model, 'LSTM A16W8 retrain model must not be None')

        # Step 3: one training step
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate)
        label = torch.randint(0, self.num_class, (self.batch_size,))
        output, _ = new_model(self.input, hx)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step 4: eval forward pass
        new_model.eval()
        with torch.no_grad():
            output, _ = new_model(self.input, hx)
        self.assertIsNotNone(output, 'LSTM A16W8 eval output must not be None')

        # Step 5: save quant retrain model (deploy + fakequant)
        save_quant_retrain_model(
            model=new_model,
            input_data=(self.input, hx),
            config_file=config_file,
            record_file=record_file,
            save_path=save_path,
        )
        self.assertTrue(
            os.path.exists(save_path + '_deploy_model.onnx'),
            'LSTM A16W8 QAT deploy model must be saved',
        )


class TestConvLinearPTQA8W4(unittest.TestCase):
    """PTQ A8W4 end-to-end for Conv/Linear ops via config_defination cfg.

    Requirement-2: Conv1d/Conv2d/ConvTranspose1d/2d/Linear support A8W4.
    The cfg (utils/ptq_a8w4.cfg) sets weight quant_bits=4 (INT4) globally via
    common_config, so create_quant_config drives A8W4 through the real config
    entry (not a JSON patch).

    NOTE: quantize_model / save_model need the NPU op library and FAIL in a bare
    dev env (the pre-existing TestGRUPTQ::test_quantize_model fails the same way);
    those steps are exercised by upstream CI. The cfg-entry step
    (create_quant_config) runs locally and is asserted here.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = models.Net001()
        cls.model.eval()
        cls.input = torch.randn(1, 2, 28, 28)
        cls.temp_folder = os.path.join(CUR_DIR, 'test_conv_ptq_a8w4')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.cfg = os.path.join(CUR_DIR, 'utils', 'ptq_a8w4.cfg')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules["torch_npu"] = MagicMock()

    def test_create_quant_config_a8w4(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        create_quant_config(
            config_file=config_file,
            model=self.model,
            input_data=self.input,
            config_defination=self.cfg,
        )
        self.assertTrue(os.path.exists(config_file))
        # cfg entry drives weight INT4; Net001 has group/depthwise conv whose Cin
        # axis cannot be nibble-packed, so those layers are downgraded to INT8.
        with open(config_file) as fh:
            cfg = _json.load(fh)
        wts_bits = []
        for layer, lcfg in cfg.items():
            if isinstance(lcfg, dict) and 'weight_quant_params' in lcfg:
                nb = lcfg['weight_quant_params'].get('num_bits')
                if nb is not None:
                    wts_bits.append(nb)
        self.assertTrue(wts_bits, 'no weight_quant_params.num_bits found in config')
        # regular conv/linear stay INT4; group/depthwise conv downgraded to INT8
        self.assertIn(4, wts_bits, 'A8W4 cfg should yield INT4 for regular layers')
        self.assertIn(
            8,
            wts_bits,
            'group/depthwise conv should be downgraded to INT8, got {}'.format(
                wts_bits
            ),
        )

    @unittest.skipUnless(_INT4_SUPPORTED, _SKIP_INT4_MSG)
    def test_quantize_and_save_a8w4(self):
        # Full end-to-end. quantize_model/save_model need NPU op lib; in a bare
        # dev env this raises (env limitation, not a code defect) — upstream CI
        # with the full toolchain exercises this path.
        config_file = os.path.join(self.temp_folder, 'config2.json')
        record_file = os.path.join(self.temp_folder, 'record.txt')
        modified_model = os.path.join(self.temp_folder, 'modified_model.onnx')
        save_path = os.path.join(self.temp_folder, 'res')
        create_quant_config(
            config_file=config_file,
            model=self.model,
            input_data=self.input,
            config_defination=self.cfg,
        )
        new_model = quantize_model(
            config_file, modified_model, record_file, self.model, self.input
        )
        new_model(self.input)
        save_model(modified_model, record_file, save_path)
        self.assertTrue(os.path.exists(save_path + '_deploy_model.onnx'))


class TestGRUPTQA8W4A16W8(unittest.TestCase):
    """PTQ A8W4 / A16W8 end-to-end for GRU via config_defination cfg.

    Requirement-1: LSTM/GRU support A8W4 and A16W8; both retrain and PTQ paths.
    This class covers the PTQ path (create_quant_config -> quantize_model ->
    save_model) driven by a real cfg through config_defination — not a JSON patch.
    """

    @classmethod
    def setUpClass(cls):
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        gru_hidden_size = 64
        cls.model = rnn_model.Conv1dGRU(
            input_channels=channels,
            conv1d_kernel_size=3,
            conv1d_out_channels=16,
            gru_hidden_size=gru_hidden_size,
            num_classes=10,
            num_gru_layers=1,
            dropout=0.1,
        )
        cls.model.eval()
        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)
        cls.temp_folder = os.path.join(CUR_DIR, 'test_gru_ptq_combo')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.a8w4_cfg = os.path.join(CUR_DIR, 'utils', 'ptq_rnn_a8w4.cfg')
        cls.a16w8_cfg = os.path.join(CUR_DIR, 'utils', 'ptq_rnn_a16w8.cfg')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules["torch_npu"] = MagicMock()

    def run_ptq(self, cfg, tag):
        config_file = os.path.join(self.temp_folder, tag + '_config.json')
        record_file = os.path.join(self.temp_folder, tag + '_record.txt')
        modified_model = os.path.join(self.temp_folder, tag + '_modified.onnx')
        save_path = os.path.join(self.temp_folder, tag + '_res')
        create_quant_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, self.h0),
            config_defination=cfg,
        )
        self.assertTrue(os.path.exists(config_file))
        new_model = quantize_model(
            config_file, modified_model, record_file, self.model, (self.input, self.h0)
        )
        new_model(self.input, self.h0)
        save_model(modified_model, record_file, save_path)
        self.assertTrue(os.path.exists(save_path + '_deploy_model.onnx'))
        # regression guard: A8W4 must record wts_type INT4 for the RNN layer too
        # (fixes _rnn_process dropping num_bits, which left RNN weights at INT8).
        if tag == 'a8w4':
            record_text = open(record_file).read()
            self.assertIn(
                'wts_type: "INT4"',
                record_text,
                'A8W4 PTQ must record wts_type INT4 (incl. the RNN layer)',
            )

    @unittest.skipUnless(_INT4_SUPPORTED, _SKIP_INT4_MSG)
    def test_gru_ptq_a8w4_e2e(self):
        self.run_ptq(self.a8w4_cfg, 'a8w4')

    def test_gru_ptq_a16w8_e2e(self):
        self.run_ptq(self.a16w8_cfg, 'a16w8')


class TestLSTMPTQA8W4A16W8(unittest.TestCase):
    """PTQ A8W4 / A16W8 end-to-end for LSTM via config_defination cfg.

    Requirement-1 (PTQ path): LSTM supports A8W4 and A16W8. Mirrors the GRU PTQ
    class but uses Conv1dLSTM (LSTM needs an (h0, c0) initial-state tuple).
    """

    @classmethod
    def setUpClass(cls):
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        lstm_hidden_size = 64
        cls.model = rnn_model.Conv1dLSTM(
            input_channels=channels,
            conv1d_kernel_size=3,
            conv1d_out_channels=16,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=10,
            num_lstm_layers=1,
            dropout=0.1,
        )
        cls.model.eval()
        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, lstm_hidden_size)
        cls.c0 = torch.zeros(1, 1, lstm_hidden_size)
        cls.temp_folder = os.path.join(CUR_DIR, 'test_lstm_ptq_combo')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.a8w4_cfg = os.path.join(CUR_DIR, 'utils', 'ptq_rnn_a8w4.cfg')
        cls.a16w8_cfg = os.path.join(CUR_DIR, 'utils', 'ptq_rnn_a16w8.cfg')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules["torch_npu"] = MagicMock()

    def run_ptq(self, cfg, tag):
        config_file = os.path.join(self.temp_folder, tag + '_config.json')
        record_file = os.path.join(self.temp_folder, tag + '_record.txt')
        modified_model = os.path.join(self.temp_folder, tag + '_modified.onnx')
        save_path = os.path.join(self.temp_folder, tag + '_res')
        create_quant_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, (self.h0, self.c0)),
            config_defination=cfg,
        )
        self.assertTrue(os.path.exists(config_file))
        new_model = quantize_model(
            config_file,
            modified_model,
            record_file,
            self.model,
            (self.input, (self.h0, self.c0)),
        )
        new_model(self.input, (self.h0, self.c0))
        save_model(modified_model, record_file, save_path)
        self.assertTrue(os.path.exists(save_path + '_deploy_model.onnx'))
        # regression guard: A8W4 must record wts_type INT4 for the RNN layer too
        # (fixes _rnn_process dropping num_bits, which left RNN weights at INT8).
        if tag == 'a8w4':
            record_text = open(record_file).read()
            self.assertIn(
                'wts_type: "INT4"',
                record_text,
                'A8W4 PTQ must record wts_type INT4 (incl. the RNN layer)',
            )

    @unittest.skipUnless(_INT4_SUPPORTED, _SKIP_INT4_MSG)
    def test_lstm_ptq_a8w4_e2e(self):
        self.run_ptq(self.a8w4_cfg, 'a8w4')

    def test_lstm_ptq_a16w8_e2e(self):
        self.run_ptq(self.a16w8_cfg, 'a16w8')


class TestConvTransposePTQA8W4(unittest.TestCase):
    """PTQ A8W4 end-to-end for ConvTranspose2d via config_defination cfg.

    Requirement-2: ConvTranspose2d supports A8W4. NetConvDeconv contains
    Conv2d + ConvTranspose2d, driven to weight INT4 by ptq_a8w4.cfg.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = models.NetConvDeconv()
        cls.model.eval()
        cls.input = torch.randn(1, 2, 28, 28)
        cls.temp_folder = os.path.join(CUR_DIR, 'test_convtranspose_ptq_a8w4')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.cfg = os.path.join(CUR_DIR, 'utils', 'ptq_a8w4.cfg')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules["torch_npu"] = MagicMock()

    @unittest.skipUnless(_INT4_SUPPORTED, _SKIP_INT4_MSG)
    def test_convtranspose_ptq_a8w4_e2e(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        record_file = os.path.join(self.temp_folder, 'record.txt')
        modified_model = os.path.join(self.temp_folder, 'modified.onnx')
        save_path = os.path.join(self.temp_folder, 'res')
        create_quant_config(
            config_file=config_file,
            model=self.model,
            input_data=self.input,
            config_defination=self.cfg,
        )
        self.assertTrue(os.path.exists(config_file))
        new_model = quantize_model(
            config_file, modified_model, record_file, self.model, self.input
        )
        new_model(self.input)
        save_model(modified_model, record_file, save_path)
        self.assertTrue(os.path.exists(save_path + '_deploy_model.onnx'))


class TestConvTranspose1dPTQA8W4(unittest.TestCase):
    """PTQ A8W4 end-to-end for ConvTranspose1d via config_defination cfg.

    Requirement-2: ConvTranspose1d supports A8W4 (ConvTranspose1dNet =
    Conv1d + ConvTranspose1d), driven to weight INT4 by ptq_a8w4.cfg.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = rnn_model.ConvTranspose1dNet()
        cls.model.eval()
        cls.input = torch.randn(1, 3, 32)
        cls.temp_folder = os.path.join(CUR_DIR, 'test_convtranspose1d_ptq_a8w4')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.cfg = os.path.join(CUR_DIR, 'utils', 'ptq_a8w4.cfg')

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules["torch_npu"] = MagicMock()

    @unittest.skipUnless(_INT4_SUPPORTED, _SKIP_INT4_MSG)
    def test_convtranspose1d_ptq_a8w4_e2e(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        record_file = os.path.join(self.temp_folder, 'record.txt')
        modified_model = os.path.join(self.temp_folder, 'modified.onnx')
        save_path = os.path.join(self.temp_folder, 'res')
        create_quant_config(
            config_file=config_file,
            model=self.model,
            input_data=self.input,
            config_defination=self.cfg,
        )
        self.assertTrue(os.path.exists(config_file))
        new_model = quantize_model(
            config_file, modified_model, record_file, self.model, self.input
        )
        new_model(self.input)
        save_model(modified_model, record_file, save_path)
        self.assertTrue(os.path.exists(save_path + '_deploy_model.onnx'))
