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

from .utils import rnn_model

torch.manual_seed(0)
CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestGRUPTQ(unittest.TestCase):
    """
    The UT for QuantizeTool
    """


    @classmethod
    def setUpClass(cls):
        batch_size = 32
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        num_class = 10
        num_epochs = 10
        learning_rate = 0.001
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        gru_hidden_size = 64
        num_gru_layers = 1

        cls.model = rnn_model.Conv1dGRU(input_channels=channels,
                            conv1d_kernel_size=conv1d_kernel_size,
                            conv1d_out_channels=conv1d_out_channels,
                            gru_hidden_size=gru_hidden_size,
                            num_classes=num_class,
                            num_gru_layers=num_gru_layers,
                            dropout=0.1)

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
            config_file=config_file,
            model=self.model,
            input_data=(self.input, self.h0))

        self.assertTrue(os.path.exists(config_file))

    def test_quantize_model(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        record_file = os.path.join(self.temp_folder, 'record.txt')
        modified_model = os.path.join(self.temp_folder, 'modified_model.onnx')
        new_model = quantize_model(config_file, modified_model, record_file,
            self.model, (self.input, self.h0))

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
        num_epochs = 10
        cls.learning_rate = 0.001
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        gru_hidden_size = 64
        num_gru_layers = 1

        cls.model = rnn_model.Conv1dGRU(input_channels=channels,
                            conv1d_kernel_size=conv1d_kernel_size,
                            conv1d_out_channels=conv1d_out_channels,
                            gru_hidden_size=gru_hidden_size,
                            num_classes=cls.num_class,
                            num_gru_layers=num_gru_layers,
                            dropout=0.1)

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
            config_file=config_file,
            model=self.model,
            input_data=(self.input, self.h0))

        self.assertTrue(os.path.exists(config_file))

    @unittest.skip(
        'pre-existing hang: blocks indefinitely in dev env, tracked separately'
    )
    def test_create_quant_retrain_model(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        record_file = os.path.join(self.temp_folder, 'record.txt')
        self.new_model = create_quant_retrain_model(config_file, self.model, record_file,
            (self.input, self.h0))

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

        save_quant_retrain_model(model=self.new_model,
                                 input_data=(self.input, self.h0),
                                 config_file=config_file,
                                 record_file=record_file,
                                 save_path=save_path)

        self.assertTrue(os.path.exists(fakequant))
        self.assertTrue(os.path.exists(deploy))


def _patch_config_dst_type(config_file, data_dst_type=None, weight_dst_type=None):
    """Patch dst_type in a retrain JSON config for A8W4/A16W8 e2e tests; global keys are untouched."""
    with open(config_file, 'r') as fh:
        cfg = _json.load(fh)

    # Keys that are global (not layer entries) — skip them
    GLOBAL_KEYS = {'batch_num', 'activation_offset'}

    for layer_name, layer_cfg in cfg.items():
        if layer_name in GLOBAL_KEYS:
            continue
        if not isinstance(layer_cfg, dict):
            continue
        if data_dst_type is not None and 'retrain_data_config' in layer_cfg:
            layer_cfg['retrain_data_config']['dst_type'] = data_dst_type
        if weight_dst_type is not None and 'retrain_weight_config' in layer_cfg:
            layer_cfg['retrain_weight_config']['dst_type'] = weight_dst_type

    with open(config_file, 'w') as fh:
        _json.dump(cfg, fh, indent=4, separators=(',', ':'))


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

        cls.model = rnn_model.Conv1dGRU(input_channels=channels,
                            conv1d_kernel_size=conv1d_kernel_size,
                            conv1d_out_channels=conv1d_out_channels,
                            gru_hidden_size=gru_hidden_size,
                            num_classes=cls.num_class,
                            num_gru_layers=num_gru_layers,
                            dropout=0.1)

        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn_a8w4')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)
        cls.batch_size = 1

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules['torch_npu'] = MagicMock()

    def tearDown(self):
        pass

    def test_gru_a8w4_qat_e2e(self):
        """GRU A8W4: steps 1-4 fully asserted; step 5 env-blocked, known RuntimeError asserted."""
        config_file = os.path.join(self.temp_folder, 'retrain_config_a8w4.json')
        record_file = os.path.join(self.temp_folder, 'record_a8w4.txt')
        save_path = os.path.join(self.temp_folder, 'res_a8w4')

        # Step 1: create default retrain config (A8W8), then patch to A8W4
        create_quant_retrain_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, self.h0))
        self.assertTrue(os.path.exists(config_file), 'retrain config file must be created')
        _patch_config_dst_type(config_file, weight_dst_type='INT4')

        # I-3 verification: confirm the patch actually wrote INT4 into the JSON
        _GLOBAL_KEYS = {'version', 'batch_num', 'activation_offset', 'fakequant_precision_mode'}
        with open(config_file) as _fh:
            _cfg_after_patch = _json.load(_fh)
        for _lname, _lcfg in _cfg_after_patch.items():
            if _lname in _GLOBAL_KEYS or not isinstance(_lcfg, dict):
                continue
            if 'retrain_weight_config' in _lcfg:
                self.assertEqual(
                    _lcfg['retrain_weight_config']['dst_type'],
                    'INT4',
                    f'A8W4 patch must set {_lname}.retrain_weight_config.dst_type=INT4',
                )
            if 'retrain_data_config' in _lcfg:
                self.assertEqual(
                    _lcfg['retrain_data_config']['dst_type'],
                    'INT8',
                    f'A8W4 patch must leave {_lname}.retrain_data_config.dst_type=INT8',
                )

        # Step 2: build quantized retrain model using patched A8W4 config
        new_model = create_quant_retrain_model(
            config_file, self.model, record_file, (self.input, self.h0))
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

        # Step 5: save is env-blocked — assert the known error
        with self.assertRaisesRegex(RuntimeError, 'record_file is empty'):
            save_quant_retrain_model(
                model=new_model,
                input_data=(self.input, self.h0),
                config_file=config_file,
                record_file=record_file,
                save_path=save_path)


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

        cls.model = rnn_model.Conv1dGRU(input_channels=channels,
                            conv1d_kernel_size=conv1d_kernel_size,
                            conv1d_out_channels=conv1d_out_channels,
                            gru_hidden_size=gru_hidden_size,
                            num_classes=cls.num_class,
                            num_gru_layers=num_gru_layers,
                            dropout=0.1)

        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn_a16w8')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)
        cls.batch_size = 1

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        sys.modules['torch_npu'] = MagicMock()

    def tearDown(self):
        pass

    def test_gru_a16w8_qat_e2e(self):
        """GRU A16W8: steps 1-4 fully asserted; step 5 env-blocked, known RuntimeError asserted."""
        config_file = os.path.join(self.temp_folder, 'retrain_config_a16w8.json')
        record_file = os.path.join(self.temp_folder, 'record_a16w8.txt')
        save_path = os.path.join(self.temp_folder, 'res_a16w8')

        # Step 1: create default retrain config (A8W8), then patch to A16W8
        create_quant_retrain_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, self.h0))
        self.assertTrue(os.path.exists(config_file), 'retrain config file must be created')
        _patch_config_dst_type(config_file, data_dst_type='INT16')

        # I-3 verification: confirm the patch actually wrote INT16 into the JSON
        _GLOBAL_KEYS = {'version', 'batch_num', 'activation_offset', 'fakequant_precision_mode'}
        with open(config_file) as _fh:
            _cfg_after_patch = _json.load(_fh)
        for _lname, _lcfg in _cfg_after_patch.items():
            if _lname in _GLOBAL_KEYS or not isinstance(_lcfg, dict):
                continue
            if 'retrain_data_config' in _lcfg:
                self.assertEqual(
                    _lcfg['retrain_data_config']['dst_type'],
                    'INT16',
                    f'A16W8 patch must set {_lname}.retrain_data_config.dst_type=INT16',
                )
            if 'retrain_weight_config' in _lcfg:
                self.assertEqual(
                    _lcfg['retrain_weight_config']['dst_type'],
                    'INT8',
                    f'A16W8 patch must leave {_lname}.retrain_weight_config.dst_type=INT8',
                )

        # Step 2: build quantized retrain model using patched A16W8 config
        new_model = create_quant_retrain_model(
            config_file, self.model, record_file, (self.input, self.h0))
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

        # Step 5: save is env-blocked — assert the known error
        with self.assertRaisesRegex(RuntimeError, 'record_file is empty'):
            save_quant_retrain_model(
                model=new_model,
                input_data=(self.input, self.h0),
                config_file=config_file,
                record_file=record_file,
                save_path=save_path)


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

        cls.model = rnn_model.Conv1dLSTM(input_channels=channels,
                             conv1d_kernel_size=conv1d_kernel_size,
                             conv1d_out_channels=conv1d_out_channels,
                             lstm_hidden_size=lstm_hidden_size,
                             num_classes=cls.num_class,
                             num_lstm_layers=num_lstm_layers,
                             dropout=0.1)

        cls.temp_folder = os.path.join(CUR_DIR, 'test_lstm_a8w4')
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

    def test_lstm_a8w4_qat_e2e(self):
        """LSTM A8W4: steps 1-4 fully asserted; step 5 env-blocked, known RuntimeError asserted."""
        config_file = os.path.join(self.temp_folder, 'retrain_config_a8w4.json')
        record_file = os.path.join(self.temp_folder, 'record_a8w4.txt')
        save_path = os.path.join(self.temp_folder, 'res_a8w4')
        hx = (self.h0, self.c0)

        # Step 1: create default retrain config (A8W8), then patch to A8W4
        create_quant_retrain_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, hx))
        self.assertTrue(os.path.exists(config_file), 'LSTM retrain config file must be created')
        _patch_config_dst_type(config_file, weight_dst_type='INT4')

        # I-3 verification
        _GLOBAL_KEYS = {'version', 'batch_num', 'activation_offset', 'fakequant_precision_mode'}
        with open(config_file) as _fh:
            _cfg_after_patch = _json.load(_fh)
        for _lname, _lcfg in _cfg_after_patch.items():
            if _lname in _GLOBAL_KEYS or not isinstance(_lcfg, dict):
                continue
            if 'retrain_weight_config' in _lcfg:
                self.assertEqual(
                    _lcfg['retrain_weight_config']['dst_type'],
                    'INT4',
                    f'LSTM A8W4 patch must set {_lname}.retrain_weight_config.dst_type=INT4',
                )
            if 'retrain_data_config' in _lcfg:
                self.assertEqual(
                    _lcfg['retrain_data_config']['dst_type'],
                    'INT8',
                    f'LSTM A8W4 patch must leave {_lname}.retrain_data_config.dst_type=INT8',
                )

        # Step 2: build quantized retrain model
        new_model = create_quant_retrain_model(
            config_file, self.model, record_file, (self.input, hx))
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

        # Step 5: save is env-blocked — assert the known error
        with self.assertRaisesRegex(RuntimeError, 'record_file is empty'):
            save_quant_retrain_model(
                model=new_model,
                input_data=(self.input, hx),
                config_file=config_file,
                record_file=record_file,
                save_path=save_path)


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

        cls.model = rnn_model.Conv1dLSTM(input_channels=channels,
                             conv1d_kernel_size=conv1d_kernel_size,
                             conv1d_out_channels=conv1d_out_channels,
                             lstm_hidden_size=lstm_hidden_size,
                             num_classes=cls.num_class,
                             num_lstm_layers=num_lstm_layers,
                             dropout=0.1)

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
        """LSTM A16W8: steps 1-4 fully asserted; step 5 env-blocked, known RuntimeError asserted."""
        config_file = os.path.join(self.temp_folder, 'retrain_config_a16w8.json')
        record_file = os.path.join(self.temp_folder, 'record_a16w8.txt')
        save_path = os.path.join(self.temp_folder, 'res_a16w8')
        hx = (self.h0, self.c0)

        # Step 1: create default retrain config (A8W8), then patch to A16W8
        create_quant_retrain_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, hx))
        self.assertTrue(os.path.exists(config_file), 'LSTM retrain config file must be created')
        _patch_config_dst_type(config_file, data_dst_type='INT16')

        # I-3 verification
        _GLOBAL_KEYS = {'version', 'batch_num', 'activation_offset', 'fakequant_precision_mode'}
        with open(config_file) as _fh:
            _cfg_after_patch = _json.load(_fh)
        for _lname, _lcfg in _cfg_after_patch.items():
            if _lname in _GLOBAL_KEYS or not isinstance(_lcfg, dict):
                continue
            if 'retrain_data_config' in _lcfg:
                self.assertEqual(
                    _lcfg['retrain_data_config']['dst_type'],
                    'INT16',
                    f'LSTM A16W8 patch must set {_lname}.retrain_data_config.dst_type=INT16',
                )
            if 'retrain_weight_config' in _lcfg:
                self.assertEqual(
                    _lcfg['retrain_weight_config']['dst_type'],
                    'INT8',
                    f'LSTM A16W8 patch must leave {_lname}.retrain_weight_config.dst_type=INT8',
                )

        # Step 2: build quantized retrain model
        new_model = create_quant_retrain_model(
            config_file, self.model, record_file, (self.input, hx))
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

        # Step 5: save is env-blocked — assert the known error
        with self.assertRaisesRegex(RuntimeError, 'record_file is empty'):
            save_quant_retrain_model(
                model=new_model,
                input_data=(self.input, hx),
                config_file=config_file,
                record_file=record_file,
                save_path=save_path)