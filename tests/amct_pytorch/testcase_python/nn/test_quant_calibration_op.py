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
import sys
import unittest
import shutil
import json
import logging
import io
import copy

from google.protobuf import text_format
import torch
import numpy as np

from amct_pytorch.amct_pytorch_inner.amct_pytorch.nn.module.quantization.quant_calibration_op import QuantCalibrationOp
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.log import LOGGER
from amct_pytorch.amct_pytorch_inner.amct_pytorch.proto import scale_offset_record_pb2

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


def read_kv_cache_factors(record_file, layer_name):
    records = scale_offset_record_pb2.ScaleOffsetRecord()
    with open(record_file, 'r') as read_file:
        pbtxt_string = read_file.read()
        text_format.Merge(pbtxt_string, records)
 
    done_flag = False
    scale = []
    offset = []
    for record in records.record:
        if record.key == layer_name:
            if not record.kv_cache_value.scale:
                raise RuntimeError("Cannot find scale of layer {} in record file".format(layer_name))
            scale.extend(record.kv_cache_value.scale)
            if not record.kv_cache_value.offset:
                raise RuntimeError("Cannot find offset of layer {} in record file".format(layer_name))
            offset.extend(record.kv_cache_value.offset)
            done_flag = True
            break
    if not done_flag:
        raise RuntimeError("Cannot find layer {} in record file".format(layer_name))
    return scale, offset
 
 
def do_quant_antiquant(data, scale, offset):
    if len(scale) == 1:
        scale *= data.shape[-1]
        offset *= data.shape[-1]
    quant_channels = []
    for i, channel in enumerate(torch.chunk(data, data.shape[-1], dim=-1)):
        quant_channel = channel / scale[i] + offset[i]
        quant_channels.append(quant_channel.to(torch.int8))

    antiquant_channels = []
    for i in range(len(quant_channels)):
        antiquant_channel = (quant_channels[i].to(data.dtype) - offset[i]) * scale[i]
        antiquant_channels.append(antiquant_channel)
    antiquant_data = torch.cat(antiquant_channels, dim=-1)

    return antiquant_data
 
 
def calc_similarity(data0, data1):
    data0_nan = np.isnan(data0)
    data0[data0_nan] = 1
    data1_nan = np.isnan(data1)
    data1[data1_nan] = 1
    similarity = similarity_1 = np.sum(np.multiply(data0, data1).astype(np.float64))\
                    /(np.sqrt(np.sum(data0.astype(np.float64)**2))\
                    *np.sqrt(np.sum(data1.astype(np.float64)**2)))*100
    if (data0 == data1).all():
        similarity = 100
    if np.isnan(similarity) or np.isinf(similarity):
        data0 = np.divide(data0,np.power(10,38))
        data1 = np.divide(data1,np.power(10,38))
        similarity = similarity_1 = np.sum(np.multiply(data0, data1).astype(np.float64))\
                    /(np.sqrt(np.sum(data0.astype(np.float64)**2))\
                    *np.sqrt(np.sum(data1.astype(np.float64)**2)))*100
        if np.isnan(similarity) or np.isinf(similarity):
            data0 = np.divide(data0,np.power(10,38))
            data1 = np.divide(data1,np.power(10,38))
            similarity = similarity_1 = np.sum(np.multiply(data0, data1).astype(np.float64))\
                    /(np.sqrt(np.sum(data0.astype(np.float64)**2))\
                    *np.sqrt(np.sum(data1.astype(np.float64)**2)))*100
    if np.isnan(similarity):
        similarity = 0
    return similarity


class TestQuantCalibrationOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestQuantCalibrationOp start!')
        cls.temp_dir = os.path.join(CUR_DIR, 'temp')
        if not os.path.exists(cls.temp_dir):
            os.mkdir(cls.temp_dir)
        cls.record_file = os.path.join(cls.temp_dir, 'record.txt')

    @classmethod
    def tearDownClass(cls):
        print('TestQuantCalibrationOp end!')
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_quant_calibration_op_failed_invalid_input_type(self):
        self.assertRaises(TypeError, QuantCalibrationOp,
            1, {'act_algo': 'ifmr'}, 'kv_cache_quant')
        self.assertRaises(TypeError, QuantCalibrationOp,
            self.record_file, 1, 'kv_cache_quant')
        self.assertRaises(TypeError, QuantCalibrationOp,
            self.record_file, {'act_algo': 'ifmr'}, 1)

    def test_quant_calibration_op_failed_invalid_algo_params(self):
        self.assertRaises(TypeError, QuantCalibrationOp,
            self.record_file, {'batch_num': 2.2})
        self.assertRaises(TypeError, QuantCalibrationOp,
            self.record_file, {'asymmetric': 'a'})
        self.assertRaises(TypeError, QuantCalibrationOp,
            self.record_file, {'quant_granularity': 1})
        self.assertRaises(TypeError, QuantCalibrationOp,
            self.record_file, {'max_percentile': 'a'})
        self.assertRaises(TypeError, QuantCalibrationOp,
            self.record_file, {'min_percentile': 'a'})
        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'search_range': 2})
        self.assertRaises(TypeError, QuantCalibrationOp,
            self.record_file, {'search_step': 'a'})
        self.assertRaises(TypeError, QuantCalibrationOp,
            self.record_file, {'act_algo': 'hfmg', 'num_of_bins': 'a'})

    def test_quant_calibration_op_failed_invalid_algo_params01(self):
        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'act_algo': 1})
        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'batch_num': -1})
        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'quant_granularity': 'abc'})
        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'max_percentile': 1.2})
        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'search_range': [1.2, 0.8]})
        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'search_step': 0.})
        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'act_algo': 'hfmg', 'num_of_bins': 1023})

        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'act_algo': 'hfmg', 'search_step': 0.01})

        self.assertRaises(RuntimeError, QuantCalibrationOp,
            self.record_file, {'act_algo': 'ifmr', 'num_of_bins': 1024})

    def test_quant_calibration_op_failed_invalid_input(self):
        op = QuantCalibrationOp(self.record_file)
        self.assertRaises(TypeError, op, 'a', torch.randn(3,4).to(dtype=torch.float64))
        op = QuantCalibrationOp(self.record_file)
        input_data = torch.randn(3,4)
        input_data[0][0] = np.inf
        self.assertRaises(RuntimeError, op, 'a', input_data)


        op = QuantCalibrationOp(self.record_file, {'act_algo': 'hfmg'})
        self.assertRaises(TypeError, op, 'a', torch.randn(3,4).to(dtype=torch.float64))
        op = QuantCalibrationOp(self.record_file, {'act_algo': 'hfmg'})
        input_data = torch.randn(3,4)
        input_data[0][0] = np.inf
        self.assertRaises(RuntimeError, op, 'a', input_data)

    def test_quant_calibration_op_success_ifmr(self):
        op = QuantCalibrationOp(self.record_file)
        input_data = torch.randn(128,128)
        op('a', input_data)
        with open(self.record_file) as f:
            self.assertIn('a', f.read())
        scale, offset = read_kv_cache_factors(self.record_file, 'a')
        fakequant_data = do_quant_antiquant(input_data, scale, offset)
        self.assertGreater(calc_similarity(input_data.cpu().numpy(), fakequant_data.cpu().numpy()), 99)

        op = QuantCalibrationOp(self.record_file)
        input_data = torch.randn(1,1)
        op('a', input_data)
        with open(self.record_file) as f:
            self.assertIn('a', f.read())
        scale, offset = read_kv_cache_factors(self.record_file, 'a')
        fakequant_data = do_quant_antiquant(input_data, scale, offset)
        self.assertGreater(calc_similarity(input_data.cpu().numpy(), fakequant_data.cpu().numpy()), 99)

        op = QuantCalibrationOp(self.record_file)
        input_data = torch.ones(128,128)
        op('a', input_data)
        with open(self.record_file) as f:
            self.assertIn('a', f.read())
        scale, offset = read_kv_cache_factors(self.record_file, 'a')
        fakequant_data = do_quant_antiquant(input_data, scale, offset)
        self.assertGreater(calc_similarity(input_data.cpu().numpy(), fakequant_data.cpu().numpy()), 99)

        op = QuantCalibrationOp(self.record_file)
        input_data = torch.zeros(128,128)
        op('a', input_data)
        with open(self.record_file) as f:
            self.assertIn('a', f.read())
        scale, offset = read_kv_cache_factors(self.record_file, 'a')
        fakequant_data = do_quant_antiquant(input_data, scale, offset)
        self.assertGreater(calc_similarity(input_data.cpu().numpy(), fakequant_data.cpu().numpy()), 99)

    def test_quant_calibration_op_success_hfmg(self):
        op = QuantCalibrationOp(self.record_file, {'act_algo': 'hfmg'})
        input_data = torch.randn(1,1)
        op('a', input_data)
        with open(self.record_file) as f:
            self.assertIn('a', f.read())
        scale, offset = read_kv_cache_factors(self.record_file, 'a')
        fakequant_data = do_quant_antiquant(input_data, scale, offset)
        self.assertGreater(calc_similarity(input_data.cpu().numpy(), fakequant_data.cpu().numpy()), 99)

        op = QuantCalibrationOp(self.record_file, {'act_algo': 'hfmg'})
        input_data = torch.randn(128,128)
        op('a', input_data)
        with open(self.record_file) as f:
            self.assertIn('a', f.read())
        scale, offset = read_kv_cache_factors(self.record_file, 'a')
        fakequant_data = do_quant_antiquant(input_data, scale, offset)
        self.assertGreater(calc_similarity(input_data.cpu().numpy(), fakequant_data.cpu().numpy()), 99)

        op = QuantCalibrationOp(self.record_file, {'act_algo': 'hfmg'})
        input_data = torch.zeros(128,128)
        op('a', input_data)
        with open(self.record_file) as f:
            self.assertIn('a', f.read())
        scale, offset = read_kv_cache_factors(self.record_file, 'a')
        fakequant_data = do_quant_antiquant(input_data, scale, offset)
        self.assertGreater(calc_similarity(input_data.cpu().numpy(), fakequant_data.cpu().numpy()), 99)

        op = QuantCalibrationOp(self.record_file, {'act_algo': 'hfmg'})
        input_data = torch.ones(128,128)
        op('a', input_data)
        with open(self.record_file) as f:
            self.assertIn('a', f.read())
        scale, offset = read_kv_cache_factors(self.record_file, 'a')
        fakequant_data = do_quant_antiquant(input_data, scale, offset)
        self.assertGreater(calc_similarity(input_data.cpu().numpy(), fakequant_data.cpu().numpy()), 99)

    def test_quant_calibration_op_success_special_condition(self):
        class CustomizedModel(torch.nn.Module):
            def __init__(self, record_1, record_2):
                super().__init__()
                self.matmul1 = torch.nn.Linear(128, 32)
                self.matmul2 = torch.nn.Linear(128, 64)
                self.matmul3 = torch.nn.Linear(128, 4)
                self.quant_cali_op = QuantCalibrationOp(record_1)
                self.quant_cali_op1 = QuantCalibrationOp(record_2)
                self.quant_cali_op2 = QuantCalibrationOp(record_2)
                self.quant_cali_op3 = QuantCalibrationOp(record_2, {'batch_num': 2})
 
            def forward(self, inputs):
                output = list()
                y = self.matmul1(inputs)
                y = self.quant_cali_op('matmul1', y)
                output.append(y)
                y = self.matmul2(inputs)
                y = self.quant_cali_op1('matmul2', y)
                y = self.quant_cali_op2('matmul2', y)
                output.append(y)
                y = self.matmul3(inputs)
                y = self.quant_cali_op3('matmul4', y)
                output.append(y)
                return output
        logger = LOGGER.logger
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        record2 = os.path.join(self.temp_dir, 'record2.txt')
        model = CustomizedModel(self.record_file, record2)
        model.eval()
        model.to(dtype=torch.float32)
        input_data = torch.randn(128, 128)
        output = model(input_data)
        output = model(input_data)
        with open(self.record_file) as f:
            self.assertIn('matmul1', f.read())
        with open(os.path.join(self.temp_dir, 'record2.txt')) as f:
            self.assertIn('matmul2', f.read())
            self.assertNotIn('matmul2', f.read())
        model(input_data)
        with open(os.path.join(self.temp_dir, 'record2.txt')) as f:
            self.assertIn('matmul2', f.read())
        log_info = log_stream.getvalue()
        logger.removeHandler(handler)
        self.assertIn('will be overwritten by AMCT', log_info)

        scale, offset = read_kv_cache_factors(self.record_file, 'matmul1')
        fakequant_data = do_quant_antiquant(output[0], scale, offset)
        self.assertGreater(calc_similarity(output[0].detach().cpu().numpy(), fakequant_data.detach().cpu().numpy()), 90)

        scale, offset = read_kv_cache_factors(record2, 'matmul2')
        fakequant_data = do_quant_antiquant(output[1], scale, offset)
        self.assertGreater(calc_similarity(output[1].detach().cpu().numpy(), fakequant_data.detach().cpu().numpy()), 90)

        scale, offset = read_kv_cache_factors(record2, 'matmul4')
        fakequant_data = do_quant_antiquant(output[2], scale, offset)
        self.assertGreater(calc_similarity(output[2].detach().cpu().numpy(), fakequant_data.detach().cpu().numpy()), 90)
