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
import shutil
import sys
import numpy as np
import unittest
from unittest import mock
from unittest.mock import patch
from collections import OrderedDict

import torch

from amct_pytorch.amct_pytorch_inner.amct_pytorch.proto import distill_config_pb2
from amct_pytorch.amct_pytorch_inner.amct_pytorch.capacity import CAPACITY
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config_base.distill_proto import DistillProtoConfig
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.vars_util import INT4, INT8


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestDistillProto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestDistillProto start!')
        config_proto_file = os.path.join(CUR_DIR, './utils/sample.cfg')
        cls.distill_proto_config = DistillProtoConfig(config_proto_file, CAPACITY)

    @classmethod
    def tearDownClass(cls):
        print('TestDistillProto end!')

    def test_parse_data_type_not_support(self):
        with self.assertRaises(ValueError):
            self.distill_proto_config.parse_data_type(-1)

    def test_parse_data_type_success(self):
        dst_type = self.distill_proto_config.parse_data_type(1)
        self.assertEqual(dst_type, INT8)

    def test_get_proto_global_config(self):
        global_config = self.distill_proto_config.get_proto_global_config()
        self.assertEqual(global_config, {'batch_num':2,'group_size':2,'data_dump':True})

    def test_get_distill_groups(self):
        distill_groups = self.distill_proto_config.get_distill_groups()
        self.assertEqual(distill_groups, [{'start_layer':'conv2', 'end_layer':'relu1'}])

    def test_get_distill_data_quant_config(self):
        data_config = self.distill_proto_config.get_distill_data_quant_config()
        self.assertEqual(data_config, {'algo':'ulq_quantize','clip_max':6,'clip_min':-6,'fixed_min':True,'dst_type':'INT8'})

    def test_get_distill_weight_quant_config(self):
        weight_config = self.distill_proto_config.get_distill_weight_quant_config()
        self.assertEqual(weight_config, {'algo':'arq_distill','channel_wise':False,'dst_type':'INT8'})

    def test_get_quant_skip_layers(self):
        skip_layers = self.distill_proto_config.get_quant_skip_layers()
        self.assertEqual(skip_layers, ['conv1'])

    def test_get_quant_skip_layer_types(self):
        skip_types = self.distill_proto_config.get_quant_skip_layer_types()
        self.assertEqual(skip_types, ['Linear'])

    def test_get_override_layers_repeat(self):
        config_proto_file = os.path.join(CUR_DIR, './utils/repeated_override.cfg')
        proto_config = DistillProtoConfig(config_proto_file, CAPACITY)
        with self.assertRaises(ValueError):
            proto_config.get_override_layers()

    def test_get_override_layers(self):
        override_layers = self.distill_proto_config.get_override_layers()
        self.assertEqual(override_layers, ['conv3'])

    def test_get_override_layer_types_repeat(self):
        config_proto_file = os.path.join(CUR_DIR, './utils/repeated_override.cfg')
        proto_config = DistillProtoConfig(config_proto_file, CAPACITY)
        with self.assertRaises(ValueError):
            proto_config.get_override_layer_types()

    def test_get_override_layer_types(self):
        override_types = self.distill_proto_config.get_override_layer_types()
        self.assertEqual(override_types, ['Conv2d'])

    def test_read_override_layer_config(self):
        data_config, weight_config = self.distill_proto_config.read_override_layer_config('conv3')
        self.assertEqual(data_config, {'algo':'ulq_quantize','clip_max':3,'clip_min':-3,'dst_type':'INT4'})
        self.assertEqual(weight_config, {'algo':'arq_distill','channel_wise':False,'dst_type':'INT4'})

    def test_read_override_type_config(self):
        data_config, weight_config = self.distill_proto_config.read_override_type_config('Conv2d')
        self.assertEqual(data_config, {'algo':'ulq_quantize','clip_max':9,'clip_min':-9,'fixed_min':False,'dst_type':'INT8'})
        self.assertEqual(weight_config, {'algo':'arq_distill','channel_wise':True,'dst_type':'INT8'})

    def test_check_distill_data_type_not_enable_int4(self):
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.common.capacity.query_capacity.Capacity.get_value', return_value=False):
            with self.assertRaises(ValueError):
                self.distill_proto_config._check_distill_data_type(INT4)

    def test_get_distill_weight_config_ulq(self):
        ulq_param = distill_config_pb2.WtsULQDistill()
        ulq_param.dst_type = 0
        ulq_param.channel_wise = False
        weight_param = distill_config_pb2.DistillWeightQuantConfig()
        weight_param.ulq_distill.CopyFrom(ulq_param)

        weight_config = self.distill_proto_config._get_distill_weight_config(weight_param)
        self.assertEqual(weight_config, {'algo':'ulq_distill','channel_wise':False,'dst_type':'INT4'})
