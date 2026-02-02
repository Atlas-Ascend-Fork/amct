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

from amct_pytorch.amct_pytorch_inner.amct_pytorch.capacity import CAPACITY
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.distill_config_base import distill_field
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.check import GraphQuerier


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestDistillField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestDistillField start!')

    @classmethod
    def tearDownClass(cls):
        print('TestDistillField end!')

    def test_group_size_0(self):
        obj = distill_field.GroupSize(GraphQuerier, CAPACITY)
        with self.assertRaises(ValueError):
            obj.build(0)

    def test_group_size_2(self):
        obj = distill_field.GroupSize(GraphQuerier, CAPACITY)
        obj.build(2)
        self.assertEqual(obj.value, 2)

    def test_group_size_default(self):
        obj = distill_field.GroupSize(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.value, 1)

    def test_data_dump_true(self):
        obj = distill_field.DataDump(GraphQuerier, CAPACITY)
        obj.build(True)
        self.assertEqual(obj.value, True)

    def test_data_dump_default(self):
        obj = distill_field.DataDump(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.value, False)

    def test_quant_enable_false(self):
        obj = distill_field.QuantEnable(GraphQuerier, CAPACITY)
        extra = ('conv1', 'Conv2d')
        obj.build(False, extra)
        self.assertEqual(obj.value, False)

    def test_quant_enable_default(self):
        obj = distill_field.QuantEnable(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.value, True)

    def test_data_type_unknown(self):
        obj = distill_field.DataType(GraphQuerier, CAPACITY)
        val = 'UNKNOWN'
        extra = ('conv1', 'Conv2d')
        with self.assertRaises(ValueError):
            obj.build(val, extra)

    def test_data_type_int4(self):
        obj = distill_field.DataType(GraphQuerier, CAPACITY)
        val = 'INT4'
        extra = ('conv1', 'Conv2d')
        obj.build(val, extra)
        self.assertEqual(obj.value, val)

    def test_data_type_default(self):
        obj = distill_field.DataType(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.value, 'INT8')

    def test_distill_data_config_all(self):
        obj = distill_field.DistillDataConfig(GraphQuerier, CAPACITY)
        val = {'algo':'ulq_quantize',
            'clip_max':1.0,
            'clip_min':-1.0,
            'fixed_min':True,
            'dst_type':'INT8'}
        extra = ('conv1', 'Conv2d')
        obj.build(val.copy(), extra)
        self.assertEqual(obj.dump(), val)

    def test_distill_data_config_invalid_field(self):
        obj = distill_field.DistillDataConfig(GraphQuerier, CAPACITY)
        val = {'invalid_field':1}
        extra = ('conv1', 'Conv2d')
        with self.assertRaises(ValueError):
            obj.build(val, extra)

    def test_distill_data_config_default(self):
        obj = distill_field.DistillDataConfig(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.dump(), {'algo':'ulq_quantize','dst_type':'INT8'})

    def test_weight_algo_invalid(self):
        obj = distill_field.WeightAlgo(GraphQuerier, CAPACITY)
        val = 'invalid_algo'
        extra = ('conv1', 'Conv2d')
        with self.assertRaises(ValueError):
            obj.build(val, extra)

    def test_weight_algo_valid(self):
        obj = distill_field.WeightAlgo(GraphQuerier, CAPACITY)
        val = 'ulq_distill'
        extra = ('conv1', 'Conv2d')
        obj.build(val, extra)
        self.assertEqual(obj.value, val)

    def test_weight_algo_default(self):
        obj = distill_field.WeightAlgo(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.value, 'arq_distill')

    def test_distill_weight_config_success(self):
        obj = distill_field.DistillWeightConfig(GraphQuerier, CAPACITY)
        val = {'algo':'arq_distill','channel_wise':False}
        extra = ('conv1', 'Conv2d')
        obj.build(val, extra)
        self.assertEqual(obj.dump(), {'algo':'arq_distill','channel_wise':False,'dst_type':'INT8'})

    def test_distill_weight_config_invalid(self):
        obj = distill_field.DistillWeightConfig(GraphQuerier, CAPACITY)
        val = {'invalid':1}
        extra = ('conv1', 'Conv2d')
        with self.assertRaises(ValueError):
            obj.build(val, extra)

    def test_distill_weight_config_default(self):
        obj = distill_field.DistillWeightConfig(GraphQuerier, CAPACITY)
        extra = ('conv1', 'Conv2d')
        obj.build_default(extra)
        self.assertEqual(obj.dump(), {'algo':'arq_distill','channel_wise':True,'dst_type':'INT8'})

    def test_layer_config_set_data_config(self):
        obj = distill_field.LayerConfig(GraphQuerier, CAPACITY)
        val = {'quant_enable':True,'distill_data_config':{'algo':'ulq_quantize','dst_type':'INT4'}}
        extra = ('conv1', 'Conv2d')
        obj.build(val, extra)
        except_val = {'quant_enable':True,
            'distill_data_config':{'algo':'ulq_quantize','dst_type':'INT4'},
            'distill_weight_config':{'algo':'arq_distill','channel_wise':True,'dst_type':'INT8'}}
        self.assertEqual(obj.dump(), except_val)

    def test_layer_config_set_weight_config(self):
        obj = distill_field.LayerConfig(GraphQuerier, CAPACITY)
        val = {'quant_enable':True,'distill_weight_config':{'algo':'arq_distill','channel_wise':True,'dst_type':'INT4'}}
        extra = ('conv1', 'Conv2d')
        obj.build(val, extra)
        except_val = {'quant_enable':True,
            'distill_data_config':{'algo':'ulq_quantize','dst_type':'INT8'},
            'distill_weight_config':{'algo':'arq_distill','channel_wise':True,'dst_type':'INT4'}}
        self.assertEqual(obj.dump(), except_val)

    def test_layer_config_default(self):
        obj = distill_field.LayerConfig(GraphQuerier, CAPACITY)
        extra = ('conv1', 'Conv2d')
        obj.build_default(extra)
        except_val = {'quant_enable':True,
            'distill_data_config':{'algo':'ulq_quantize','dst_type':'INT8'},
            'distill_weight_config':{'algo':'arq_distill','channel_wise':True,'dst_type':'INT8'}}
        self.assertEqual(obj.dump(), except_val)

    def test_distill_group(self):
        obj = distill_field.DistillGroup(GraphQuerier, CAPACITY)
        val = [['layer1','layer2'],['layer3','layer4']]
        obj.build(val)
        self.assertEqual(obj.value, val)

    def test_distill_root_config_not_supported_layer(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        value = {'version':1,'batch_num':1,'group_size':1,'data_dump':False,
            'distill_group':[['layer1','layer2']],
            'not_supported_layer':{}}
        extra = {'conv1':'Conv2d'}
        with self.assertRaises(ValueError):
            obj.build(value, extra)

    def test_distill_root_config_no_enabled_layer(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        value = {'version':1,'batch_num':1,'group_size':1,'data_dump':False,
            'distill_group':[['layer1','layer2']],
            'conv1':{'quant_enable':False}}
        extra = {'conv1':'Conv2d'}
        with self.assertRaises(ValueError):
            obj.build(value, extra)

    def test_distill_root_config_layer_dst_type_not_equal(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        value = {'version':1,'batch_num':1,'group_size':1,'data_dump':False,
            'distill_group':[['layer1','layer2']],
            'conv1':{'quant_enable':True,
                'distill_data_config':{'algo':'ulq_quantize','dst_type':'INT8'},
                'distill_weight_config':{'algo':'arq_distill','channel_wise':True,'dst_type':'INT4'}}}
        extra = {'conv1':'Conv2d'}
        with self.assertRaises(ValueError):
            obj.build(value, extra)

    def test_distill_root_config_success(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        value = {'version':1,'batch_num':2,'group_size':2,'data_dump':True,
            'distill_group':[['layer1','layer2']],
            'conv1':{'quant_enable':True}}
        extra = {'conv1':'Conv2d','conv2':'Conv2d'}
        obj.build(value, extra)
        except_val = {'version':1,'batch_num':2,'group_size':2,'data_dump':True,
            'distill_group':[['layer1','layer2']],
            'conv1':{'quant_enable':True,
                'distill_data_config':{'algo':'ulq_quantize','dst_type':'INT8'},
                'distill_weight_config':{'algo':'arq_distill','channel_wise':True,'dst_type':'INT8'}},
            'conv2':{'quant_enable':False,
                'distill_data_config':{'algo':'ulq_quantize','dst_type':'INT8'},
                'distill_weight_config':{'algo':'arq_distill','channel_wise':True,'dst_type':'INT8'}}}
        self.assertEqual(obj.dump(), except_val)

    def test_distill_root_config_default(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        groups = [['layer1','layer2']]
        extra = {'conv1':'Conv2d'}
        obj.build_default(groups, extra)
        except_val = {'version':1,'batch_num':1,'group_size':1,'data_dump':False,
            'distill_group':[['layer1','layer2']],
            'conv1':{'quant_enable':True,
                'distill_data_config':{'algo':'ulq_quantize','dst_type':'INT8'},
                'distill_weight_config':{'algo':'arq_distill','channel_wise':True,'dst_type':'INT8'}}}
        self.assertEqual(obj.dump(), except_val)