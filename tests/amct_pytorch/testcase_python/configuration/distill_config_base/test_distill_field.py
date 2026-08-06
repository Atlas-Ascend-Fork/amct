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
import logging
import os
import unittest


from amct_pytorch.classic.graph_based.amct_pytorch.capacity import CAPACITY
from amct_pytorch.classic.graph_based.amct_pytorch.configuration.check import (
    GraphQuerier,
)
from amct_pytorch.classic.graph_based.amct_pytorch.configuration.distill_config_base import (
    distill_field,
)

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

logger = logging.getLogger(__name__)

ALGO = 'algo'

CONV1 = 'conv1'
DST_TYPE = 'dst_type'

QUANT_ENABLE = 'quant_enable'

INT8 = 'INT8'


class TestDistillField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.info('TestDistillField start!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestDistillField end!')

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
        self.assertTrue(obj.value)

    def test_data_dump_default(self):
        obj = distill_field.DataDump(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertFalse(obj.value)

    def test_quant_enable_false(self):
        obj = distill_field.QuantEnable(GraphQuerier, CAPACITY)
        extra = (CONV1, 'Conv2d')
        obj.build(False, extra)
        self.assertFalse(obj.value)

    def test_quant_enable_default(self):
        obj = distill_field.QuantEnable(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertTrue(obj.value)

    def test_data_type_unknown(self):
        obj = distill_field.DataType(GraphQuerier, CAPACITY)
        val = 'UNKNOWN'
        extra = (CONV1, 'Conv2d')
        with self.assertRaises(ValueError):
            obj.build(val, extra)

    def test_data_type_int4(self):
        obj = distill_field.DataType(GraphQuerier, CAPACITY)
        val = 'INT4'
        extra = (CONV1, 'Conv2d')
        obj.build(val, extra)
        self.assertEqual(obj.value, val)

    def test_data_type_default(self):
        obj = distill_field.DataType(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.value, INT8)

    def test_distill_data_config_all(self):
        obj = distill_field.DistillDataConfig(GraphQuerier, CAPACITY)
        val = {
            ALGO: "ulq_quantize",
            'clip_max': 1.0,
            'clip_min': -1.0,
            'fixed_min': True,
            DST_TYPE: INT8,
        }
        extra = (CONV1, 'Conv2d')
        obj.build(val.copy(), extra)
        self.assertEqual(obj.dump(), val)

    def test_distill_data_config_invalid_field(self):
        obj = distill_field.DistillDataConfig(GraphQuerier, CAPACITY)
        val = {'invalid_field': 1}
        extra = (CONV1, 'Conv2d')
        with self.assertRaises(ValueError):
            obj.build(val, extra)

    def test_distill_data_config_default(self):
        obj = distill_field.DistillDataConfig(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.dump(), {ALGO: 'ulq_quantize', DST_TYPE: INT8})

    def test_weight_algo_invalid(self):
        obj = distill_field.WeightAlgo(GraphQuerier, CAPACITY)
        val = 'invalid_algo'
        extra = (CONV1, 'Conv2d')
        with self.assertRaises(ValueError):
            obj.build(val, extra)

    def test_weight_algo_valid(self):
        obj = distill_field.WeightAlgo(GraphQuerier, CAPACITY)
        val = 'ulq_distill'
        extra = (CONV1, 'Conv2d')
        obj.build(val, extra)
        self.assertEqual(obj.value, val)

    def test_weight_algo_default(self):
        obj = distill_field.WeightAlgo(GraphQuerier, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.value, 'arq_distill')

    def test_distill_weight_config_success(self):
        obj = distill_field.DistillWeightConfig(GraphQuerier, CAPACITY)
        val = {ALGO: 'arq_distill', 'channel_wise': False}
        extra = (CONV1, 'Conv2d')
        obj.build(val, extra)
        self.assertEqual(
            obj.dump(), {ALGO: "arq_distill", "channel_wise": False, DST_TYPE: INT8}
        )

    def test_distill_weight_config_invalid(self):
        obj = distill_field.DistillWeightConfig(GraphQuerier, CAPACITY)
        val = {'invalid': 1}
        extra = (CONV1, 'Conv2d')
        with self.assertRaises(ValueError):
            obj.build(val, extra)

    def test_distill_weight_config_default(self):
        obj = distill_field.DistillWeightConfig(GraphQuerier, CAPACITY)
        extra = (CONV1, 'Conv2d')
        obj.build_default(extra)
        self.assertEqual(
            obj.dump(), {ALGO: "arq_distill", "channel_wise": True, DST_TYPE: INT8}
        )

    def test_layer_config_set_data_config(self):
        obj = distill_field.LayerConfig(GraphQuerier, CAPACITY)
        val = {
            QUANT_ENABLE: True,
            'distill_data_config': {ALGO: 'ulq_quantize', DST_TYPE: 'INT4'},
        }
        extra = (CONV1, 'Conv2d')
        obj.build(val, extra)
        except_val = {
            QUANT_ENABLE: True,
            'distill_data_config': {ALGO: 'ulq_quantize', DST_TYPE: 'INT4'},
            "distill_weight_config": {
                ALGO: "arq_distill",
                "channel_wise": True,
                DST_TYPE: INT8,
            },
        }
        self.assertEqual(obj.dump(), except_val)

    def test_layer_config_set_weight_config(self):
        obj = distill_field.LayerConfig(GraphQuerier, CAPACITY)
        val = {
            QUANT_ENABLE: True,
            "distill_weight_config": {
                ALGO: "arq_distill",
                "channel_wise": True,
                DST_TYPE: "INT4",
            },
        }
        extra = (CONV1, 'Conv2d')
        obj.build(val, extra)
        except_val = {
            QUANT_ENABLE: True,
            'distill_data_config': {ALGO: 'ulq_quantize', DST_TYPE: INT8},
            "distill_weight_config": {
                ALGO: "arq_distill",
                "channel_wise": True,
                DST_TYPE: "INT4",
            },
        }
        self.assertEqual(obj.dump(), except_val)

    def test_layer_config_default(self):
        obj = distill_field.LayerConfig(GraphQuerier, CAPACITY)
        extra = (CONV1, 'Conv2d')
        obj.build_default(extra)
        except_val = {
            QUANT_ENABLE: True,
            'distill_data_config': {ALGO: 'ulq_quantize', DST_TYPE: INT8},
            "distill_weight_config": {
                ALGO: "arq_distill",
                "channel_wise": True,
                DST_TYPE: INT8,
            },
        }
        self.assertEqual(obj.dump(), except_val)

    def test_distill_group(self):
        obj = distill_field.DistillGroup(GraphQuerier, CAPACITY)
        val = [['layer1', 'layer2'], ['layer3', 'layer4']]
        obj.build(val)
        self.assertEqual(obj.value, val)

    def test_distill_root_config_not_supported_layer(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        value = {
            "version": 1,
            "batch_num": 1,
            "group_size": 1,
            "data_dump": False,
            'distill_group': [['layer1', 'layer2']],
            "not_supported_layer": {},
        }
        extra = {CONV1: 'Conv2d'}
        with self.assertRaises(ValueError):
            obj.build(value, extra)

    def test_distill_root_config_no_enabled_layer(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        value = {
            "version": 1,
            "batch_num": 1,
            "group_size": 1,
            "data_dump": False,
            'distill_group': [['layer1', 'layer2']],
            CONV1: {QUANT_ENABLE: False},
        }
        extra = {CONV1: 'Conv2d'}
        with self.assertRaises(ValueError):
            obj.build(value, extra)

    def test_distill_root_config_layer_dst_type_not_equal(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        value = {
            "version": 1,
            "batch_num": 1,
            "group_size": 1,
            "data_dump": False,
            'distill_group': [['layer1', 'layer2']],
            CONV1: {
                QUANT_ENABLE: True,
                "distill_data_config": {ALGO: "ulq_quantize", DST_TYPE: INT8},
                "distill_weight_config": {
                    ALGO: "arq_distill",
                    "channel_wise": True,
                    DST_TYPE: "INT4",
                },
            },
        }
        extra = {CONV1: 'Conv2d'}
        with self.assertRaises(ValueError):
            obj.build(value, extra)

    def test_distill_root_config_success(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        value = {
            "version": 1,
            "batch_num": 2,
            "group_size": 2,
            "data_dump": True,
            'distill_group': [['layer1', 'layer2']],
            CONV1: {QUANT_ENABLE: True},
        }
        extra = {CONV1: 'Conv2d', 'conv2': 'Conv2d'}
        obj.build(value, extra)
        except_val = {
            "version": 1,
            "batch_num": 2,
            "group_size": 2,
            "data_dump": True,
            'distill_group': [['layer1', 'layer2']],
            CONV1: {
                QUANT_ENABLE: True,
                "distill_data_config": {ALGO: "ulq_quantize", DST_TYPE: INT8},
                "distill_weight_config": {
                    ALGO: "arq_distill",
                    "channel_wise": True,
                    DST_TYPE: INT8,
                },
            },
            "conv2": {
                QUANT_ENABLE: False,
                "distill_data_config": {ALGO: "ulq_quantize", DST_TYPE: INT8},
                "distill_weight_config": {
                    ALGO: "arq_distill",
                    "channel_wise": True,
                    DST_TYPE: INT8,
                },
            },
        }
        self.assertEqual(obj.dump(), except_val)

    def test_distill_root_config_default(self):
        obj = distill_field.DistillRootConfig(GraphQuerier, CAPACITY)
        groups = [['layer1', 'layer2']]
        extra = {CONV1: 'Conv2d'}
        obj.build_default(groups, extra)
        except_val = {
            "version": 1,
            "batch_num": 1,
            "group_size": 1,
            "data_dump": False,
            'distill_group': [['layer1', 'layer2']],
            CONV1: {
                QUANT_ENABLE: True,
                "distill_data_config": {ALGO: "ulq_quantize", DST_TYPE: INT8},
                "distill_weight_config": {
                    ALGO: "arq_distill",
                    "channel_wise": True,
                    DST_TYPE: INT8,
                },
            },
        }
        self.assertEqual(obj.dump(), except_val)
