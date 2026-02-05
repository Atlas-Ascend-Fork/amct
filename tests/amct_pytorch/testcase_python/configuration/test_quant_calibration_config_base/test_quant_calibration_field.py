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
import unittest

from amct_pytorch.graph_based_compression.amct_pytorch.capacity import CAPACITY
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.quant_calibration_config_base import quant_calibration_field as field
from amct_pytorch.graph_based_compression.amct_pytorch.utils.model_util import ModuleHelper



class TestQuantCalibrationField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestQuantCalibrationField start!')

    @classmethod
    def tearDownClass(cls):
        print('TestQuantCalibrationField end!')

    def test_act_algo_build(self):
        obj = field.ActAlgo(ModuleHelper, CAPACITY)
        obj.build('ifmr', ['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 'ifmr')

        obj.build('hfmg', ['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 'hfmg')

    def test_act_algo_default(self):
        obj = field.ActAlgo(ModuleHelper, CAPACITY)
        obj.build_default(['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 'ifmr')

    def test_act_algo_build_error(self):
        obj = field.ActAlgo(ModuleHelper, CAPACITY)
        self.assertRaises(TypeError, obj.build, 1, ['matmul_1', 'Linear'])
        self.assertRaises(ValueError, obj.build, 'qat', ['matmul_1', 'Linear'])

    def test_max_percentile_build(self):
        obj = field.MaxPercentile(ModuleHelper, CAPACITY)
        obj.build(0.8, ['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 0.8)

    def test_max_percentile_default(self):
        obj = field.MaxPercentile(ModuleHelper, CAPACITY)
        obj.build_default(['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 0.999999)

    def test_max_percentile_build_error(self):
        obj = field.MaxPercentile(ModuleHelper, CAPACITY)
        self.assertRaises(TypeError, obj.build, 'abc', ['matmul_1', 'Linear'])
        self.assertRaises(ValueError, obj.build, 1.2, ['matmul_1', 'Linear'])

    def test_min_percentile_build(self):
        obj = field.MinPercentile(ModuleHelper, CAPACITY)
        obj.build(0.8, ['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 0.8)

    def test_min_percentile_default(self):
        obj = field.MinPercentile(ModuleHelper, CAPACITY)
        obj.build_default(['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 0.999999)

    def test_min_percentile_build_error(self):
        obj = field.MinPercentile(ModuleHelper, CAPACITY)
        self.assertRaises(TypeError, obj.build, 'abc', ['matmul_1', 'Linear'])

    def test_search_range_build(self):
        obj = field.SearchRange(ModuleHelper, CAPACITY)
        obj.build([0.8, 1.2], ['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), [0.8, 1.2])

    def test_search_range_default(self):
        obj = field.SearchRange(ModuleHelper, CAPACITY)
        obj.build_default(['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), [0.7, 1.3])

    def test_search_range_build_error(self):
        obj = field.SearchRange(ModuleHelper, CAPACITY)
        self.assertRaises(TypeError, obj.build, 'abc', ['matmul_1', 'Linear'])
        self.assertRaises(ValueError, obj.build, [1.2], ['matmul_1', 'Linear'])
        self.assertRaises(ValueError, obj.build, [1.2, 0.8], ['matmul_1', 'Linear'])

    def test_search_step_build(self):
        obj = field.SearchStep(ModuleHelper, CAPACITY)
        obj.build(0.1, ['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 0.1)

    def test_search_step_default(self):
        obj = field.SearchStep(ModuleHelper, CAPACITY)
        obj.build_default(['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 0.01)

    def test_search_step_build_error(self):
        obj = field.SearchStep(ModuleHelper, CAPACITY)
        self.assertRaises(TypeError, obj.build, 'abc', ['matmul_1', 'Linear'])
        self.assertRaises(ValueError, obj.build, 0., ['matmul_1', 'Linear'])

    def test_asymmetric_build(self):
        obj = field.Asymmetric(ModuleHelper, CAPACITY)
        obj.build(False, ['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), False)

    def test_asymmetric_default(self):
        obj = field.Asymmetric(ModuleHelper, CAPACITY)
        obj.build_default(['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), None)

    def test_asymmetric_build_error(self):
        obj = field.Asymmetric(ModuleHelper, CAPACITY)
        self.assertRaises(TypeError, obj.build, 'abc', ['matmul_1', 'Linear'])

    def test_num_of_bins_build(self):
        obj = field.NumOfBins(ModuleHelper, CAPACITY)
        obj.build(8192, ['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 8192)

    def test_num_of_bins_default(self):
        obj = field.NumOfBins(ModuleHelper, CAPACITY)
        obj.build_default(['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 4096)

    def test_num_of_bins_build_error(self):
        obj = field.NumOfBins(ModuleHelper, CAPACITY)
        self.assertRaises(TypeError, obj.build, 'abc', ['matmul_1', 'Linear'])
        self.assertRaises(ValueError, obj.build, 2047, ['matmul_1', 'Linear'])

    def test_activation_offset_build(self):
        obj = field.ActivationOffset(ModuleHelper, CAPACITY)
        obj.build(False)
        self.assertEqual(obj.dump(), False)

    def test_activation_offset_default(self):
        obj = field.ActivationOffset(ModuleHelper, CAPACITY)
        obj.build_default()
        self.assertEqual(obj.dump(), True)

    def test_activation_offset_build_error(self):
        obj = field.ActivationOffset(ModuleHelper, CAPACITY)
        self.assertRaises(TypeError, obj.build, 'abc')

    def test_act_granularity_build(self):
        obj = field.ActQuantGranularity(ModuleHelper, CAPACITY)
        obj.build(1, ['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 1)

    def test_act_granularity_default(self):
        obj = field.ActQuantGranularity(ModuleHelper, CAPACITY)
        obj.build_default(['matmul_1', 'Linear'])
        self.assertEqual(obj.dump(), 0)

        obj.build_default(['matmul_1', 'Conv'])
        self.assertEqual(obj.dump(), 0)

    def test_act_granularity_build_error(self):
        obj = field.ActQuantGranularity(ModuleHelper, CAPACITY)
        self.assertRaises(TypeError, obj.build, 'abc', ['matmul_1', 'Linear'])
        self.assertRaises(ValueError, obj.build, 1, ['matmul_1', 'Conv'])

    def test_kv_cache_layer_config_build_default(self):
        obj = field.KVCacheLayerConfig(ModuleHelper, CAPACITY)
        obj.build_default(['matmul_1', 'Linear'])
        ret = obj.dump()
        self.assertEqual(ret.get('kv_data_quant_config').get('act_algo'), 'ifmr')

    def test_kv_cache_layer_config_build(self):
        obj = field.KVCacheLayerConfig(ModuleHelper, CAPACITY)
        obj.build({}, ['matmul_1', 'Linear'])
        ret = obj.dump()
        self.assertEqual(ret.get('kv_data_quant_config').get('act_algo'), 'ifmr')

        obj.build({'kv_data_quant_config': {'act_algo': 'hfmg'}}, ['matmul_1', 'Linear'])
        ret = obj.dump()
        self.assertEqual(ret.get('kv_data_quant_config').get('act_algo'), 'hfmg')

    def test_kv_cache_root_config_build_default(self):
        obj = field.KVCacheRootConfig(ModuleHelper, CAPACITY)
        obj.build_default({'kv_cache_quant_layers': {'matmul_1': 'Linear'}})
        ret = obj.dump()
        self.assertIn('kv_data_quant_config', ret.get('matmul_1'))

    def test_kv_cache_root_config_build(self):
        obj = field.KVCacheRootConfig(ModuleHelper, CAPACITY)
        obj.build({'matmul_1': {'kv_data_quant_config': {'act_algo': 'hfmg'}}},
                  {'kv_cache_quant_layers': {'matmul_1': 'Linear'}})
        ret = obj.dump()

        self.assertIn('kv_data_quant_config', ret.get('matmul_1'))
        self.assertEqual(ret.get('matmul_1').get('kv_data_quant_config').get('act_algo'), 'hfmg')

    def test_calibration_general_config_item_build_default(self):
        obj = field.CalibrationGeneralConfigItem(ModuleHelper, CAPACITY)
        obj.build_default({'kv_cache_quant_layers': {'matmul_1': 'Linear'}})
        ret = obj.dump()
        self.assertEqual(ret.get('batch_num'), 1)
        self.assertEqual(ret.get('activation_offset'), True)

    def test_calibration_general_config_item_build(self):
        obj = field.CalibrationGeneralConfigItem(ModuleHelper, CAPACITY)
        obj.build({'batch_num': 8, 'activation_offset': False},
                  {'kv_cache_quant_layers': {'matmul_1': 'Linear'}})
        ret = obj.dump()
        self.assertEqual(ret.get('batch_num'), 8)
        self.assertEqual(ret.get('activation_offset'), False)


    def test_quant_calibration_config_root_build_default(self):
        obj = field.QuantCalibrationConfigRoot(ModuleHelper, CAPACITY)
        obj.build_default({'kv_cache_quant_layers': {'matmul_1': 'Linear'}})
        ret = obj.dump()
        self.assertEqual(ret.get('batch_num'), 1)
        self.assertEqual(ret.get('activation_offset'), True)
        self.assertIn('matmul_1', ret)

    def test_quant_calibration_config_root_build(self):
        obj = field.QuantCalibrationConfigRoot(ModuleHelper, CAPACITY)
        obj.build({'batch_num': 8, 'activation_offset': False, 'matmul_1': {'kv_data_quant_config': {'act_algo': 'hfmg'}}},
                  {'kv_cache_quant_layers': {'matmul_1': 'Linear'}})
        ret = obj.dump()
        self.assertEqual(ret.get('batch_num'), 8)
        self.assertEqual(ret.get('activation_offset'), False)
        self.assertIn('matmul_1', ret)
