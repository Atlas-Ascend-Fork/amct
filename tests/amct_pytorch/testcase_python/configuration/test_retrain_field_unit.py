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
import unittest
from unittest.mock import MagicMock

from amct_pytorch.classic.graph_based.amct_pytorch.common.retrain_config import (
    retrain_field as rf,
)

logger = logging.getLogger(__name__)

LAYER = 'layerA'


def _item(cls):
    return cls(MagicMock(), MagicMock())


class TestSimpleFields(unittest.TestCase):
    def test_version(self):
        item = _item(rf.Version)
        self.assertRaises(TypeError, item.build, '1')
        self.assertRaises(ValueError, item.build, 2)
        item.build(1)
        self.assertEqual(item.value, 1)
        item.build_default()
        self.assertEqual(item.value, 1)

    def test_batch_num(self):
        item = _item(rf.BatchNum)
        self.assertRaises(TypeError, item.build, '4')
        self.assertRaises(ValueError, item.build, 0)
        item.build(4)
        self.assertEqual(item.value, 4)
        item.build_default()
        self.assertEqual(item.value, 1)

    def test_fakequant_precision_mode(self):
        item = _item(rf.FakequantPrecisionMode)
        self.assertRaises(TypeError, item.build, 1)
        self.assertRaises(ValueError, item.build, 'BAD')
        item.build('DEFAULT')
        self.assertEqual(item.value, 'DEFAULT')
        item.build_default()
        self.assertEqual(item.value, 'DEFAULT')

    def test_retrain_enable(self):
        item = _item(rf.RetrainEnable)
        self.assertRaises(TypeError, item.build, 'no', [LAYER])
        item.build(True, [LAYER])
        self.assertTrue(item.value)
        item.build_default()
        self.assertTrue(item.value)

    def test_data_algo(self):
        item = _item(rf.DataAlgo)
        self.assertRaises(TypeError, item.build, 1, [LAYER])
        self.assertRaises(ValueError, item.build, 'bad', [LAYER])
        item.build('ulq_quantize', [LAYER])
        self.assertEqual(item.value, 'ulq_quantize')
        item.build_default()
        self.assertEqual(item.value, 'ulq_quantize')

    def test_clip_max(self):
        item = _item(rf.ClipMax)
        self.assertRaises(TypeError, item.build, 1, [LAYER])
        self.assertRaises(ValueError, item.build, -1.0, [LAYER])
        item.build(1.0, [LAYER])
        self.assertEqual(item.value, 1.0)

    def test_clip_min(self):
        item = _item(rf.ClipMin)
        self.assertRaises(TypeError, item.build, 1, [LAYER])
        self.assertRaises(ValueError, item.build, 1.0, [LAYER])
        item.build(-1.0, [LAYER])
        self.assertEqual(item.value, -1.0)

    def test_fixed_min(self):
        item = _item(rf.FixedMin)
        self.assertRaises(TypeError, item.build, 'no', [LAYER])
        item.build(True, [LAYER])
        self.assertTrue(item.value)

    def test_data_type(self):
        item = _item(rf.DataType)
        self.assertRaises(TypeError, item.build, 1, [LAYER])
        self.assertRaises(ValueError, item.build, 'INT3', [LAYER])
        item.build('INT8', [LAYER])
        self.assertEqual(item.value, 'INT8')
        item.build_default()
        self.assertEqual(item.value, 'INT8')

    def test_weight_algo(self):
        item = _item(rf.WeightAlgo)
        self.assertRaises(ValueError, item.build, 'bad', [LAYER])
        item.build('arq_retrain', [LAYER])
        self.assertEqual(item.value, 'arq_retrain')
        item.build_default()
        self.assertEqual(item.value, 'arq_retrain')

    def test_channel_wise(self):
        item = _item(rf.ChannelWise)
        self.assertRaises(ValueError, item.build, True, [LAYER, 'Linear'])
        item.build(True, [LAYER, 'Conv2d'])
        self.assertTrue(item.value)
        item.build_default([LAYER, 'Linear'])
        self.assertFalse(item.value)
        item.build_default([LAYER, 'Conv2d'])
        self.assertTrue(item.value)

    def test_regular_prune_enable(self):
        item = _item(rf.RegularPruneEnable)
        self.assertRaises(TypeError, item.build, 'no', [LAYER])
        item.build(False, [LAYER])
        self.assertFalse(item.value)
        item.build_default()
        self.assertTrue(item.value)

    def test_prune_ratio(self):
        item = _item(rf.PruneRatio)
        self.assertRaises(TypeError, item.build, 1, [LAYER])
        self.assertRaises(ValueError, item.build, 0.0, [LAYER])
        self.assertRaises(ValueError, item.build, 1.0, [LAYER])
        item.build(0.5, [LAYER])
        self.assertEqual(item.value, 0.5)

    def test_ascend_optimized(self):
        item = _item(rf.AscendOptimized)
        self.assertRaises(TypeError, item.build, 'no', [LAYER])
        item.build(True, [LAYER])
        self.assertTrue(item.value)
        item.build_default()
        self.assertTrue(item.value)

    def test_n_out_of_m_type(self):
        item = _item(rf.NOutOfMType)
        self.assertRaises(ValueError, item.build, 'M8N4', [LAYER])
        item.build('M4N2', [LAYER])
        self.assertEqual(item.value, 'M4N2')

    def test_update_freq(self):
        item = _item(rf.UpdateFreq)
        self.assertRaises(TypeError, item.build, '1', [LAYER])
        self.assertRaises(ValueError, item.build, -1, [LAYER])
        item.build(10, [LAYER])
        self.assertEqual(item.value, 10)


class TestContainerFields(unittest.TestCase):
    def test_retrain_data_config_full(self):
        item = _item(rf.RetrainDataConfig)
        val = {'algo': 'ulq_quantize', 'clip_max': 1.0, 'clip_min': -1.0,
               'fixed_min': True, 'dst_type': 'INT8'}
        item.build(val, [LAYER])
        dumped = item.dump()
        self.assertEqual(dumped.get('algo'), 'ulq_quantize')
        self.assertEqual(dumped.get('dst_type'), 'INT8')

    def test_retrain_data_config_default(self):
        item = _item(rf.RetrainDataConfig)
        item.build_default()
        dumped = item.dump()
        self.assertEqual(dumped.get('algo'), 'ulq_quantize')

    def test_retrain_data_config_invalid_key(self):
        item = _item(rf.RetrainDataConfig)
        self.assertRaises(ValueError, item.build, {'unknown': 1}, [LAYER])

    def test_retrain_weight_config_arq(self):
        item = _item(rf.RetrainWeightConfig)
        item.build({'algo': 'arq_retrain', 'channel_wise': True,
                    'dst_type': 'INT8'}, [LAYER, 'Conv2d'])
        self.assertEqual(item.dump().get('algo'), 'arq_retrain')

    def test_retrain_weight_config_default(self):
        item = _item(rf.RetrainWeightConfig)
        item.build_default([LAYER, 'Conv2d'])
        self.assertEqual(item.dump().get('algo'), 'arq_retrain')

    def test_retrain_weight_config_invalid_algo(self):
        item = _item(rf.RetrainWeightConfig)
        self.assertRaises(ValueError, item.build,
                          {'algo': 'bad_algo'}, [LAYER, 'Conv2d'])

    def test_retrain_weight_config_invalid_key(self):
        item = _item(rf.RetrainWeightConfig)
        self.assertRaises(ValueError, item.build,
                          {'algo': 'arq_retrain', 'bad': 1}, [LAYER, 'Conv2d'])


if __name__ == '__main__':
    unittest.main()
