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

from amct_pytorch.classic.graph_based.amct_pytorch.proto import (
    scale_offset_record_pytorch_pb2 as pb,
)
from amct_pytorch.classic.graph_based.amct_pytorch.common.utils import (
    record_file_operator as rfo,
)

logger = logging.getLogger(__name__)

REC = pb.ScaleOffsetRecord


class TestDstTypeGenerator(unittest.TestCase):
    def test_ok(self):
        from amct_pytorch.classic.graph_based.amct_pytorch.common.config.field import (
            WTS_SUPPORT_NUM_BITS,
        )
        bits = list(WTS_SUPPORT_NUM_BITS)[0]
        self.assertEqual(rfo.dst_type_generator(bits, WTS_SUPPORT_NUM_BITS),
                         'INT{}'.format(bits))

    def test_invalid(self):
        self.assertRaises(ValueError, rfo.dst_type_generator, 7, [4, 8])


class TestModuleFuncs(unittest.TestCase):
    def setUp(self):
        self.records = REC()

    def test_record_weights_scale_offset_add_and_update(self):
        rfo.record_weights_scale_offset(self.records, 'l1', [0.5], [0],
                                        num_bits=8, scale_r=[0.1], offset_r=[0])
        self.assertEqual(len(self.records.record), 1)
        self.assertEqual(self.records.record[0].value.wts_type, 'INT8')
        # update existing key
        rfo.record_weights_scale_offset(self.records, 'l1', [0.7], [0],
                                        num_bits=8, scale_r=[0.2], offset_r=[0])
        self.assertEqual(len(self.records.record), 1)
        self.assertAlmostEqual(self.records.record[0].value.scale_w[0], 0.7, places=5)

    def test_record_recurrence_weights_add_and_update(self):
        rfo.record_recurrence_weights_scale_offset(self.records, 'l1', [0.5], [0])
        rfo.record_recurrence_weights_scale_offset(self.records, 'l1', [0.6], [0])
        self.assertEqual(len(self.records.record), 1)
        rfo.record_recurrence_weights_scale_offset(self.records, 'l2', [0.6], [0])
        self.assertEqual(len(self.records.record), 2)

    def test_record_skip_status_add_and_update(self):
        rfo.record_skip_status(self.records, 'l1', True)
        self.assertTrue(self.records.record[0].value.skip_fusion)
        rfo.record_skip_status(self.records, 'l1', False)
        self.assertFalse(self.records.record[0].value.skip_fusion)

    def test_read_weights_scale_offset_ok(self):
        rfo.record_weights_scale_offset(self.records, 'l1', [0.5, 0.6], [0, 0])
        scale, offset = rfo.read_weights_scale_offset(self.records, 'l1')
        self.assertEqual(len(scale), 2)
        self.assertEqual(offset, [0, 0])

    def test_read_weights_scale_offset_missing_layer(self):
        self.assertRaises(RuntimeError, rfo.read_weights_scale_offset,
                          self.records, 'ghost')

    def test_read_weights_scale_offset_missing_scale(self):
        rec = self.records.record.add()
        rec.key = 'l1'  # no scale_w/offset_w set
        self.assertRaises(RuntimeError, rfo.read_weights_scale_offset,
                          self.records, 'l1')

    def test_record_and_read_shift_bits(self):
        rfo.record_shift_bits(self.records, 'l1', [2, 3])
        self.assertEqual(rfo.read_shift_bits(self.records, 'l1'), [2, 3])
        rfo.record_shift_bits(self.records, 'l1', [4])  # update
        self.assertEqual(rfo.read_shift_bits(self.records, 'l1'), [4])

    def test_read_shift_bits_missing(self):
        self.assertRaises(RuntimeError, rfo.read_shift_bits, self.records, 'ghost')

    def test_record_activation_scale_offset_add_update(self):
        rfo.record_activation_scale_offset(self.records, 'l1', 0.5, 0,
                                           num_bits=8, scale_h=0.1, offset_h=0)
        self.assertEqual(self.records.record[0].value.act_type, 'INT8')
        rfo.record_activation_scale_offset(self.records, 'l1', 0.9, 1,
                                           num_bits=8, scale_h=0.2, offset_h=1)
        self.assertAlmostEqual(self.records.record[0].value.scale_d, 0.9, places=5)

    def test_read_activation_scale_offset_ok(self):
        rfo.record_activation_scale_offset(self.records, 'l1', 0.5, 3)
        scale, offset = rfo.read_activation_scale_offset(self.records, 'l1')
        self.assertAlmostEqual(scale, 0.5, places=5)
        self.assertEqual(offset, 3)

    def test_read_activation_scale_offset_missing_layer(self):
        self.assertRaises(RuntimeError, rfo.read_activation_scale_offset,
                          self.records, 'ghost')

    def test_read_activation_scale_offset_missing_scale_d(self):
        rec = self.records.record.add()
        rec.key = 'l1'
        self.assertRaises(RuntimeError, rfo.read_activation_scale_offset,
                          self.records, 'l1')

    def test_record_dmq_balancer_factor_add_update(self):
        rfo.record_dmq_balancer_factor(self.records, 'l1', [1.0, 2.0])
        self.assertEqual(len(self.records.record[0].value.tensor_balance_factor), 2)
        rfo.record_dmq_balancer_factor(self.records, 'l1', [3.0])
        self.assertEqual(len(self.records.record[0].value.tensor_balance_factor), 1)

    def test_create_empty_record_ok_and_dup(self):
        rfo.create_empty_record(self.records, 'l1')
        self.assertEqual(self.records.record[0].key, 'l1')
        self.assertRaises(RuntimeError, rfo.create_empty_record, self.records, 'l1')

    def test_record_kv_cache_scale_offset_add_update(self):
        rfo.record_kv_cache_scale_offset(self.records, 'l1', [0.5], [0])
        self.assertEqual(len(self.records.record[0].kv_cache_value.scale), 1)
        rfo.record_kv_cache_scale_offset(self.records, 'l1', [0.6, 0.7], [0, 0])
        self.assertEqual(len(self.records.record[0].kv_cache_value.scale), 2)

    def test_record_quant_factors_add_update(self):
        rfo.record_quant_factors(self.records, 'l1', {'scale_w': [0.5], 'offset_w': [0]})
        self.assertEqual(len(self.records.record), 1)
        rfo.record_quant_factors(self.records, 'l1', {'scale_d': 0.3})
        self.assertEqual(len(self.records.record), 1)
        self.assertAlmostEqual(self.records.record[0].value.scale_d, 0.3, places=5)


if __name__ == '__main__':
    unittest.main()
