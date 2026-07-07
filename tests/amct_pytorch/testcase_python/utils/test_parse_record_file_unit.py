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

import numpy as np

from amct_pytorch.classic.graph_based.amct_pytorch.proto import (
    scale_offset_record_pytorch_pb2 as pb,
)
from amct_pytorch.classic.graph_based.amct_pytorch.common.utils import (
    parse_record_file as prf,
)

logger = logging.getLogger(__name__)

RM = prf.RecordManager
REC = pb.ScaleOffsetRecord


def _mgr(record=None, node_type='Conv2d', node_name='l1', capacity=None):
    if record is None:
        recs = REC()
        record = recs.record.add()
        record.key = node_name
    node = MagicMock()
    node.name = node_name
    node.type = node_type
    cap = capacity or {'SHIFT_N_TYPES': ['Conv2d'], 'NO_WEIGHT_QUANT_TYPES': []}
    return RM(record, node, cap, MagicMock())


class TestRecordManagerStatic(unittest.TestCase):
    def test_is_in_range_included(self):
        self.assertTrue(RM.is_in_range(np.array([1, 2, 3]), 0, 5))
        self.assertFalse(RM.is_in_range(np.array([1, 9]), 0, 5))

    def test_is_in_range_excluded(self):
        self.assertTrue(RM.is_in_range(np.array([1, 2]), 0, 5, included=False))
        self.assertFalse(RM.is_in_range(np.array([0, 2]), 0, 5, included=False))

    def test_get_range(self):
        self.assertEqual(RM.get_range('INT16'), [-32768, 32767])
        self.assertEqual(RM.get_range('INT8'), [-128, 127])
        self.assertEqual(RM.get_range('unknown'), [-128, 127])

    def test_check_cluster_center_ok(self):
        RM.check_cluster_center([1] * 16)

    def test_check_cluster_center_bad_len(self):
        self.assertRaises(ValueError, RM.check_cluster_center, [1, 2, 3])

    def test_check_cluster_center_out_of_range(self):
        self.assertRaises(ValueError, RM.check_cluster_center, [200] * 16)

    def test_check_layer_params_range_ok(self):
        params = {
            'act_type': 'INT8',
            'data_offset': np.array([0]),
            'weight_offset': np.array([0]),
            'data_scale': np.array([0.5]),
            'weight_scale': np.array([0.5]),
            'shift_n': np.array([0]),
        }
        RM.check_layer_params_range(params, 'l1')

    def test_check_layer_params_range_bad_offset_d(self):
        params = {
            'act_type': 'INT8',
            'data_offset': np.array([999]),
            'weight_offset': np.array([0]),
            'data_scale': np.array([0.5]),
            'weight_scale': np.array([0.5]),
            'shift_n': np.array([0]),
        }
        self.assertRaises(ValueError, RM.check_layer_params_range, params, 'l1')

    def test_check_layer_params_range_bad_scale_w(self):
        params = {
            'act_type': 'INT8',
            'data_offset': np.array([0]),
            'weight_offset': np.array([0]),
            'data_scale': np.array([0.5]),
            'weight_scale': np.array([0.0]),
            'shift_n': np.array([0]),
        }
        self.assertRaises(ValueError, RM.check_layer_params_range, params, 'l1')

    def test_check_layer_params_range_bad_shift_n(self):
        params = {
            'act_type': 'INT8',
            'data_offset': np.array([0]),
            'weight_offset': np.array([0]),
            'data_scale': np.array([0.5]),
            'weight_scale': np.array([0.5]),
            'shift_n': np.array([99]),
        }
        self.assertRaises(ValueError, RM.check_layer_params_range, params, 'l1')

    def test_check_rnn_layer_params_range_ok(self):
        params = {
            'act_type': 'INT8',
            'h_offset': np.array([0]),
            'recurrence_weight_offset': np.array([0]),
            'h_scale': np.array([0.5]),
            'recurrence_weight_scale': np.array([0.5]),
        }
        RM.check_rnn_layer_params_range(params, 'l1')

    def test_check_rnn_layer_params_range_bad_offset_h(self):
        params = {
            'act_type': 'INT8',
            'h_offset': np.array([999]),
            'recurrence_weight_offset': np.array([0]),
            'h_scale': np.array([0.5]),
            'recurrence_weight_scale': np.array([0.5]),
        }
        self.assertRaises(ValueError, RM.check_rnn_layer_params_range, params, 'l1')


class TestRecordManagerGetters(unittest.TestCase):
    def test_get_key_value_skip(self):
        recs = REC()
        r = recs.record.add()
        r.key = 'l1'
        r.value.skip_fusion = True
        mgr = _mgr(record=r)
        self.assertEqual(mgr.get_key(), 'l1')
        self.assertTrue(mgr.get_skip_fusion())
        self.assertIsNotNone(mgr.get_value())

    def test_check_quant_value_empty_true(self):
        mgr = _mgr()
        self.assertTrue(mgr.check_quant_value_empty())

    def test_check_quant_value_empty_false(self):
        recs = REC()
        r = recs.record.add()
        r.key = 'l1'
        r.value.scale_d = 0.5
        self.assertFalse(_mgr(record=r).check_quant_value_empty())

    def test_get_dst_type_unset(self):
        self.assertEqual(_mgr().get_dst_type(), 'UNSET')

    def test_get_dst_type_int8(self):
        recs = REC()
        r = recs.record.add()
        r.key = 'l1'
        r.value.dst_type = 'INT8'
        self.assertEqual(_mgr(record=r).get_dst_type(), 'INT8')

    def test_get_dst_type_invalid(self):
        recs = REC()
        r = recs.record.add()
        r.key = 'l1'
        r.value.dst_type = 'INT99'
        self.assertRaises(RuntimeError, _mgr(record=r).get_dst_type)

    def test_get_dst_type_avgpool_int4(self):
        recs = REC()
        r = recs.record.add()
        r.key = 'AvgPool_1'
        r.value.dst_type = 'INT4'
        self.assertRaises(RuntimeError, _mgr(record=r, node_name='AvgPool_1').get_dst_type)

    def test_get_act_type_unset(self):
        self.assertEqual(_mgr().get_act_type(), 'UNSET')

    def test_get_act_type_invalid(self):
        recs = REC()
        r = recs.record.add()
        r.key = 'l1'
        r.value.act_type = 'INT99'
        self.assertRaises(RuntimeError, _mgr(record=r).get_act_type)

    def test_get_wts_type_unset(self):
        self.assertEqual(_mgr().get_wts_type(), 'UNSET')

    def test_get_wts_type_invalid(self):
        recs = REC()
        r = recs.record.add()
        r.key = 'l1'
        r.value.wts_type = 'INT99'
        self.assertRaises(RuntimeError, _mgr(record=r).get_wts_type)

    def test_get_op_dtype_absent_field_noop(self):
        # current proto has no op_data_type field; get_op_dtype guards with
        # hasattr and is a no-op, leaving layer_params untouched.
        params = {}
        _mgr().get_op_dtype(params)
        self.assertEqual(params, {})

    def test_get_act_quant_factor_missing(self):
        self.assertRaises(RuntimeError, _mgr().get_act_quant_factor, 'scale_d', np.float32)

    def test_get_wts_quant_factor_desirable_missing(self):
        self.assertRaises(RuntimeError, _mgr().get_wts_quant_factor, 'scale_w', np.float32)

    def test_get_wts_quant_factor_not_desirable_default(self):
        arr = _mgr().get_wts_quant_factor('scale_w', np.float32, desirable=False)
        self.assertEqual(list(arr), [1.0])

    def test_get_shift_n_missing(self):
        self.assertRaises(RuntimeError, _mgr().get_shift_n, True)

    def test_get_shift_n_not_desirable_empty(self):
        arr = _mgr().get_shift_n(desirable=False)
        self.assertEqual(arr.size, 0)

    def test_get_cluster_center_none(self):
        self.assertIsNone(_mgr().get_cluster_center())

    def test_get_tensor_balance_factor_none(self):
        self.assertIsNone(_mgr().get_tensor_balance_factor())

    def test_get_fakequant_precision_mode_default(self):
        self.assertEqual(_mgr().get_fakequant_precision_mode(), 'DEFAULT')


if __name__ == '__main__':
    unittest.main()
