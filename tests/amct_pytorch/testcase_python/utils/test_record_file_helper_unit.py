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
import shutil
import tempfile
import unittest

from amct_pytorch.classic.graph_based.amct_pytorch.proto import (
    scale_offset_record_pytorch_pb2 as pb,
)
from amct_pytorch.classic.graph_based.amct_pytorch.common.utils import (
    record_file_operator as rfo,
)

logger = logging.getLogger(__name__)

REC = pb.ScaleOffsetRecord


class TestScaleOffsetRecordHelper(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='amct_test_rfo_')
        self.helper = rfo.ScaleOffsetRecordHelper(REC)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_keys_and_records(self):
        self.helper.record_weights_scale_offset('l1', [0.5], [0])
        self.helper.record_weights_scale_offset('l2', [0.6], [0])
        self.assertEqual(self.helper.keys, ['l1', 'l2'])
        self.assertIsNotNone(self.helper.records)

    def test_record_weights_with_dst_type_and_update(self):
        self.helper.record_weights_scale_offset('l1', [0.5], [0], dst_type='INT8')
        rec = self.helper.get_record('l1')
        self.assertEqual(rec.wts_type, 'INT8')
        self.helper.record_weights_scale_offset('l1', [0.7], [0], dst_type='INT4')
        self.assertEqual(self.helper.get_record('l1').wts_type, 'INT4')

    def test_check_record_ok(self):
        self.helper.record_weights_scale_offset('l1', [0.5], [0])
        rec = self.helper.get_record('l1')
        self.assertTrue(rfo.ScaleOffsetRecordHelper.check_record('l1', rec))

    def test_check_record_length_mismatch(self):
        self.helper.record_weights_scale_offset('l1', [0.5, 0.6], [0])
        rec = self.helper.get_record('l1')
        self.assertRaises(RuntimeError,
                          rfo.ScaleOffsetRecordHelper.check_record, 'l1', rec)

    def test_check_record_illegal_scale(self):
        self.helper.record_activation_scale_offset('l1', 0.0, 0)
        rec = self.helper.get_record('l1')
        self.assertRaises(ValueError,
                          rfo.ScaleOffsetRecordHelper.check_record, 'l1', rec)

    def test_check_record_illegal_offset_w(self):
        self.helper.record_weights_scale_offset('l1', [0.5], [3])
        rec = self.helper.get_record('l1')
        self.assertRaises(ValueError,
                          rfo.ScaleOffsetRecordHelper.check_record, 'l1', rec)

    def test_check_record_illegal_offset_d(self):
        self.helper.record_activation_scale_offset('l1', 0.5, 9999)
        rec = self.helper.get_record('l1')
        self.assertRaises(ValueError,
                          rfo.ScaleOffsetRecordHelper.check_record, 'l1', rec)

    def test_init_from_file_and_update_and_dump(self):
        # build a record, dump to file, then re-init from it
        self.helper.record_weights_scale_offset('l1', [0.5], [0])
        path = os.path.join(self.tmp_dir, 'rec.txt')
        self.helper.dump(path)
        self.assertTrue(os.path.isfile(path))

        helper2 = rfo.ScaleOffsetRecordHelper(REC)
        helper2.init_from_file(path)
        self.assertIn('l1', helper2.keys)
        # update_record writes back to the loaded file
        helper2.record_weights_scale_offset('l2', [0.6], [0])
        helper2.update_record()
        helper3 = rfo.ScaleOffsetRecordHelper(REC)
        helper3.init_from_file(path)
        self.assertEqual(set(helper3.keys), {'l1', 'l2'})

    def test_init_from_file_parse_error(self):
        path = os.path.join(self.tmp_dir, 'bad.txt')
        with open(path, 'w') as fid:
            fid.write('not a valid proto text @@@')
        self.assertRaises(RuntimeError, self.helper.init_from_file, path)

    def test_update_record_without_file(self):
        self.assertRaises(RuntimeError, self.helper.update_record)

    def test_init_and_merge(self):
        self.helper.record_weights_scale_offset('l1', [0.5], [0])
        other = REC()
        rfo.record_weights_scale_offset(other, 'l2', [0.6], [0])
        self.helper.merge(other)
        self.assertEqual(set(self.helper.keys), {'l1', 'l2'})

    def test_merge_conflict(self):
        self.helper.record_weights_scale_offset('l1', [0.5], [0])
        other = REC()
        rfo.record_weights_scale_offset(other, 'l1', [0.6], [0])
        self.assertRaises(RuntimeError, self.helper.merge, other)

    def test_init_replaces_records(self):
        new_records = REC()
        rfo.record_weights_scale_offset(new_records, 'x', [0.5], [0])
        self.helper.init(new_records)
        self.assertEqual(self.helper.keys, ['x'])

    def test_has_key_and_delete_key(self):
        self.helper.record_weights_scale_offset('l1', [0.5], [0])
        self.assertTrue(self.helper.has_key('l1'))
        self.assertFalse(self.helper.has_key('ghost'))
        self.assertTrue(self.helper.delete_key('l1'))
        self.assertFalse(self.helper.delete_key('l1'))

    def test_get_record_missing(self):
        self.assertIsNone(self.helper.get_record('ghost'))

    def test_recurrence_and_skip_status(self):
        self.helper.record_recurrence_weights_scale_offset('l1', [0.5], [0])
        self.helper.record_skip_status('l1', True)
        self.assertTrue(self.helper.get_record('l1').skip_fusion)
        self.helper.record_skip_status('l1', False)  # update existing
        self.assertFalse(self.helper.get_record('l1').skip_fusion)
        self.helper.record_skip_status('l2', True)   # add new

    def test_read_weights_scale_offset(self):
        self.helper.record_weights_scale_offset('l1', [0.5, 0.6], [0, 0])
        scale, offset = self.helper.read_weights_scale_offset('l1')
        self.assertEqual(len(scale), 2)
        self.assertRaises(RuntimeError, self.helper.read_weights_scale_offset, 'ghost')

    def test_read_weights_missing_scale(self):
        self.helper.record_skip_status('l1', True)  # creates record without scale_w
        self.assertRaises(RuntimeError, self.helper.read_weights_scale_offset, 'l1')

    def test_record_and_read_shift_bits(self):
        self.helper.record_shift_bits('l1', [2, 3])
        self.assertEqual(self.helper.read_shift_bits('l1'), [2, 3])
        self.helper.record_shift_bits('l1', [4])  # update
        self.assertEqual(self.helper.read_shift_bits('l1'), [4])
        self.assertRaises(RuntimeError, self.helper.read_shift_bits, 'ghost')

    def test_record_activation_variants(self):
        self.helper.record_activation_scale_offset('l1', 0.5, 0, dst_type='INT8')
        self.assertEqual(self.helper.get_record('l1').act_type, 'INT8')
        self.helper.record_activation_scale_offset('l1', 0.9, 1, dst_type='INT16')  # update
        self.helper.record_activation_h_scale_offset('l1', 0.3, 0)
        self.helper.record_activation_h_scale_offset('l2', 0.4, 0)  # add new

    def test_read_activation_scale_offset(self):
        self.helper.record_activation_scale_offset('l1', 0.5, 3)
        scale, offset = self.helper.read_activation_scale_offset('l1')
        self.assertEqual(offset, 3)
        self.assertRaises(RuntimeError, self.helper.read_activation_scale_offset, 'ghost')

    def test_read_activation_missing_scale_d(self):
        self.helper.record_skip_status('l1', True)
        self.assertRaises(RuntimeError, self.helper.read_activation_scale_offset, 'l1')


if __name__ == '__main__':
    unittest.main()
