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
"""Unit tests of the shift value record handling in Recorder."""

import os
import unittest

import pytest

from google.protobuf import text_format

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.recorder.recorder import (
    Recorder,
)
from amct_pytorch.classic.graph_based.amct_pytorch.proto import (
    scale_offset_record_pb2,
)
from amct_pytorch.classic.graph_based.amct_pytorch.common.utils.record_file_operator import (
    record_fixed_shift_value,
)
from amct_pytorch.classic.graph_based.amct_pytorch.utils.singleton_record import (
    SingletonScaleOffsetRecord,
)

pytestmark = pytest.mark.cpu


CONV_SHIFT_ONLY = (
    'record {\n  key: "conv"\n  value {\n    fixed_shift_value: 42\n  }\n}\n'
)
FC_WITH_SCALE_W = (
    'record {\n  key: "fc"\n  value {\n    scale_w: 0.5\n    offset_w: 0\n  }\n}\n'
)


def _read_file_records(path):
    """Parse the record file into a proto message."""
    records = scale_offset_record_pb2.ScaleOffsetRecord()
    with open(path, 'r') as fid:
        text_format.Merge(fid.read(), records)
    return records


def _find_record(records, layer_name):
    """Return the first record entry matching layer_name."""
    for item in records.record:
        if item.key == layer_name:
            return item
    return None


class TestRecorderShiftValue(unittest.TestCase):
    """Shift value behaviors of the record read / write path."""

    def setUp(self):
        """Create a temp record file per test."""
        self.record_file = os.path.join(
            '/tmp', 'test_recorder_shift_value_%s.txt' % self.id().rsplit('.', 1)[-1]
        )

    def tearDown(self):
        """Remove the temp record file."""
        if os.path.exists(self.record_file):
            os.remove(self.record_file)

    def test_shift_value_writer_refresh_no_duplicates(self):
        """Regression of the Clear() fix: the singleton refreshed by the shift
        value writer already carries the file content, reading the file again
        must not append duplicated entries."""
        with open(self.record_file, 'w') as fid:
            fid.write(CONV_SHIFT_ONLY)

        recorder = Recorder(self.record_file)
        recorder.record_quant_layer(['conv'])
        # the ShiftCalibrationModule writer refreshes the shared singleton
        # with the file content, the recorder shares the same object
        refreshed = scale_offset_record_pb2.ScaleOffsetRecord()
        text_format.Merge(CONV_SHIFT_ONLY, refreshed)
        SingletonScaleOffsetRecord().record.CopyFrom(refreshed)

        # the activation calibration arrives for the weight-only layer
        recorder(['conv'], 'ifmr', {'scale_d': 0.1, 'offset_d': 0})

        records = _read_file_records(self.record_file)
        keys = [item.key for item in records.record]
        self.assertEqual(keys, ['conv'])
        value = records.record[0].value
        self.assertEqual(value.fixed_shift_value, 42)
        self.assertAlmostEqual(value.scale_d, 0.1)

    def test_layers_num_excludes_shift_only_records(self):
        """records carrying only fixed_shift_value do not take part in the
        activation calibration accounting: the write of the quant factors
        still triggers when every quantized layer is calibrated."""
        with open(self.record_file, 'w') as fid:
            fid.write(CONV_SHIFT_ONLY + FC_WITH_SCALE_W)

        # no record_quant_layer call: the layer number falls back to the
        # file content count, shift-only entries are excluded
        recorder = Recorder(self.record_file)
        recorder(['fc'], 'ifmr', {'scale_d': 0.2, 'offset_d': 1})

        records = _read_file_records(self.record_file)
        fc_record = _find_record(records, 'fc')
        self.assertIsNotNone(fc_record)
        self.assertAlmostEqual(fc_record.value.scale_d, 0.2)
        conv_record = _find_record(records, 'conv')
        self.assertIsNotNone(conv_record)
        self.assertFalse(conv_record.value.HasField('scale_d'))


class TestRecordFixedShiftValueUpsert(unittest.TestCase):
    """record_fixed_shift_value only touches the target field."""

    def test_upsert_existing_entry_keeps_other_fields(self):
        records = scale_offset_record_pb2.ScaleOffsetRecord()
        text_format.Merge(FC_WITH_SCALE_W, records)

        record_fixed_shift_value(records, 'fc', 42)

        self.assertEqual(len(records.record), 1)
        value = records.record[0].value
        self.assertEqual(value.fixed_shift_value, 42)
        self.assertAlmostEqual(list(value.scale_w)[0], 0.5)

    def test_append_new_entry(self):
        records = scale_offset_record_pb2.ScaleOffsetRecord()

        record_fixed_shift_value(records, 'conv', 13)

        self.assertEqual(len(records.record), 1)
        self.assertEqual(records.record[0].key, 'conv')
        self.assertEqual(records.record[0].value.fixed_shift_value, 13)
