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
"""Unit tests of _split_shift_value_records in quantize_tool."""

import unittest

import pytest

from amct_pytorch.classic.graph_based.amct_pytorch.quantize_tool import (
    _split_shift_value_records,
)

pytestmark = pytest.mark.cpu


class TestSplitShiftValueRecords(unittest.TestCase):
    """Records are split into the quant view and the shift value view."""

    def test_off_identity(self):
        """no fixed_shift_value in the records: quant view equals the input."""
        records = {
            'conv': {'scale_w': 0.5, 'offset_w': 0},
            'fc': {'scale_d': 0.1, 'offset_d': 1},
        }
        quant_records, shift_records = _split_shift_value_records(records)
        self.assertEqual(quant_records, records)
        self.assertEqual(shift_records, {})

    def test_mixed_record_goes_to_both_views(self):
        """a weight-only layer carries quant factors and fixed_shift_value."""
        records = {
            'conv': {'scale_w': 0.5, 'fixed_shift_value': 42},
            'fc': {'scale_d': 0.1},
        }
        quant_records, shift_records = _split_shift_value_records(records)
        self.assertEqual(
            quant_records, {'conv': {'scale_w': 0.5}, 'fc': {'scale_d': 0.1}}
        )
        self.assertEqual(shift_records, {'conv': {'fixed_shift_value': 42}})

    def test_shift_only_record_drops_from_quant_view(self):
        """an FP16 path layer carries only fixed_shift_value: the quantization
        passes would fail on such an empty record."""
        records = {'conv': {'fixed_shift_value': 42}}
        quant_records, shift_records = _split_shift_value_records(records)
        self.assertEqual(quant_records, {})
        self.assertEqual(shift_records, {'conv': {'fixed_shift_value': 42}})

    def test_non_dict_value_kept_in_quant_view(self):
        """global scale style values are not dict, keep them untouched."""
        records = {'conv': {'fixed_shift_value': 42}, 'global_scale': 0.5}
        quant_records, shift_records = _split_shift_value_records(records)
        self.assertEqual(quant_records, {'global_scale': 0.5})
        self.assertEqual(shift_records, {'conv': {'fixed_shift_value': 42}})
