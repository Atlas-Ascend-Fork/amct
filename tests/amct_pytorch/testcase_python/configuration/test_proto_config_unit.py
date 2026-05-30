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

from amct_pytorch.classic.graph_based.amct_pytorch.common.config.proto_config import (
    ProtoConfig,
    ProtoDataType,
)

logger = logging.getLogger(__name__)

ACT = 'activation_quant_params'
WGT = 'weight_quant_params'
CW = 'channel_wise'


class _FakeIfmr:
    dst_type = 1
    max_percentile = 0.999
    min_percentile = 0.001
    search_range_start = 0.7
    search_range_end = 1.3
    search_step = 0.01
    asymmetric = True

    def __init__(self, has_asym=True):
        self._has_asym = has_asym

    def HasField(self, field):
        return field == 'asymmetric' and self._has_asym


class _FakeHfmg:
    num_of_bins = 2048
    dst_type = 2
    asymmetric = False

    def __init__(self, has_asym=True):
        self._has_asym = has_asym

    def HasField(self, field):
        return field == 'asymmetric' and self._has_asym


class _FakeAda:
    num_iteration = 100
    reg_param = 0.01
    beta_range_start = 20.0
    beta_range_end = 2.0
    warm_start = 0.2
    channel_wise = True

    def __init__(self, has_cw=True):
        self._has_cw = has_cw

    def HasField(self, field):
        return field == CW and self._has_cw


class TestProtoConfigStatic(unittest.TestCase):
    def test_parse_data_type_ok(self):
        self.assertEqual(ProtoConfig._parse_data_type(ProtoDataType.INT8.value), 8)
        self.assertEqual(ProtoConfig._parse_data_type(ProtoDataType.INT16.value), 16)

    def test_parse_data_type_invalid(self):
        # INT4 (0) not in dtype_define
        self.assertRaises(ValueError, ProtoConfig._parse_data_type,
                          ProtoDataType.INT4.value)

    def test_channel_wise_is_true_no_weight(self):
        self.assertFalse(ProtoConfig._channel_wise_is_true({}))

    def test_channel_wise_is_true_arq_true(self):
        cfg = {WGT: {'wts_algo': 'arq_quantize', CW: True}}
        self.assertTrue(ProtoConfig._channel_wise_is_true(cfg))

    def test_channel_wise_is_true_default_algo(self):
        # wts_algo None counts as arq_quantize
        cfg = {WGT: {CW: True}}
        self.assertTrue(ProtoConfig._channel_wise_is_true(cfg))

    def test_channel_wise_is_true_ada_false(self):
        cfg = {WGT: {'wts_algo': 'ada_quantize'}}
        self.assertFalse(ProtoConfig._channel_wise_is_true(cfg))

    def test_get_ifmr_config_with_asym(self):
        cfg = type('C', (), {'ifmr_quantize': _FakeIfmr(has_asym=True)})()
        params = {}
        ProtoConfig._get_ifmr_config(cfg, params)
        self.assertEqual(params['act_algo'], 'ifmr')
        self.assertEqual(params['num_bits'], 8)
        self.assertEqual(params['search_range'], [0.7, 1.3])
        self.assertTrue(params['asymmetric'])

    def test_get_ifmr_config_without_asym(self):
        cfg = type('C', (), {'ifmr_quantize': _FakeIfmr(has_asym=False)})()
        params = {}
        ProtoConfig._get_ifmr_config(cfg, params)
        self.assertIsNone(params['asymmetric'])

    def test_get_hfmg_config_with_asym(self):
        cfg = type('C', (), {'hfmg_quantize': _FakeHfmg(has_asym=True)})()
        params = {}
        ProtoConfig._get_hfmg_config(cfg, params)
        self.assertEqual(params['act_algo'], 'hfmg')
        self.assertEqual(params['num_of_bins'], 2048)
        self.assertEqual(params['num_bits'], 16)
        self.assertFalse(params['asymmetric'])

    def test_get_hfmg_config_without_asym(self):
        cfg = type('C', (), {'hfmg_quantize': _FakeHfmg(has_asym=False)})()
        params = {}
        ProtoConfig._get_hfmg_config(cfg, params)
        self.assertIsNone(params['asymmetric'])

    def test_get_ada_quantize_with_channel_wise(self):
        params = {}
        ProtoConfig._get_ada_quantize(_FakeAda(has_cw=True), params)
        self.assertEqual(params['num_iteration'], 100)
        self.assertEqual(params['beta_range'], [20.0, 2.0])
        self.assertTrue(params[CW])

    def test_get_ada_quantize_without_channel_wise(self):
        params = {}
        ProtoConfig._get_ada_quantize(_FakeAda(has_cw=False), params)
        self.assertNotIn(CW, params)


if __name__ == '__main__':
    unittest.main()
