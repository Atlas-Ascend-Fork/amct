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

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.ada_round import (
    kl_divergence as kl,
)

logger = logging.getLogger(__name__)


class TestFindScaleAndOffsetByKl(unittest.TestCase):
    def test_per_tensor_shape(self):
        torch.manual_seed(0)
        data = torch.randn(64, 8)
        scale, offset = kl.find_scale_and_offset_by_kl(data, channel_wise=False)
        self.assertEqual(scale.shape[0], 1)
        self.assertEqual(offset.shape[0], 1)

    def test_per_tensor_scale_positive(self):
        # scale derives from hist_min/-128 with hist_min<0, so scale must be > 0
        torch.manual_seed(42)
        data = torch.randn(2000) * 3.0
        scale, _ = kl.find_scale_and_offset_by_kl(data, channel_wise=False)
        self.assertGreater(float(scale[0]), 0.0)

    def test_per_tensor_scale_is_saturating(self):
        # KL picks a saturation threshold <= data range, so scale*128 <= abs_max
        torch.manual_seed(42)
        data = torch.randn(2000) * 3.0
        scale, _ = kl.find_scale_and_offset_by_kl(data, channel_wise=False)
        self.assertLessEqual(float(scale[0]) * 128, float(data.abs().max()) + 1e-3)

    def test_scale_linear_in_data_range(self):
        # doubling the data magnitude should ~double the chosen scale; this
        # exercises the whole histogram->merge->kl->threshold chain end to end.
        torch.manual_seed(42)
        data = torch.randn(2000) * 3.0
        scale1, _ = kl.find_scale_and_offset_by_kl(data, channel_wise=False)
        scale2, _ = kl.find_scale_and_offset_by_kl(data * 2, channel_wise=False)
        self.assertAlmostEqual(float(scale2[0]) / float(scale1[0]), 2.0, places=4)

    def test_per_channel_shape(self):
        torch.manual_seed(0)
        data = torch.randn(16, 8)
        scale, offset = kl.find_scale_and_offset_by_kl(data, channel_wise=True)
        self.assertEqual(scale.shape[0], 16)
        self.assertEqual(offset.shape[0], 16)

    def test_per_channel_independent_scales(self):
        # each channel optimized independently: the larger-magnitude channel
        # must get the larger scale.
        torch.manual_seed(7)
        data = torch.stack([torch.randn(500) * 1.0, torch.randn(500) * 10.0])
        scale, _ = kl.find_scale_and_offset_by_kl(data, channel_wise=True)
        self.assertGreater(float(scale[1]), float(scale[0]))


class TestKlOptimize(unittest.TestCase):
    def test_init_defaults(self):
        opt = kl.KlOptimize(torch.randn(100))
        self.assertEqual(opt.hist_bins, kl.HIST_BINS)
        self.assertEqual(opt.num_search, kl.NUM_SEARCH)
        self.assertEqual(opt.quant_bins, kl.QUANT_BINS)

    def test_optimize_kl_degenerate(self):
        # all-equal tensor -> hist_max == hist_min -> early return defaults
        opt = kl.KlOptimize(torch.zeros(32))
        scale, offset = opt.optimize_kl()
        self.assertEqual(float(scale), 1.0)
        self.assertEqual(int(offset), 0)

    def test_optimize_kl_normal(self):
        # for a real distribution the chosen scale is positive and the
        # saturation threshold (scale*128) stays within the data range.
        torch.manual_seed(1)
        tensor = torch.randn(500)
        opt = kl.KlOptimize(tensor)
        scale, offset = opt.optimize_kl()
        self.assertEqual(int(offset), 0)
        self.assertGreater(float(scale), 0.0)
        self.assertLessEqual(float(scale) * 128, float(tensor.abs().max()) + 1e-3)


if __name__ == '__main__':
    unittest.main()
