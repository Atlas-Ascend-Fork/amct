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
from unittest.mock import patch

import numpy as np

from amct_pytorch.classic.graph_based.amct_pytorch.common.auto_calibration import (
    cosine_similarity_sensitivity,
)


class TestCosineSimilaritySensitivity(unittest.TestCase):
    @patch.object(cosine_similarity_sensitivity.LOGGER, 'logd')
    def test_nonzero_inputs_are_compared_with_debug_log(self, mock_logd):
        sensitivity = cosine_similarity_sensitivity.CosineSimilaritySensitivity()

        similarity = sensitivity.compare(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

        self.assertAlmostEqual(similarity, 1.0)
        mock_logd.assert_called_once_with('data shape: (1, 2), other shape: (1, 2)')
