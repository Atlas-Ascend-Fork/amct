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
import os
import shutil
import tempfile
import unittest

import torch

from amct_pytorch.graph_based_compression.amct_pytorch.utils.model_util import load_pth_file


def mark_payload_loaded(marker_path):
    with open(marker_path, 'w') as marker_file:
        marker_file.write('loaded')
    return {}


class UnsafePayload:
    def __init__(self, marker_path):
        self.marker_path = marker_path

    def __reduce__(self):
        return mark_payload_loaded, (self.marker_path,)


class TestSafeTorchLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='amct_safe_load_')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_pth_file_rejects_unsafe_pickle_payload(self):
        pth_file = os.path.join(self.temp_dir, 'unsafe.pth')
        marker_path = os.path.join(self.temp_dir, 'payload_loaded')
        torch.save(UnsafePayload(marker_path), pth_file)

        model = torch.nn.Linear(1, 1)
        with self.assertRaises(Exception):
            load_pth_file(model, pth_file, None)

        self.assertFalse(os.path.exists(marker_path))


if __name__ == '__main__':
    unittest.main()
