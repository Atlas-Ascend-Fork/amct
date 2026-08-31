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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from amct_pytorch.classic.graph_based.amct_pytorch.common.capacity import (
    query_capacity,
)


class TestQueryCapacity(unittest.TestCase):
    def create_config(self, directory, content):
        config = Path(directory) / 'capacity.csv'
        config.write_text(content, encoding='utf-8')
        return config

    def test_show_capacities_uses_project_logger(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.create_config(directory, 'feature,bool,true\n')
            capacity = query_capacity.Capacity(config)

            with patch.object(query_capacity.LOGGER, 'logi') as mock_logi:
                capacity.show_capacities()

        mock_logi.assert_called_once_with("{'feature': True}")

    def test_unknown_capacity_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.create_config(directory, 'feature,number,1\n')

            with self.assertRaisesRegex(ValueError, 'Unknown type number'):
                query_capacity.Capacity(config)

    def test_unknown_boolean_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.create_config(directory, 'feature,bool,enabled\n')

            with self.assertRaisesRegex(
                ValueError, 'Unknown value enabled for feature'
            ):
                query_capacity.Capacity(config)
