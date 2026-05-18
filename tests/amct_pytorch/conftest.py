# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import os
import sys
from unittest.mock import MagicMock

# Add project root to path so amct_pytorch is importable without installation
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(_project_root))

# Mock graph_based_compression to avoid requiring compiled proto files.
# The calibration modules under test (minmax, smoothquant, etc.) do not
# depend on graph_based_compression; it is only pulled in by amct_pytorch/__init__.py.
_mock = MagicMock()
_mock.__all__ = []
sys.modules.setdefault('amct_pytorch.graph_based_compression', _mock)
sys.modules.setdefault('amct_pytorch.graph_based_compression.amct_pytorch', _mock)
