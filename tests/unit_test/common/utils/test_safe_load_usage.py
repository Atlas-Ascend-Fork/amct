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
from pathlib import Path


ISSUE_82_FILES = [
    "amct_pytorch/experimental/quantization/DeepSeekV3.2/extract_calib_data.py",
    "amct_pytorch/experimental/quantization/DeepSeekV3.2/pp/forward/infer.py",
    "amct_pytorch/experimental/quantization/DeepSeekV3.2/cores/calibrator/utils.py",
    "amct_pytorch/experimental/quantization/DeepSeekV3.2/deploy.py",
    "amct_pytorch/experimental/flatquant/flat_quant_module/flat_utils.py",
    "amct_pytorch/classic/graph_based/amct_pytorch/parser/module_based_record_parser.py",
    "amct_pytorch/classic/graph_based/amct_pytorch/utils/model_util.py",
]


def test_issue_82_files_do_not_disable_safe_torch_load():
    for file_name in ISSUE_82_FILES:
        source = Path(file_name).read_text(encoding="utf-8")
        assert "weights_only=False" not in source


def test_issue_82_files_use_safe_torch_load_for_pickle_artifacts():
    for file_name in ISSUE_82_FILES:
        source = Path(file_name).read_text(encoding="utf-8")
        assert "safe_torch_load" in source
