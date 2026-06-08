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
import inspect
import logging

import torch

WEIGHTS_ONLY = "weights_only"


def _supports_weights_only():
    return WEIGHTS_ONLY in inspect.signature(torch.load).parameters


def safe_torch_load(file_path, **kwargs):
    """
    Load PyTorch artifacts with safe deserialization by default.

    On PyTorch versions that provide the weights_only argument (>= 1.13),
    enable it by default to avoid unsafe pickle deserialization. On older
    versions that AMCT still supports (torch 1.5 ~ 1.12), weights_only is
    unavailable, so it is stripped and the load falls back to torch's native
    behaviour with a warning, rather than failing outright.
    """
    if _supports_weights_only():
        kwargs.setdefault(WEIGHTS_ONLY, True)
    else:
        kwargs.pop(WEIGHTS_ONLY, None)
        logging.warning(
            "Current PyTorch %s does not support torch.load(weights_only=...); "
            "falling back to unsafe pickle loading for %s. Only load trusted "
            "files, and upgrade to PyTorch >= 1.13 for safe deserialization.",
            torch.__version__,
            file_path,
        )
    return torch.load(file_path, **kwargs)
