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

import torch

WEIGHTS_ONLY = "weights_only"


def _supports_weights_only():
    return WEIGHTS_ONLY in inspect.signature(torch.load).parameters


def safe_torch_load(file_path, **kwargs):
    """
    Load tensor-only PyTorch artifacts with safe deserialization by default.

    Supported PyTorch versions provide the weights_only argument. Do not fall
    back to unsafe pickle loading when the runtime does not support it.
    """
    if not _supports_weights_only():
        raise RuntimeError(
            "safe_torch_load requires torch.load to support weights_only, "
            f"but current PyTorch {torch.__version__} does not."
        )
    kwargs.setdefault(WEIGHTS_ONLY, True)
    return torch.load(file_path, **kwargs)
