# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

import inspect

import torch

WEIGHTS_ONLY = "weights_only"


def _supports_weights_only():
    return WEIGHTS_ONLY in inspect.signature(torch.load).parameters


def safe_torch_load(file_path, **kwargs):
    """
    Load PyTorch artifacts with safe deserialization by default.

    On PyTorch versions that provide the weights_only argument (>= 1.13),
    enable it by default to avoid unsafe pickle deserialization. On older
    versions where weights_only is unavailable, it is stripped and the load
    falls back to torch's native behaviour with a warning, rather than
    failing outright.

    This is a self-contained copy for the experimental DeepSeekV3.2 sample,
    which is designed to run standalone (entry scripts executed from this
    directory) without depending on the top-level amct_pytorch package.
    """
    if _supports_weights_only():
        kwargs.setdefault(WEIGHTS_ONLY, True)
    else:
        kwargs.pop(WEIGHTS_ONLY, None)
        from loguru import logger
        logger.warning(
            f"Current PyTorch {torch.__version__} does not support "
            f"torch.load(weights_only=...); falling back to unsafe pickle "
            f"loading for {file_path}. Only load trusted files, and upgrade "
            f"to PyTorch >= 1.13 for safe deserialization."
        )
    return torch.load(file_path, **kwargs)
