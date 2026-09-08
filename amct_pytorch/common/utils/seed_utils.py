# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import random

import numpy as np
import torch
from transformers import set_seed

__all__ = ['seed_everything']


def seed_everything(seed: int = 0):
    '''Set random seeds for reproducibility.

    Mirrors examples/algorithms/flatquant/src/utils.py: seeds python, numpy,
    torch and the NPU RNGs (manual_seed + manual_seed_all), and forwards to
    ``transformers.set_seed`` for the HuggingFace side. The cudnn flags of
    the example are intentionally dropped: this repo pins CPU-only torch
    (no cudnn backend), and ``deterministic=True`` carries a global
    performance cost.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    set_seed(seed)
