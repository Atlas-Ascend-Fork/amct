# -*- coding: UTF-8 -*-
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

__all__ = [
    'svd_quant',
]

import importlib.util
import os
import torch

_lib_spec = importlib.util.find_spec("libsvd_quant")
_lib_path = (
    _lib_spec.origin
    if _lib_spec is not None and _lib_spec.origin is not None
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsvd_quant.so")
)
if not os.path.isfile(_lib_path):
    raise ImportError("Unable to locate the libsvd_quant extension")
torch.ops.load_library(_lib_path)

# Bind the name advertised in __all__ to the registered operator, so that
# `from amct_ops.svd_quant import svd_quant` and `import *` resolve to it.
svd_quant = torch.ops.amct.svd_quant
