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
"""
Dump tensor data to binary files.
"""

import os
import numpy as np
import torch
from torch import nn
from ....amct_pytorch.utils.log import LOGGER


class DUMP(nn.Module):
    """Dump tensor data to binary files and pass through inputs."""
    _DTYPE_INDEX = {torch.float32: 0, torch.float64: 1, torch.int32: 2}

    def __init__(self, layers_name, dump_config):
        """
        Initialize DUMP module.

        Args:
            layers_name: List of layer names (only first element used).
            dump_config: Config with dump_dir and batch_num attributes.
        """
        super().__init__()
        self.dump_dir = dump_config.dump_dir
        self.layer_name = layers_name[0]
        self.batch_num = dump_config.batch_num
        self.cur_batch = 0

    def forward(self, inputs):
        """
        Dump input tensor and return it unchanged.

        Args:
            inputs: Tensor to dump (float32/float64/int32).

        Returns:
            Input tensor unchanged.
        """
        self.cur_batch += 1
        if self.batch_num != -1 and self.cur_batch > self.batch_num:
            return inputs

        self._dump(inputs)
        LOGGER.logi(f"[{self.layer_name}] dump {self.cur_batch}/{self.batch_num}", 'DUMP')

        return inputs

    def _dump(self, tensor):
        """
        Dump tensor to binary file.

        Binary format: type_index(float32) + dim_count(float32) + shape(float32*) + data
        Type mapping: 0=float32, 1=float64, 2=int32

        Args:
            tensor: Tensor to dump (float32/float64/int32).

        Raises:
            RuntimeError: If dtype is unsupported.
        """
        if tensor.dtype not in self._DTYPE_INDEX:
            raise RuntimeError("Unsupported dtype!")

        data = tensor.clone().cpu().contiguous()
        header = [self._DTYPE_INDEX[tensor.dtype], len(data.shape), *data.shape]

        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f"{self.layer_name}_activation_batch{self.cur_batch}.bin")

        with open(path, 'wb') as f:
            f.write(np.array(header, dtype=np.float32).tobytes())
            f.write(data.numpy().tobytes())
