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

import numpy as np
from .sensitivity_base import SensitivityBase


class CosineSimilaritySensitivity(SensitivityBase): # pylint: disable=R0903
    """ class of cosine similarity sensitivity"""
    def __init__(self): # pylint: disable=W0235
        super(CosineSimilaritySensitivity, self).__init__()

    def compare(self, data, other):
        """ calculate the cosine similarity of original output data and fake quant output data

        Args:
            data (np.array): the original data to compare
            other (np.array): the fake quant data to compare

        Returns:
            np.array: similarity value
        """
        data = data.flatten()
        other = other.flatten()
        data = np.asmatrix(data)
        other = np.asmatrix(other)
        # if original data is all 0, fake quant is all 0, no quantization error
        if np.all(data == 0):
            return 1
        # if original data is not all 0, fake quant is all 0, quantization error exists
        if np.all(other == 0):
            return 0
        print(data.shape, other.shape)
        num = float(data * other.T)
        denom = np.linalg.norm(data) * np.linalg.norm(other)
        cos = num / denom
        similarity = 0.5 + 0.5 * cos
        return similarity
