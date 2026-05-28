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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class SensitivityBase:
    """ base class of sensitivity"""
    def __init__(self):
        pass

    @staticmethod
    def normalize_data(data):
        """ normalize data to [0,1]

        Args:
            data (np.array): data to do normalize

        Returns:
            np.array: data after normalize
        """
        max_val = data.max()
        min_val = data.min()
        normalized_data = (data - min_val) / (max_val - min_val + np.finfo(np.float32).eps)
        return normalized_data

    def compare(self, data, other):
        """ calculate the compare metric of original single layer output
            and fake quant single layer output
        """
        raise NotImplementedError


class MseSensitivity(SensitivityBase):
    """ class of mse similarity sensitivity"""
    def __init__(self, do_normalization=False):
        """ Init func.

        Args:
            do_normalization (bool, optional): normalize data befor calculating mse. Defaults to False.
        """
        super(MseSensitivity, self).__init__()
        self.do_normalization = do_normalization

    def compare(self, data, other):
        """ Calculate the cosine similarity of data and other data

        Args:
            data (np.array): the data to compare
            other (np.array): other data to compare

        Returns:
            np.array: similarity value
        """
        if self.do_normalization:
            data = self.normalize_data(data)
            other = self.normalize_data(other)
        # ignore batch_size
        similarity = np.linalg.norm(data - other) / data.shape[0]
        return similarity
