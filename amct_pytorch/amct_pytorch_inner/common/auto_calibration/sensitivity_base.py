#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

automatic calibration sensitivity base class

"""
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
