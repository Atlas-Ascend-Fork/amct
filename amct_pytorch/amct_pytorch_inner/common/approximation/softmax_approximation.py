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

from .softmax_offline_coef import SOFTMAX_RANGE2COEF
from ..utils.files import find_dump_file
from ..utils.files import parse_dump_data
from ...utils.log import LOGGER # pylint: disable=E0402


class SoftmaxCalibrator():
    """
    Function: execute calibration for softmax op to find approximate coefficient.
    APIs: calibrate
    """
    def __init__(self, dump_dir, prefix):
        self.data = None
        self.cali_data = None
        self.quarter_point = None
        self.dump_dir = dump_dir
        self.prefix = prefix

    @staticmethod
    def _exp_approximation(data, a, b, clip_value):
        data = np.clip(data, clip_value, 0)
        data_square = data * data
        data_cube = data_square * data
        data_numerator = 1 + data + a * data_square + b * data_cube
        return np.clip(data_numerator, 0, 1)

    def calibrate(self):
        '''
        Function: Use dump data, do calibration for softmax op, get approximate coefficient.
        Arguments: None
        Return:
            a, b, clip_value: approximate coefficient
        '''
        index = 0
        candidate = {}
        already_selected = set()

        self._concat_dump_data()
        self._process_dump_data()

        for ele in np.arange(int(self.quarter_point), 0, 0.01):
            range_value = round(ele, 2)
            for key in SOFTMAX_RANGE2COEF:
                if range_value in key and SOFTMAX_RANGE2COEF[key] not in already_selected:
                    already_selected.add(SOFTMAX_RANGE2COEF[key])
                    candidate[index] = SOFTMAX_RANGE2COEF[key]
                    index += 1
        if len(candidate) == 0:
            candidate[0] = (0.3912846101546833, 0.05358591935341088, -3.48124210167968)

        loss_dict = {}
        accurate_result = np.exp(self.cali_data)
        for key, _ in candidate.items():
            a, b, clip_value = candidate.get(key)
            temp = accurate_result - self._exp_approximation(self.cali_data, a, b, clip_value)
            loss_dict[key] = np.linalg.norm(temp.astype(float), ord=2)
        min_key = min(loss_dict.keys(), key=(lambda x: loss_dict.get(x)))
        LOGGER.logd('choose approximate coef {} for layer {}'.format(candidate.get(min_key), self.prefix))
        return candidate.get(min_key)

    def _concat_dump_data(self):
        concat_data = None
        dump_files = find_dump_file(self.dump_dir, self.prefix)
        for file_name in dump_files:
            temp = parse_dump_data(file_name, with_type=True)
            temp = temp.flatten()
            if concat_data is None:
                concat_data = temp
            else:
                concat_data = np.concatenate((concat_data, temp))
        self.data = concat_data

    def _process_dump_data(self):
        concat_data = self.data.flatten()
        concat_data = concat_data - np.max(concat_data)
        _, bin_edges = np.histogram(concat_data, bins=2)

        self.cali_data = concat_data[np.where(concat_data > bin_edges[1])]
        _, bin_edges_ = np.histogram(self.cali_data, bins=2)
        self.quarter_point = bin_edges_[1]
        self.quarter_point = max(self.quarter_point, -30)
