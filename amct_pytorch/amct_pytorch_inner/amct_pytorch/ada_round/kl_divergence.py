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
import math
import torch
import numpy as np

from ...amct_pytorch.custom_op.utils import check_quant_data
from ...amct_pytorch.custom_op.utils import process_scale
from ...amct_pytorch.common.utils.vars_util import DEFAULT_NUM_BITS
from ...amct_pytorch.common.utils.util import version_higher_than

HIST_BINS = 512
NUM_SEARCH = 20
QUANT_BINS = pow(2, DEFAULT_NUM_BITS) - 1


def find_scale_and_offset_by_kl(data_tensor, channel_wise):
    '''
    Function: get scale and offset by kl divergence 
    Parameters:
        data_tensor: model input
        channel_wise: indicates whether to enable the pre channel
    Return:
        scale and offset
    '''
    check_quant_data(data_tensor, 'weight')

    if not channel_wise:
        scale = torch.zeros(1, device=data_tensor.device)
        offset = torch.zeros(1, dtype=torch.int32, device=data_tensor.device)
        scale[0], offset[0] = KlOptimize(data_tensor).optimize_kl()
        scale, offset = process_scale(scale, offset, False)
        return scale, offset

    scale = torch.zeros(data_tensor.shape[0], device=data_tensor.device)
    offset = torch.zeros(data_tensor.shape[0], dtype=torch.int32, device=data_tensor.device)
 
    for i in range(scale.shape[0]):
        scale[i], offset[i] = KlOptimize(data_tensor[i]).optimize_kl()
    scale, offset = process_scale(scale, offset, False)
    return scale, offset


class KlOptimize():
    '''
    Function: kl divergence quantization.
    APIs: optimize_kl
    '''
    def __init__(self, tensor):
        self.tensor = tensor.flatten()
        self.hist_bins = HIST_BINS
        self.num_search = NUM_SEARCH
        self.histogram = [0] * self.hist_bins
        self.hist_min = 0
        self.hist_max = 0
        self.quant_bins = QUANT_BINS
 
    def optimize_kl(self):
        '''
        Function: Calculate scale by means of kl
        Parameters:
            None
        Return:
            scale offset
        '''
        scale = torch.tensor(1)
        offset = torch.tensor(0)
        kl_min = float('inf')
        if not self._init_histogram():
            return scale, offset

        # search to find the best saturate threshold
        for i in range(self.num_search):
            hist_bin_start = i
            hist_bin_end = self.hist_bins - i

            # histogram window size
            hist_win_num = hist_bin_end - hist_bin_start

            # saturate the left & right
            p = self.histogram[hist_bin_start: hist_bin_end]

            p[0] += sum(self.histogram[:hist_bin_start])
            p[-1] += sum(self.histogram[hist_bin_end:])

            # quantize p to q, quant bins is 255 for 8 bit
            # q has the same length with p
            q = [0] * len(p)

            # merge some bins to get 255 bins, merge_bins
            merge_bins = hist_win_num / self.quant_bins
            for j in range(self.quant_bins):
                # start & end for each q bin
                index_start = math.ceil(j * merge_bins)
                index_end = math.ceil((j + 1) * merge_bins)
                if j == self.quant_bins - 1:
                    index_end = hist_win_num

                # calc sum & norm in the merged bin
                sum_val = 0
                norm = 0
                for k in range(index_start, index_end):
                    sum_val += p[k]
                    norm += (p[k] != 0)
                
                # split merged histogram
                if norm == 0:
                    continue
                merge_value = sum_val / norm
                for k in range(index_start, index_end):
                    q[k] = merge_value if p[k] != 0 else q[k]

            # compute kl
            p = self._condition_histogram(p)
            q = self._condition_histogram(q)
            kl = self._compute_kl(p, q)
            if kl < kl_min:
                hist_bin_width = (self.hist_max - self.hist_min) / self.hist_bins
                hist_min = self.hist_min + i * hist_bin_width
                kl_min = kl
                scale = hist_min / -128
    
        return scale, offset
    
    def _init_histogram(self):
        '''init histogram from tensor'''
        # "max_cpu" not implemented for 'half' in torch 1.10.0
        if version_higher_than(torch.__version__, '1.10.0'):
            hist_max = self.tensor.abs().max()
        else:
            hist_max = self.tensor.to(torch.float32).abs().max()
        hist_min = -hist_max

        if hist_max == hist_min:
            return False

        self.histogram = torch.histc(self.tensor.to(torch.float32), bins=self.hist_bins, 
                                     min=hist_min, max=hist_max).tolist()

        self.hist_min = hist_min
        self.hist_max = hist_max
        return True

    def _compute_kl(self, p, q):
        '''kl divergence Compute'''
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        p_normalized = p / np.sum(p)
        q_normalized = q / np.sum(q)
        kl = np.sum(p_normalized * np.log(p_normalized / q_normalized))
        return kl

    def _condition_histogram(self, hist):
        '''adjust histogram'''

        is_zero = [val == 0 for val in hist]
        zeros_num = is_zero.count(True)
        non_zeros_num = len(hist) - zeros_num
 
        eps_zeros = 0.0001
        eps_non_zeros = eps_zeros * zeros_num / non_zeros_num
        if eps_non_zeros >= 1:
            # zeros_num >> non_zeros_num
            return hist
 
        for i, value in enumerate(hist):
            hist[i] = eps_zeros if value == 0 else (value + eps_non_zeros)
        return hist