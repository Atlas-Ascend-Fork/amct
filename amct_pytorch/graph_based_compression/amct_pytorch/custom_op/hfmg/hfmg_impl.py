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

from ....amct_pytorch.utils.vars import BASE, FLT_EPSILON, MIN_BIN_RATIO, STEP_DIVISOR, HFMG_POW
from ....amct_pytorch.custom_op.utils import process_scale, calculate_scale_offset

BASENUM = 2


class DataBin:
    def __init__(self, count, lower_bound, higher_bound):
        """
        Init objective.

        Args:
            count (int): count of the bin
            lower_bound (torch.Tensor): lower bound of data
            higher_bound (torch.Tensor): higher bound of data
        """
        self.count = count
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound


def get_norm(delta_begin, delta_end, density):
    """
    Calculate norm value.

    Args:
        delta_begin (torch.Tensor): delta begin
        delta_end (torch.Tensor): delta end
        density (torch.Tensor): density
    
    Returns: 
        norm (torch.Tensor): norm value
    """
    norm = (delta_end.pow(HFMG_POW) - delta_begin.pow(HFMG_POW)) / HFMG_POW
    norm = norm * density
    return norm


def hfmg_merge_inter(data_bins, merged_data_bins, same_range, merged_data_min, merged_bin_width):
    """
    Merge original data_bins into merged_bins.
    
    Args:
        data_bins (list): the list of original DataBin
        merged_data_bins (list): the list of new merged DataBin
        same_range (bool): the range of original and merged bin is the same or not
        merged_data_min (float): min data of merged data
        merged_bin_width (float): width of merged data
    
    Returns:
        merged_data_bins (list): the list of merged DataBin
    """
    num_of_bins = len(data_bins)

    if same_range:
        # do not need to change data range
        for i in range(num_of_bins):
            merged_data_bins[i].count += data_bins[i].count
    else:
        # reallocate the count of existed bins into the new mergedDataBins
        # here assumes that data are uniform differential distributed in the bins
        for data_bin in data_bins:
            lower = data_bin.lower_bound
            upper = data_bin.higher_bound
            idx_low = int((lower - merged_data_min) // merged_bin_width)
            idx_high = int((upper - merged_data_min) // merged_bin_width)
            
            idx_high = num_of_bins - 1 if idx_high >= num_of_bins else idx_high
            
            if idx_low == idx_high:
                # original bin just include by a merged bin
                merged_data_bins[idx_low].count += data_bin.count
            else:
                # original bin across two merged bin, assume that data is uniformly distributed in bin
                scale_of_bin = (merged_data_bins[idx_low].higher_bound - lower) / (upper - lower)
                count_of_lower = int(data_bin.count * scale_of_bin)
                count_of_higher = data_bin.count - count_of_lower
                merged_data_bins[idx_low].count += count_of_lower
                merged_data_bins[idx_high].count += count_of_higher
    
    return merged_data_bins


def hfmg_get_search_range(data_bins):
    """
    Generate optimal quantization interval search range.
    
    Args:
        data_bins (list): the list of DataBin
    
    Returns:
        search_range (list): the list of searched ranges
    """
    search_range = []
    if not data_bins:
        return search_range
    
    left = 0
    right = len(data_bins) - 1
    total_num = sum(b.count for b in data_bins)
    
    # stepSize must be >= 1
    step_size = max(math.ceil(total_num / STEP_DIVISOR), 1)
    min_bins = math.ceil(len(data_bins) / MIN_BIN_RATIO)
    # specify the search range first
    while right - left > min_bins:
        temp_left = left
        temp_right = right
        left_sum = 0
        right_sum = 0
        
        while temp_left < temp_right and left_sum < step_size:
            left_sum += data_bins[temp_left].count
            temp_left += 1
        
        while temp_left < temp_right and right_sum < step_size:
            right_sum += data_bins[temp_right].count
            temp_right -= 1
        
        if (right - temp_right) >= (temp_left - left):
            right = temp_right
        else:
            left = temp_left
        
        search_range.append((left, right))
    
    return search_range


def hfmg_calculate_loss(data_bins, dst_num_bins, search_range):
    """
    Calculate the L2 loss for bin merging.
    
    Args:
        data_bins (list): the list of DataBin
        dst_num_of_bins (int): destination num of bins
        search_range (list): the list of search range
    
    Returns:
        l2_loss (torch.Tensor): l2 loss
    """
    num_bins = len(data_bins)
    data_min = data_bins[0].lower_bound
    data_max = data_bins[-1].higher_bound
    bin_width = (data_max - data_min) / num_bins
    
    device = bin_width.device
    data_bin_count = torch.tensor([bin_data.count for bin_data in data_bins], device=device)
    range_len = len(search_range)
    densities = (data_bin_count / bin_width).reshape(-1, 1).repeat(1, range_len)

    search_range = torch.tensor(search_range, device=device)
    bin_index_lefts = search_range[:, 0]
    bin_index_rights = search_range[:, 1]

    dst_bin_widths = bin_width * (bin_index_rights - bin_index_lefts + 1) / dst_num_bins
    bins_arange = torch.arange(num_bins, device=device)
    src_bin_begins = (bins_arange.reshape(-1, 1).repeat(1, range_len) - bin_index_lefts) * bin_width
    src_bin_ends = src_bin_begins + bin_width
    temp1s = (src_bin_begins // dst_bin_widths).relu()
    temp2s = (src_bin_ends // dst_bin_widths).relu()

    idx_max_tensor = torch.tensor(dst_num_bins - 1, device=device)
    dst_bin_begin_idxs = torch.min(idx_max_tensor, temp1s)
    dst_bin_end_idxs = torch.min(idx_max_tensor, temp2s)

    half_dst_bin_widths = (dst_bin_widths / BASE).reshape(-1, range_len).repeat(num_bins, 1)
    dst_bin_begin_centers = dst_bin_begin_idxs * dst_bin_widths + half_dst_bin_widths

    same_bin_flag = dst_bin_begin_idxs == dst_bin_end_idxs
    diff_bin_flag = dst_bin_begin_idxs != dst_bin_end_idxs

    delta_begins = src_bin_begins - dst_bin_begin_centers
    delta_ends = src_bin_ends - dst_bin_begin_centers

    # if src_bin is entirely within 1 dst_bin
    l2_loss = get_norm(delta_begins * same_bin_flag, delta_ends * same_bin_flag, densities * same_bin_flag).sum(dim=0)

    # the left dst bin and src bin overlapping parts
    l2_loss += get_norm(
        delta_begins * diff_bin_flag, half_dst_bin_widths * diff_bin_flag, densities * diff_bin_flag).sum(dim=0)

    # middle full dst bin loss
    mid_bins = (dst_bin_end_idxs - dst_bin_begin_idxs) * diff_bin_flag - 1
    mid_loss = get_norm(
        -half_dst_bin_widths * diff_bin_flag, half_dst_bin_widths * diff_bin_flag, densities * diff_bin_flag)
    l2_loss += (mid_bins * mid_loss).sum(dim=0)

    # the right dst bin and src bin overlapping parts
    delta_ends = (src_bin_ends - dst_bin_end_idxs * dst_bin_widths - half_dst_bin_widths) * diff_bin_flag
    l2_loss += get_norm(-half_dst_bin_widths * diff_bin_flag, delta_ends, densities * diff_bin_flag).sum(dim=0)

    l2_loss = torch.where(dst_bin_widths < FLT_EPSILON, torch.tensor(0., device=device), l2_loss)

    return l2_loss


def hfmg_compute(data_bins, num_bits, with_offset):
    """
    Calculate the optimal quantization range with loss.
    
    Args:
        data_bins (list): the list of DataBin
        num_bits (int): number of bins
        with_offset (bool): with offset or not
    
    Returns:
        scale (torch.Tensor): Quant factor to do scaling
        offset (torch.Tensor): Quant factor to do offseting
    """
    search_range = hfmg_get_search_range(data_bins)
    
    dst_num_bins = BASE ** num_bits
    l2_loss = hfmg_calculate_loss(data_bins, dst_num_bins, search_range)

    best_idx = torch.argmin(l2_loss)
    best_left, best_right = search_range[best_idx]
    
    data_min = data_bins[0].lower_bound
    data_max = data_bins[-1].higher_bound
    bin_width = (data_max - data_min) / len(data_bins)
    
    clip_min = (data_min + best_left * bin_width).to(l2_loss.device)
    clip_max = (data_min + (best_right + 1) * bin_width).to(l2_loss.device)
    
    scale, offset = calculate_scale_offset(clip_max, clip_min, with_offset, "INT" + str(num_bits))
    
    return scale, offset


def transform_hist(hist, hist_range, nbins):
    """
    Generate DataBin from histogram.
    
    Args:
        hist (torch.Tensor): histogram of data
        hist_range (torch.Tensor): the range of histogram
        nbins (int): number of bins
    
    Returns:
        data_bins (list): the list of DataBin
    """
    hist_vec = hist.cpu().tolist()
    
    if nbins == 0:
        return []
    
    data_min = hist_range[0]
    data_max = hist_range[-1]
    bin_width = (data_max - data_min) / nbins
    
    data_bins = []
    for i in range(nbins):
        lower = data_min + i * bin_width
        upper = data_min + (i + 1) * bin_width
        data_bins.append(DataBin(hist_vec[i], lower, upper))
    
    return data_bins


def hfmg_arq_pytorch(min_tensor, max_tensor, num_bits, with_offset):
    """
    Calculate scale and offset.
    
    Args:
        min_tensor (torch.Tensor): min value of input_data
        max_tensor (torch.Tensor): max value of input_data
        num_bits (int): number of bins
        with_offset (bool): with offset or not
    
    Returns:
        scale (torch.Tensor): Quant factor to do scaling
        offset (torch.Tensor): Quant factor to do offseting
    """
    scale, offset = calculate_scale_offset(max_tensor, min_tensor, with_offset, "INT" + str(num_bits))
    scale, offset = process_scale(scale, offset, with_offset, num_bits)
    
    return scale, offset


def hfmg_merge_pytorch(hist, hist_range, new_hist, new_hist_range, nbins=4096):
    """
    Merge histograms.
    
    Args:
        hist (torch.Tensor): histogram of data
        hist_range (torch.Tensor): the range of histogram
        new_hist (torch.Tensor): histogram of new data
        new_hist_range (torch.Tensor): the range of new histogram
        nbins (int, optional): number of bins
    
    Returns:
        merged_hist (torch.Tensor): merged histogram
    """
    # Convert tensors to DataBins
    data_bins = transform_hist(hist, hist_range, nbins)
    
    merged_data_bins = transform_hist(new_hist, new_hist_range, nbins)
    
    # Get merged parameters
    merged_min = min(new_hist_range[0], hist_range[0])
    merged_max = max(new_hist_range[-1], hist_range[-1])
    merged_bin_width = (merged_max - merged_min) / nbins
    
    # Check if ranges are same
    same_range = (math.isclose(new_hist_range[0], hist_range[0], abs_tol=FLT_EPSILON) and 
                  math.isclose(new_hist_range[-1], hist_range[-1], abs_tol=FLT_EPSILON))
    
    hfmg_merge_inter(data_bins, merged_data_bins, same_range, merged_min, merged_bin_width)

    merged_hist = torch.tensor([bin.count for bin in merged_data_bins], dtype=torch.int32, device=hist.device)

    return merged_hist


def hfmg_forward_pytorch(hist, hist_range, num_bits=8, with_offset=False, nbins=4096):
    """
    Main quantization forward pass.
    
    Args:
        hist (torch.Tensor): histogram of data
        hist_range (torch.Tensor): the range of histogram
        num_bits (int, optional): quant bits
        with_offset (bool, optional): with offset or not
        nbins (int, optional): number of bins
    
    Returns:
        scale (torch.Tensor): Quant factor to do scaling
        offset (torch.Tensor): Quant factor to do offseting
    """
    data_bins = transform_hist(hist, hist_range, nbins)
    
    scale, offset = hfmg_compute(data_bins, num_bits, with_offset)

    device = hist.device
    clip_max_out = torch.tensor(1., device=device).mul_(BASENUM ** (num_bits - 1) - 1).sub_(offset).mul_(scale)
    clip_min_out = torch.tensor(1., device=device).mul_(BASENUM ** (num_bits - 1)).add_(offset).mul_(-scale) 

    return scale, offset, clip_max_out, clip_min_out


def hfmg_backward_pytorch(grad):
    """
    Compute the gradient for the HFMG backward pass.

    Args:
        grad (torch.Tensor): The gradient tensor from the subsequent layer.
    """
    return grad
