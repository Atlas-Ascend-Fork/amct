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
__all__ = ['check_dtype', 'check_diff_dtype', 'reshape_params']

import numpy as np

SUPPORTED_DTYPE = [np.float16, np.float32, np.float64]


def check_dtype(arr):
    """Check input data dtype support or not
    """
    if arr is None:
        return
    if arr.dtype not in SUPPORTED_DTYPE:
        raise RuntimeError(f"only support dtype in {SUPPORTED_DTYPE}")


def check_diff_dtype(arr1, arr2):
    """check whether arr1 and arr2 datatype are the same"""
    if arr1 is None or arr2 is None:
        return
    if arr1.dtype != arr2.dtype:
        raise RuntimeError("arr1.dtype %s and arr2.dtype %s must be same" %
                           (arr1.dtype, arr2.dtype))


def reshape_params(params, conv_weight, with_trans=False):
    '''Reshape parameters of BN/Scale layer before fusing.
    The params is with size matching the 'Channel' dimension.
    '''
    if not with_trans:
        # Channel of output is dim[0], expand shape to 4 or 5 dims based on dim[0]
        if len(conv_weight.shape) == 4:
            shape = [-1, 1, 1, 1]
        elif len(conv_weight.shape) == 3:
            shape = [-1, 1, 1]
        else:
            shape = [-1, 1, 1, 1, 1]
    else:
        # Channel of output is dim[1], expand shape to 4 or 5 dims based on dim[1]
        if len(conv_weight.shape) == 4:
            shape = [1, -1, 1, 1]
        elif len(conv_weight.shape) == 3:
            shape = [1, -1, 1]
        else:
            shape = [1, -1, 1, 1, 1]

    return params.reshape(shape)


class FuseConvBnBase():
    """Function: The implement of ARQ quantize algorithm
       APIs: preprocess_params, quantize_data
    """

    @classmethod
    def get_conv_bn_param(cls, conv_params, bn_params):
        ''' Get detail params such as mean, variance'''
        raise NotImplementedError(
            'FuseConvBnBase not implement get_conv_bn_param')

    @classmethod
    def fuse_conv_bn(cls, conv_params, bn_params):
        """check the input conv bn params

        Arguments:
            conv_params {list} -- list of conv_weights, conv_bias
            bn_params {list} -- list bn_mean, bn_variance,
                                bn_scale_factor, bn_epsilon

        return:
            the fused conv weights and bias
        """
        cls._check_params(conv_params, bn_params)
        params = cls.get_conv_bn_param(conv_params, bn_params)
        conv_weight, conv_bias, bn_mean, bn_variance, bn_epsilon = params

        fused_weight = cls.fuse_weight(conv_weight, bn_variance, bn_epsilon)
        fused_bias = cls.fuse_bias(conv_bias, bn_mean, bn_variance, bn_epsilon)
        return fused_weight, fused_bias

    @classmethod
    def fuse_weight(cls, conv_weight, bn_variance, bn_epsilon):
        '''Inputs:
            conv_weight: a np.array, the weight to be fused.
            bn_variance: a np.array, the variance of BN layer.
            bn_epsilon: a small value, the epsilon of BN layer.
        Returns:
            fused_weight: a np.array, the fused weight.
        '''
        variance = np.add(bn_variance, bn_epsilon)
        variance = reshape_params(variance, conv_weight, with_trans=False)

        stdev = np.sqrt(variance)
        fused_weight = np.divide(conv_weight, stdev)
        return fused_weight

    @classmethod
    def fuse_bias(cls, bias, mean, variance, epsilon):
        """ Fuse bias with BN layer's parameters.

        Inputs:
            bias: a np.array, the bias to be fused.
            mean: a np.array, the mean of BN layer.
            variance: a np.array, the variance of BN layer.
        Returns:
            fused_bias: a np.array, the fused bias.
        """
        if bias is None:
            tmp_bias = np.multiply(mean, -1)
        else:
            tmp_bias = bias - mean
        variance = np.add(variance, epsilon)
        stdev = np.sqrt(variance)
        fused_bias = np.divide(tmp_bias, stdev)
        return fused_bias

    @classmethod
    def fuse_bn_conv(cls, bn_params, conv_params):
        """Do BatchNorm + Convolution prameters fusion
        Arguments:
            bn_params {list}: list [bn_mean, bn_variance,
                                bn_scale_factor, bn_epsilon]
            conv_params {list}: list of [conv_weights, conv_bias]
        return:
            the fused conv weights and bias
        """
        cls._check_params(conv_params, bn_params, channel_axis=1)
        params = cls.get_conv_bn_param(
            conv_params, bn_params)
        conv_weight, conv_bias, bn_mean, bn_variance, bn_epsilon = params

        fused_weight = cls.bn_conv_fuse_weight(conv_weight, bn_variance,
                                               bn_epsilon)
        fused_bias = cls.bn_conv_fuse_bias(fused_weight, conv_bias, bn_mean)
        return fused_weight, fused_bias

    @classmethod
    def bn_conv_fuse_weight(cls, conv_weight, bn_variance, bn_epsilon):
        '''Inputs:
            conv_weight: a np.array, the weight to be fused.
            bn_variance: a np.array, the variance of BN layer.
            bn_epsilon: a small value, the epsilon of BN layer.
        Returns:
            fused_weight: a np.array, the fused weight.
        '''
        variance = np.add(bn_variance, bn_epsilon)
        stdev = np.sqrt(variance)
        stdev = reshape_params(stdev, conv_weight, with_trans=True)
        fused_weight = np.divide(conv_weight, stdev)
        return fused_weight

    @classmethod
    def bn_conv_fuse_bias(cls, fused_weight, conv_bias, bn_mean):
        """ Fuse bias with BN layer's parameters.
        Inputs:
            fused_weight:a np.array, the weights already be fused.
            conv_bias: a np.array, the bias to be fused.
            bn_mean: a np.array, the mean of BN layer.
        Returns:
            fused_bias: a np.array, the fused bias.
        """
        bn_mean = reshape_params(bn_mean, fused_weight, with_trans=True)
        tmp_bias_item = np.multiply(fused_weight, bn_mean)
        if len(fused_weight.shape) == 4:
            bias_item = np.sum(tmp_bias_item, axis=(1, 2, 3))
        if len(fused_weight.shape) == 5:
            bias_item = np.sum(tmp_bias_item, axis=(1, 2, 3, 4))
        if conv_bias is None:
            fused_bias = -bias_item
        else:
            fused_bias = conv_bias - bias_item
        return fused_bias
    
    @classmethod
    def _check_params(cls, conv_params, bn_params, channel_axis=0):
        """check the input conv bn params

        Arguments:
            conv_params {list} -- list of conv_weights, conv_bias
            bn_params {list} -- list bn_mean, bn_variance,
                                bn_scale_factor, bn_epsilon
            channel_axis -- axis of conv channel

        Raises:
            RuntimeError: the dtype not support
        """
        conv_weight = conv_params[0]
        conv_bias = conv_params[1]
        bn_mean = bn_params[0]
        bn_variance = bn_params[1]

        check_dtype(conv_weight)
        if conv_bias is not None:
            check_dtype(conv_bias)
        check_dtype(bn_mean)
        check_dtype(bn_variance)

        check_diff_dtype(conv_weight, conv_bias)
        check_diff_dtype(conv_weight, bn_mean)
        check_diff_dtype(conv_weight, bn_variance)

        if len(conv_weight.shape) not in (3, 4, 5):
            raise RuntimeError('conv weight is not 3, 4 or 5 dimension')
        channels = conv_weight.shape[channel_axis]
        if bn_mean.shape[0] != channels:
            raise RuntimeError(
                'mean shape {} not equal to channels {}'.format(
                    bn_mean.shape[0], channels))
        if bn_variance.shape[0] != channels:
            raise RuntimeError(
                'variance shape {} not equal to channels {}'.format(
                    bn_variance.shape[0], channels))
