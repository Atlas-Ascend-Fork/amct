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
from .conv_bn_fuse_base import check_dtype
from .conv_bn_fuse_base import check_diff_dtype
from .conv_bn_fuse_base import reshape_params

SUPPORT_DIMENSION = (3, 4, 5)


class FuseConvScaleBase():
    """Function: The implement of ARQ quantize algorithm
       APIs: preprocess_params, quantize_data
    """

    @staticmethod
    def _check_params(conv_weight, conv_bias, scale, beta, channel_axis=0):
        """check the input conv scale params

        Arguments:
            conv_weight: a np.array, the weight to be fused.
            conv_bias: a np.array, the bias to be fused.
            beta: a np.array, the beta of Scale layer.
            scale: a np.array, the scale of Scale layer.
        """

        if conv_weight is None:
            raise RuntimeError("conv_weight can not be none")
        if scale is None:
            raise RuntimeError("scale can not be none")

        check_dtype(conv_bias)
        check_dtype(conv_weight)
        check_dtype(scale)
        check_dtype(beta)

        check_diff_dtype(conv_weight, conv_bias)
        check_diff_dtype(conv_weight, scale)
        check_diff_dtype(conv_weight, beta)

        if conv_weight.ndim not in SUPPORT_DIMENSION:
            raise RuntimeError("conv_weight support ndim is {}".format(SUPPORT_DIMENSION))
        if scale.ndim != 1:
            raise RuntimeError("scale only support ndim is 1")
        if conv_weight.shape[channel_axis] != scale.shape[0]:
            raise RuntimeError(
                "weight channel:{} must be same with scale shape:{}" \
                .format(conv_weight.shape[channel_axis], scale.shape[0]))
        if beta is not None and \
            conv_weight.shape[channel_axis] != beta.shape[0]:
            raise RuntimeError(
                "weight channel:{} must be same with beta shape:{}" \
                .format(conv_weight.shape[channel_axis], beta.shape[0]))
    
    @classmethod
    def fuse_conv_scale(cls, conv_weight, conv_bias, scale, beta):
        ''' the func to fuse conv + Scale
        including weight, bias fuse.
        inputs:conv_weight conv_bias scale beta.
        return:fused_weight fused_bias.
        '''
        cls._check_params(conv_weight, conv_bias, scale, beta)
        fused_weight = cls.fuse_weight(conv_weight, scale)
        fused_bias = cls.fuse_bias(conv_bias, scale, beta)
        return fused_weight, fused_bias

    @classmethod
    def fuse_weight(cls, conv_weight, scale):
        '''Inputs:
            conv_weight: a np.array, the weight to be fused.
            scale: a np.array, the scale of Scale layer.
        Returns:
            fused_weight: a np.array, the fused weight.
        '''
        scale = reshape_params(scale, conv_weight, with_trans=False)
        fused_weight = np.multiply(conv_weight, scale)
        return fused_weight

    @classmethod
    def fuse_bias(cls, conv_bias, scale, beta):
        """ Fuse bias with BN layer's parameters.
        Inputs:
            conv_bias: a np.array, the bias to be fused.
            scale: a np.array, the scale of Scale layer.
            beta: a np.array, the beta of Scale layer.
        Returns:
            fused_bias: a np.array, the fused bias.
        """
        if conv_bias is None:
            fused_bias = beta
            return fused_bias
        tmp_bias = np.multiply(conv_bias, scale)
        if beta is None:
            return tmp_bias
        fused_bias = np.add(tmp_bias, beta)
        return fused_bias

    @classmethod
    def fuse_scale_conv(cls, conv_weight, conv_bias, scale, beta):
        """the func to fuse scale + conv
        including weight, bias fuse.
        inputs:conv_weight conv_bias scale beta.
        return:fused_weight fused_bias.
        """
        cls._check_params(conv_weight, conv_bias, scale, beta, channel_axis=1)
        fused_weight = cls.scale_conv_fuse_weight(conv_weight, scale)
        fused_bias = cls.scale_conv_fuse_bias(conv_weight, conv_bias, beta)
        return fused_weight, fused_bias

    @classmethod
    def scale_conv_fuse_weight(cls, conv_weight, scale):
        """Do convolution's weights and Scale's scale fusion
        """
        scale = reshape_params(scale, conv_weight, with_trans=True)
        fused_weight = np.multiply(conv_weight, scale)
        return fused_weight

    @classmethod
    def scale_conv_fuse_bias(cls, conv_weight, conv_bias, beta):
        """Do convolution's bias and Scale's scale, beta fusion
        """
        fused_bias = None
        if beta is not None:
            beta = reshape_params(beta, conv_weight, with_trans=True)
            tmp_bias_item = np.multiply(conv_weight, beta)
            if conv_weight.ndim == 4:
                fused_bias = np.sum(tmp_bias_item, axis=(1, 2, 3))
            if conv_weight.ndim == 5:
                fused_bias = np.sum(tmp_bias_item, axis=(1, 2, 3, 4))
        if conv_bias is not None:
            if fused_bias is None:
                fused_bias = conv_bias
            else:
                fused_bias += conv_bias
        return fused_bias
