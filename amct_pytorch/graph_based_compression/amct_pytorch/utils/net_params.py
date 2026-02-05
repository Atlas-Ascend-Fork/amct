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

from ...amct_pytorch.common.utils.net_params import ParamsHelper
from ...amct_pytorch.utils.model_util import ModuleHelper


class ParamsHelperTorch:
    """
    Torch Helper to calculate FLOPs.
    """
    def __init__(self, original_model):
        """
        Function:
        Init Function.

        Args:
        original_model: torch.nn.Module, init the module helper.
                    Fetch each layer easily.
        """
        self.original_module_helper = ModuleHelper(original_model)

    @staticmethod
    def calc_torch_conv_flops(layer_module, shape_info):
        """
        Function: calc the flops for torch Conv2d.

        Parameters:
        shape_info: dict, {'input_shape': tensor.shape, 'output_shape': tensor.shape}

        Return:
        flops: float, FLOPs value.
        """
        output_shape = shape_info.get('output_shape')
        # batch_size, out_channel, out_h, out_w
        out_channel, out_h, out_w = output_shape[1], output_shape[2], output_shape[3]

        group = layer_module.groups

        weight_shape = layer_module.weight.shape
        in_channel, k_h, k_w = weight_shape[1], weight_shape[2], weight_shape[3]
        in_channel *= group

        # not layer_module.bias cause to boolean value of Tensor with more than one value is ambiguous
        has_bias = layer_module.bias is not None

        flops = ParamsHelper.calc_conv_flops(
            in_channel=in_channel,
            out_channel=out_channel,
            k_h=k_h,
            k_w=k_w,
            out_h=out_h,
            out_w=out_w,
            group=group,
            has_bias=has_bias
            )
        return flops

    @staticmethod
    def calc_torch_deconv_flops(layer_module, shape_info):
        """
        Function: calc the flops for torch ConvTranspose2d.

        Parameters:
        shape_info: dict, {'input_shape': tensor.shape, 'output_shape': tensor.shape}

        Return:
        flops: float, FLOPs value.
        """
        # get output shape
        output_shape = shape_info.get('output_shape')
        # batch_size, out_channel, out_h, out_w
        out_channel = output_shape[1]

        group = layer_module.groups

        # get input shape
        input_shape = shape_info.get('input_shape')
        in_channel, in_h, in_w = input_shape[1], input_shape[2], input_shape[3]

        # get weight shape
        weight_shape = layer_module.weight.shape
        k_h, k_w = weight_shape[2], weight_shape[3]

        has_bias = layer_module.bias is not None
        flops = ParamsHelper.calc_deconv_flops(
            in_channel=in_channel,
            out_channel=out_channel,
            k_h=k_h,
            k_w=k_w,
            in_h=in_h,
            in_w=in_w,
            group=group,
            has_bias=has_bias
            )
        return flops

    @staticmethod
    def calc_torch_linear_flops(layer_module, shape_info):
        """
        Function: calc the flops for torch Linear.

        Parameters:
        shape_info: dict, {'input_shape': tensor.shape, 'output_shape': tensor.shape}

        Return:
        flops: float, FLOPs value.
        """
        output_shape = shape_info.get('output_shape')
        input_shape = shape_info.get('input_shape')
        has_bias = layer_module.bias is not None
        flops = ParamsHelper.calc_matmul_flops(
            input_shape=input_shape,
            output_shape=output_shape,
            has_bias=has_bias
        )
        return flops

    @staticmethod
    def calc_torch_avgpool_flops(layer_module, shape_info):
        """
        Function: calc the flops for torch AvgPool2d.

        Parameters:
        shape_info: dict, {'input_shape': tensor.shape, 'output_shape': tensor.shape}

        Return:
        flops: float, FLOPs value.
        """
        output_shape = shape_info.get('output_shape')
        input_shape = shape_info.get('input_shape')

        out_channel, out_h, out_w = output_shape[1], output_shape[2], output_shape[3]
        in_channel = input_shape[1]
        group = in_channel

        if isinstance(layer_module.kernel_size, tuple):
            k_h, k_w = layer_module.kernel_size
        else:
            k_h = layer_module.kernel_size
            k_w = k_h

        flops = ParamsHelper.calc_conv_flops(
            in_channel=in_channel,
            out_channel=out_channel,
            k_h=k_h,
            k_w=k_w,
            out_h=out_h,
            out_w=out_w,
            group=group,
            has_bias=False
        )
        return flops

    def get_name_to_module(self, layer_name):
        """
        Function: get single module according to layer_name.

        Args:
        layer_name: string, model's layer name.
        """
        # if None, helper get_module raise error.
        module = self.original_module_helper.get_module(layer_name)
        return module

    def get_flops(self, layer_name, shape_infos):
        """
        Function: Get FLOPs for each layer.

        Args:
        layer_name: string, module's name in original_model.
        shape_infos: dict, contains `input_shape` and `output_shape` with multi-batch.
        {'input_shape': [shape0, shape1...],
        'output_shape': [shape0, shape1...]}

        Return:
        flops: float, mean of multi-batch.
        """
        mapping_func = {
            'Conv2d': 'calc_torch_conv_flops',
            'ConvTranspose2d': 'calc_torch_deconv_flops',
            'Linear': 'calc_torch_linear_flops',
            'AvgPool2d': 'calc_torch_avgpool_flops'
        }
        # the layer_module should has been check for not custom class before getting name.
        layer_module = self.get_name_to_module(layer_name)
        layer_type = type(layer_module).__name__
        if layer_type not in mapping_func.keys():
            raise ValueError("Layer [{}] [{}] do not support yet.".format(
                layer_name, layer_type
            ))

        flops_list = []
        for input_shape, output_shape in zip(shape_infos.get('input_shape'), shape_infos.get('output_shape')):
            shape_info = {'input_shape': input_shape, 'output_shape': output_shape}
            flops = getattr(ParamsHelperTorch, mapping_func.get(layer_type))(layer_module, shape_info)
            flops_list.append(flops)
        if not flops_list:
            raise ValueError('Layer [{}] [{}], invalid shape_infos {} to get FLOPs.'.format(
                layer_name, layer_type, shape_infos))
        flops = sum(flops_list) / len(flops_list)

        return flops
