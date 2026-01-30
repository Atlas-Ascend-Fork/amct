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

utils for onnx conv

"""
from amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from amct_pytorch.utils.quant_node import QuantOpInfo


class OnnxConvUtil:
    """
    Function: Utils for onnx conv or deconv node
    APIs: is_depthwise_conv, is_group_conv, get_group, get_depthwise_multiplier
    """
    def __init__(self, node):
        """
        Function: init function
        param:node
        Return:None
        """
        self.node = node
        self.check()

    def check(self):
        """
        Function: check whether the node is conv
        param: none
        Return: None
        """
        if self.node.type not in ['Conv', 'ConvTranspose']:
            raise RuntimeError('Only support [Conv, ConvTranspose] but {} is {}'
                                .format(self.node.name, self.node.type))

    def is_depthwise_conv(self):
        """
        Function: check if a node is depthwise conv
        Return: bool, True if the node is depthwise.
        """
        filter_shape = QuantOpInfo.get_weight_tensor(self.node).dims
        if filter_shape[1] != 1:
            return False
        return True

    def is_group_conv(self):
        """
        Function: check if a node is group conv, not including depthwise conv
        Return: bool, True if the node is group conv.
        """
        # not support group is 1
        group = self.get_group()
        if group == 1:
            return False
        filter_shape = QuantOpInfo.get_weight_tensor(self.node).dims
        if filter_shape[1] == 1:
            return False
        return True

    def get_group(self):
        """
        Function: get group of the node.
        Return: group.
        """
        try:
            group = AttributeProtoHelper(self.node.proto).get_attr_value('group')
        except RuntimeError:
            group = 1
        return group

    def get_depthwise_multiplier(self):
        """
        Function: get depthwise multiplier of the node.
        Return: depthwise multiplier of the node.
        """
        group = self.get_group()
        filter_shape = QuantOpInfo.get_weight_tensor(self.node).dims
        if filter_shape[0] % group != 0:
            raise RuntimeError('unexpected depthwise node {}, filter_shape is {} and group is {}.'
                               .format(self.node.name, filter_shape, group))
        multiplier =  filter_shape[0] // group
        return multiplier
