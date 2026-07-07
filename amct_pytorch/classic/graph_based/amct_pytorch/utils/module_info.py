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
import torch

from ...amct_pytorch.utils.log import LOGGER

WEIGHT_PARAM_NAME = 'weight'


class ModuleInfo:
    """ help to find some features for module from torch"""
    @staticmethod
    def get_wts_cout_cin(module):
        """ get weight's cout axis and cin axis"""
        info_map = {
            'Conv2d': [0, 1],
            'Conv3d': [0, 1],
            'Linear': [0, 1],
            'ConvTranspose1d': [1, 0],
            'ConvTranspose2d': [1, 0],
            'ConvTranspose3d': [1, 0],
            'Conv1d': [0, 1],
        }
        support_type = info_map.keys()
        mod_type = type(module).__name__
        if mod_type not in support_type:
            raise TypeError('Invalid module type {}, only support {}'.format(mod_type, support_type))

        cout_axis, cin_axis = info_map.get(mod_type)

        return cout_axis, cin_axis

    @staticmethod
    def get_param_tensor(module, param_name):
        """ get param' tensor, param's name is param_name"""
        param_obj = getattr(module, param_name)
        if param_obj is None:
            return None
        # Parameter is also Tensor, judge it at first
        if isinstance(param_obj, torch.nn.parameter.Parameter):
            return param_obj.data
        if isinstance(param_obj, torch.Tensor):
            return param_obj

        raise RuntimeError('Unexpected type {} to get {}'.format(type(param_obj), param_name))

    @staticmethod
    def set_param_tensor(module, param_name, data_tensor):
        """ set param' tensor as data_tensor, param's name is param_name"""
        param_obj = getattr(module, param_name)
        if isinstance(param_obj, torch.nn.parameter.Parameter):
            param_obj.data = data_tensor
            return
        if isinstance(param_obj, torch.Tensor):
            setattr(module, param_name, data_tensor)
            return

        raise RuntimeError('Unexpected type {} to set {}'.format(type(param_obj), param_name))


def create_prune_helper(module):
    """ create helper to prune a module """
    if Conv2dModulePruneHelper.match_pattern(module):
        return Conv2dModulePruneHelper(module)

    if ConvTransposeModulePruneHelper.match_pattern(module):
        return ConvTransposeModulePruneHelper(module)

    if LinearModulePruneHelper.match_pattern(module):
        return LinearModulePruneHelper(module)

    if BnModulePruneHelper.match_pattern(module):
        return BnModulePruneHelper(module)

    raise RuntimeError('Not support module type {} '.format(type(module)))


class ModulePruneHelper:
    """helper to prune a module """
    param_cout_cin = {}
    support_type = []

    def __init__(self, module):
        """ init function"""
        self.module = module
        self.backup = {}
        self.cin_prune_list = None
        self.cout_prune_list = None

    @staticmethod
    def restore(module, backup, module_name):
        """
        Function: restore module to eliminate prune effects according to backup
        param:module, torch.nn.Module
        backup:a dict, param and feature to restore
        module_name:string, name of module
        """
        for name, value in backup.items():
            if not hasattr(module, name):
                raise RuntimeError("module {} doesn't have {}".format(module_name, name))
            if isinstance(value, torch.Tensor):
                ModuleInfo.set_param_tensor(module, name, value)
            else:
                setattr(module, name, value)
    
    @classmethod
    def match_pattern(cls, module):
        """
        Function: Match pattern of module to do prune
        Parameter: None
        Return: bool, matched or not
        """
        mod_type = type(module).__name__
        if mod_type not in cls.support_type:
            return False
        return True

    def do_prune(self, cin_prune_list, cout_prune_list):
        """do prune """
        self.cin_prune_list = cin_prune_list
        self.cout_prune_list = cout_prune_list
        self.modify_param()
        self.modify_attr()
        return self.backup

    def modify_param(self):
        """modify module's param"""
        object_module = self.module
        cin_prune_list = self.cin_prune_list
        cout_prune_list = self.cout_prune_list
        param_cout_cin = self.param_cout_cin

        for param_name in param_cout_cin:
            data_tensor = ModuleInfo.get_param_tensor(object_module, param_name)
            if data_tensor is None:
                continue
            self.backup[param_name] = data_tensor
            cout_axis, cin_axis = param_cout_cin[param_name]
            if cin_axis is not None and cin_prune_list:
                data_tensor = prune_tensor_channel(data_tensor, cin_axis, cin_prune_list)
                ModuleInfo.set_param_tensor(object_module, param_name, data_tensor)
            if cout_axis is not None and cout_prune_list:
                data_tensor = prune_tensor_channel(data_tensor, cout_axis, cout_prune_list)
                ModuleInfo.set_param_tensor(object_module, param_name, data_tensor)

    def modify_attr(self):
        """modify module's attr"""
        pass


class Conv2dModulePruneHelper(ModulePruneHelper):
    """helper to prune a conv module """
    support_type = ['Conv2d', ]
    param_cout_cin = {
        'weight': [0, 1],
        'bias': [0, None]}

    def is_depthwise(self):
        """whether module is depthwise or not"""
        if self.module.groups == 1:
            return False
        if self.module.groups != self.module.in_channels:
            return False

        return True

    def is_group_conv(self):
        """whether module is group conv or not"""
        if self.module.groups == 1:
            return False
        if self.module.groups == self.module.in_channels:
            return False

        return True

    def modify_param(self):
        """
        Function: modify module's param
        param: None
        return: None
        """
        if self.is_depthwise() and self.cin_prune_list:
            cin_prune_list = []
            cout_prune_list = self.cin_prune_list
        else:
            cin_prune_list = self.cin_prune_list
            cout_prune_list = self.cout_prune_list
        param_cout_cin = self.param_cout_cin

        for param_name in param_cout_cin:
            data_tensor = ModuleInfo.get_param_tensor(self.module, param_name)
            if data_tensor is None:
                continue
            self.backup[param_name] = data_tensor
            self._modify_param_cout(param_name, cout_prune_list)
            self._modify_param_cin(param_name, cin_prune_list)

    def modify_attr(self):
        """modify module's attr"""
        if self.is_depthwise():
            self.backup['groups'] = self.module.groups
            self.module.groups -= len(self.cin_prune_list)
        self.backup['in_channels'] = self.module.in_channels
        self.backup['out_channels'] = self.module.out_channels
        self.module.in_channels -= len(self.cin_prune_list)
        self.module.out_channels -= len(self.cout_prune_list)

    def _modify_param_cout(self, param_name, cout_prune_list):
        """
        Function: modify module's param on cout
        param: None
        return: None
        """
        data_tensor = ModuleInfo.get_param_tensor(self.module, param_name)
        cout_axis, _ = self.param_cout_cin.get(param_name)
        if cout_axis is not None and cout_prune_list:
            data_tensor = prune_tensor_channel(data_tensor, cout_axis, cout_prune_list)
            ModuleInfo.set_param_tensor(self.module, param_name, data_tensor)

    def _modify_param_cin(self, param_name, cin_prune_list):
        """
        Function: modify module's param on cin
        param: None
        return: None
        """
        data_tensor = ModuleInfo.get_param_tensor(self.module, param_name)
        _, cin_axis = self.param_cout_cin.get(param_name)
        if cin_axis is not None and cin_prune_list:
            if self.is_group_conv() and param_name == WEIGHT_PARAM_NAME:
                data_tensor = trans_shape(data_tensor, self.module.groups)
            data_tensor = prune_tensor_channel(data_tensor, cin_axis, cin_prune_list)
            if self.is_group_conv() and param_name == WEIGHT_PARAM_NAME:
                data_tensor = restore_shape(data_tensor, self.module.groups)
            ModuleInfo.set_param_tensor(self.module, param_name, data_tensor)


class ConvTransposeModulePruneHelper(ModulePruneHelper):
    """helper to prune a deconv module """
    support_type = ['ConvTranspose2d', ]
    param_cout_cin = {
        'weight': [1, 0],
        'bias': [0, None]}

    def is_depthwise(self):
        """whether module is depthwise or not"""
        if self.module.groups == 1:
            return False
        if self.module.groups != self.module.out_channels:
            return False

        return True

    def is_group_conv(self):
        """whether module is group conv or not"""
        if self.module.groups == 1:
            return False
        if self.module.groups == self.module.out_channels:
            return False

        return True

    def modify_param(self):
        """modify module's param"""
        object_module = self.module
        cin_prune_list = self.cin_prune_list
        cout_prune_list = self.cout_prune_list
        param_cout_cin = self.param_cout_cin

        for param_name in param_cout_cin:
            data_tensor = ModuleInfo.get_param_tensor(object_module, param_name)
            if data_tensor is None:
                continue
            self.backup[param_name] = data_tensor
            cout_axis, cin_axis = param_cout_cin[param_name]
            if cin_axis is not None and cin_prune_list:
                data_tensor = prune_tensor_channel(data_tensor, cin_axis, cin_prune_list)
                ModuleInfo.set_param_tensor(object_module, param_name, data_tensor)
            if cout_axis is not None and cout_prune_list:
                if self.is_depthwise() and param_name == WEIGHT_PARAM_NAME:
                    continue
                if self.is_group_conv() and param_name == WEIGHT_PARAM_NAME:
                    data_tensor = trans_shape(data_tensor, object_module.groups)
                data_tensor = prune_tensor_channel(data_tensor, cout_axis, cout_prune_list)
                if self.is_group_conv() and param_name == WEIGHT_PARAM_NAME:
                    data_tensor = restore_shape(data_tensor, object_module.groups)
                ModuleInfo.set_param_tensor(object_module, param_name, data_tensor)

    def modify_attr(self):
        """modify module's attr"""
        if self.is_depthwise():
            self.backup['groups'] = self.module.groups
            self.module.groups -= len(self.cout_prune_list)
        self.backup['in_channels'] = self.module.in_channels
        self.backup['out_channels'] = self.module.out_channels
        self.module.in_channels -= len(self.cin_prune_list)
        self.module.out_channels -= len(self.cout_prune_list)


class LinearModulePruneHelper(ModulePruneHelper):
    """helper to prune a Linear module """
    support_type = ['Linear', ]
    param_cout_cin = {
        'weight': [0, 1],
        'bias': [0, None]}

    def modify_attr(self):
        """modify module's attr"""
        self.backup['in_features'] = self.module.in_features
        self.backup['out_features'] = self.module.out_features
        self.module.in_features -= len(self.cin_prune_list)
        self.module.out_features -= len(self.cout_prune_list)


class BnModulePruneHelper(ModulePruneHelper):
    """helper to prune a bn module """
    support_type = ['BatchNorm1d', 'BatchNorm2d']
    param_cout_cin = {
        'weight': [0, 0],
        'bias': [0, 0],
        'running_mean': [0, 0],
        'running_var': [0, 0]}

    def modify_attr(self):
        """modify module's attr"""
        self.backup['num_features'] = self.module.num_features
        self.module.num_features -= len(self.cin_prune_list)


def prune_tensor_channel(tensor, axis, prune_channel):
    """prune tensor in axis according prune_channel"""
    pre_shape = tensor.shape
    remain_channel = [idx for idx in range(pre_shape[axis]) if idx not in prune_channel]
    new_tensor = torch.index_select(tensor, dim=axis, index=torch.LongTensor(remain_channel).to(tensor.device))
    LOGGER.logd(
        'After Prune(axis: {}, prune_channel: len is {} and val is {}), tensor shape has been changed from {} to {}'
        .format(axis, len(prune_channel), prune_channel, pre_shape, new_tensor.shape), 'PruneModelPass')
    return new_tensor


def trans_shape(wts_tensor, group):
    """ trans tensor's shape to ignore group"""
    ori_shape = np.array(wts_tensor.shape).tolist()
    ori_shape.insert(0, group)
    ori_shape[1] = ori_shape[1] // group
    # g, cin/g, cout/g, h, w
    wts_tensor = wts_tensor.reshape(ori_shape)
    # cin/g, g, cout/g, h, w
    wts_tensor = wts_tensor.transpose(0, 1)
    ori_shape[2] = int(ori_shape[0] * ori_shape[2])
    del ori_shape[0]
    # cin/g, cout, h, w
    wts_tensor = wts_tensor.reshape(ori_shape)
    return wts_tensor


def restore_shape(wts_tensor, group):
    """ restore tensor's shape to considerate group"""
    # cin/g, cout, h, w
    ori_shape = np.array(wts_tensor.shape).tolist()
    ori_shape.insert(1, group)
    ori_shape[2] = ori_shape[2] // group
    # cin/g, g, cout/g, h, w
    wts_tensor = wts_tensor.reshape(ori_shape)
    # g, cin/g, cout/g, h, w
    wts_tensor = wts_tensor.transpose(0, 1)
    ori_shape[1] = int(ori_shape[0] * ori_shape[1])
    del ori_shape[0]
    # cin/g, cout, h, w
    wts_tensor = wts_tensor.reshape(ori_shape)
    return wts_tensor
