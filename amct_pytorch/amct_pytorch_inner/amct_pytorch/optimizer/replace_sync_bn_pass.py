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
import torch

from ...amct_pytorch.optimizer.base_module_fusion_pass \
    import BaseModuleFusionPass
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.log import LOGGER


class RepalceSyncBNPass(BaseModuleFusionPass):
    """
    Function: Replace the synchronized BN with a normal BN.
    APIs: match_pattern, do_pass
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseModuleFusionPass.__init__(self)

    @staticmethod
    def match_pattern(module, name, graph=None):
        """
        Function:Match the SyncBatchNorm module in model.
        Parameters:
            module: module to be matched
            name: module's name
            graph: graph structure, not necessary
        Return: True: matched
                False: mismatch
        """
        if not isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
            return False

        return True

    @staticmethod
    def do_pass(model, object_module, object_name, graph=None):
        """
        Function: Replace the synchronized BN with a normal BN.
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        # Step1: find module and its parent
        model_helper = ModuleHelper(model)
        parent_node = model_helper.get_parent_module(object_name)

        # Step2: create replace node
        replace_node = torch.nn.BatchNorm2d(
            object_module.num_features,
            object_module.eps, object_module.momentum,
            object_module.affine,
            object_module.track_running_stats)
        if object_module.affine:
            with torch.no_grad():
                replace_node.weight.copy_(object_module.weight)
                replace_node.bias.copy_(object_module.bias)
            # keep requires_grad unchanged
            replace_node.weight.requires_grad = \
                object_module.weight.requires_grad
            replace_node.bias.requires_grad = object_module.bias.requires_grad
        replace_node.running_mean = object_module.running_mean
        replace_node.running_var = object_module.running_var
        replace_node.num_batches_tracked = object_module.num_batches_tracked
        replace_node.training = object_module.training
        replace_node = \
            replace_node.to(next(object_module.parameters()).device)

        # Step3: replace new model
        setattr(parent_node, object_name.split('.')[-1], replace_node)

        LOGGER.logd(
            "Replace Sync BatchNorm with BatchNorm2d({}) "
            "successfully!".format(object_name), 'RepalceSyncBNPass')
