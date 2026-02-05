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
from collections import namedtuple

import torch
import torch.nn as nn

from ...amct_pytorch.optimizer.base_module_fusion_pass \
    import BaseModuleFusionPass
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.nn.module.quantization.qat_base import QATBase
from ...amct_pytorch.custom_op.utils import tensor
from ...amct_pytorch.custom_op.utils import process_scale


REPLACE_DICT = {
    'Conv2dQAT': nn.Conv2d,
    'LinearQAT': nn.Linear,
}
BIAS = 'bias'

QuantFactor = namedtuple('QuantFactor', ['scale_d', 'offset_d', 'scale_w', 'offset_w'])


class DeleteQatPass(BaseModuleFusionPass):
    """
    Function: Delete QAT module about compressed quantization.
    APIs: match_pattern, do_pass
    """
    def __init__(self, record_helper):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseModuleFusionPass.__init__(self)
        self.record_helper = record_helper

    def match_pattern(self, module, name, graph=None):
        """
        Function:Match the QAT module compressed in model.
        Parameters:
            module: module to be matched
            name: module's name
            graph: graph structure, not necessary
        Return: True: matched
                False: mismatch
        """
        if not isinstance(module, QATBase):
            return False

        return True

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Delete QAT module about compressed quantization.
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        # generate record
        quant_factor = get_quant_factor(object_module)
        scale_d, offset_d, scale_w, offset_w = \
            quant_factor.scale_d, quant_factor.offset_d, quant_factor.scale_w, quant_factor.offset_w
        self.record_helper.record_activation_scale_offset(
            object_name, scale_d.cpu().tolist(), int(offset_d.cpu().tolist()))
        self.record_helper.record_weights_scale_offset(
            object_name, scale_w.cpu().tolist(), list(map(int, offset_w.cpu().tolist())))

        # generate ori op
        ori_op = generate_ori_op(object_module)

        # find module and its parent
        model_helper = ModuleHelper(model)
        parent_node = model_helper.get_parent_module(object_name)
        # replace new model
        setattr(parent_node, object_name.split('.')[-1], ori_op)

        LOGGER.logd(
            "Delete QAT module of '{}' success!".format(object_name), 'DeleteDistllPass')


def generate_ori_op(qat_op):
    """
    Function: get ori_op parameters from QAT module
    Parameters: 
        qat_op: QAT module
    Return: ori_op
    """
    ori_op_params = qat_op._get_ori_op_params(qat_op)
    if BIAS in ori_op_params:
        ori_op_params[BIAS] = ori_op_params.get(BIAS) is not None
    ori_op = REPLACE_DICT.get(type(qat_op).__name__)(**ori_op_params)
    setattr(ori_op, 'weight', qat_op.weight)
    setattr(ori_op, BIAS, qat_op.bias)
    ori_op.to(qat_op.weight.device)
    if not qat_op.training:
        ori_op.eval()
    return ori_op


def get_quant_factor(quant_module):
    """
    Function: get quant factor function.
    Inputs: 
        quant_module: QAT module
    Returns: 
        scale_d, offset_d, scale_w, offset_w
    """
    scale_d = quant_module.acts_scale.squeeze()
    offset_d = quant_module.acts_offset_deploy.to(scale_d.dtype).squeeze()
    scale_w = quant_module.wts_scales
    offset_w = quant_module.wts_offsets

    scale_d, offset_d = process_scale(
        scale_d, offset_d, True,
        quant_module.act_num_bits)
    scale_w, offset_w = process_scale(
        scale_w, offset_w, False,
        quant_module.wts_num_bits)
    return QuantFactor._make([scale_d, offset_d, scale_w, offset_w])
