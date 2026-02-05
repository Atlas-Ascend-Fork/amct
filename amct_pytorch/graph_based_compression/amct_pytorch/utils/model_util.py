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
import copy
from collections.abc import Iterable

import torch

from ...amct_pytorch.common.utils.util import version_higher_than
from ...amct_pytorch.utils.vars import AMCT_OPERATIONS
from ...amct_pytorch.utils.vars import AMCT_RETRAIN_OPERATIONS
from ...amct_pytorch.utils.vars import AMCT_DISTILL_OPERATIONS
from ...amct_pytorch.configuration.configuration import Configuration


class ModuleHelper():
    """
    Funtion: Helper for torch.nn.module
    APIS: get_module, get_parent_module
    """
    def __init__(self, model):
        ''' init function '''
        self.named_module_dict = {}
        for name, mod in model.named_modules():
            self.named_module_dict[name] = mod

    @staticmethod
    def get_name_type_dict(model):
        '''get all layer name to type dict'''
        layer_type = {}
        for name, mod in model.named_modules():
            layer_type[name] = type(mod).__name__
        return layer_type

    @staticmethod
    def deep_copy(model):
        """deepcopy a model """
        try:
            new_model = copy.deepcopy(model)
            return new_model
        except Exception as exception:
            raise RuntimeError("The model cannot do copy.deepcopy, "
                               "exception is: {}".format(exception)) from exception

    @staticmethod
    def replace_module_by_name(model, name, mod):
        """ replace module in model by a new mod according to module name """
        tokens = name.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], mod)

    def get_module(self, name):
        ''' get a module from name'''
        module = self.named_module_dict.get(name)
        if module is None:
            raise RuntimeError(f"module {name} not in model.")
        return module

    def get_parent_module(self, name):
        ''' get parent module from name'''
        if name == '':
            raise RuntimeError("model has no parent module.")

        parent_name = '.'.join(name.split('.')[:-1])
        parent_module = self.named_module_dict.get(parent_name)
        if parent_module is None:
            raise RuntimeError(f"module {parent_name} not in model.")
        return parent_module

    def check_amct_op(self):
        """ Check all the layers whose type is in AMCT_OPERATIONS in model
        """
        amct_layers = {}
        for name in self.named_module_dict:
            mod_type = type(self.named_module_dict[name]).__name__
            if mod_type in AMCT_OPERATIONS:
                amct_layers[name] = mod_type

        if amct_layers:
            raise RuntimeError("The model cannot be quantized for following "\
                "quant layers are in the model {}".format(amct_layers))

    def check_amct_retrain_op(self):
        """ Check all the layers whose type is in AMCT_RETRAIN_OPERATIONS in
            model
        """
        amct_retrain_layers = {}
        for name in self.named_module_dict:
            mod_type = type(self.named_module_dict[name]).__name__
            if mod_type in AMCT_RETRAIN_OPERATIONS:
                amct_retrain_layers[name] = mod_type

        if not amct_retrain_layers:
            raise RuntimeError("The model cannot be quantized because it is "\
                "not a model after quantitative retraining.")

    def check_amct_distill_op(self):
        """ Check all the layers whose type is in AMCT_DISTILL_OPERATIONS in
            model
        """
        amct_distill_layers = {}
        for name in self.named_module_dict:
            mod_type = type(self.named_module_dict[name]).__name__
            if mod_type in AMCT_DISTILL_OPERATIONS:
                amct_distill_layers[name] = mod_type

        if not amct_distill_layers:
            raise RuntimeError("The model cannot be distill because it is "\
                "not a model after compressing.")


def load_pth_file(model, pth_file, state_dict_name):
    """
    Function: load pth file to model
    Inputs:
        model: user model
        pth_file: parameters file
        state_dict_name: parameters state name
    Outputs:
        model: model loaded parameters
    """
    load_kwargs = {'map_location': torch.device('cpu')}
    if version_higher_than(torch.__version__, '2.1.0'):
        load_kwargs['weights_only'] = False
    checkpoint = torch.load(pth_file, **load_kwargs)
    if state_dict_name:
        if checkpoint.get(state_dict_name):
            state_dict = checkpoint[state_dict_name]
        else:
            raise KeyError("The pth_file has no key name: "
                           "{}".format(state_dict_name))
    else:
        state_dict = checkpoint
    is_pth_parallel = all((state.startswith('module.') for state in state_dict.keys()))
    is_model_parallel = isinstance(model, (torch.nn.parallel.DistributedDataParallel,))

    if not is_pth_parallel and is_model_parallel:
        model.module.load_state_dict(state_dict)
    elif is_pth_parallel and not is_model_parallel:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[key[7:]] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    return model


def get_node_output_info(model, input_data):
    """
    Function: get node output's dtype in model through forward
    Inputs:
        model: user model
        input_data: data used in forward
    Outputs:
        result: node output's dtype in model
    """
    copied_model = copy.deepcopy(model)

    def hook_fn(module, inputs, output):
        if isinstance(output, torch.Tensor):
            output_dtype[module] = output.dtype
        elif isinstance(output, Iterable):
            output_dtype[module] = output[0].dtype

    output_dtype = dict()
    names_module_map = dict()
    config = Configuration()
    for name, module in copied_model.named_modules():
        if name not in config.get_quant_config():
            continue
        names_module_map[name] = module
        module.register_forward_hook(hook_fn)
    if isinstance(input_data, torch.Tensor):
        copied_model.forward(input_data)
    elif isinstance(input_data, tuple):
        if isinstance(input_data[-1], dict):
            copied_model.forward(*input_data[:-1], **input_data[-1])
        else:
            copied_model.forward(*input_data)
    result = dict()
    for idx, name in enumerate(names_module_map):
        module = names_module_map.get(name)
        if module not in output_dtype:
            continue
        result[name] = list()
        attr_dtype = dict()
        attr_dtype['attr_name'] = 'op_data_type'
        attr_dtype['attr_type'] = 'STRING'
        attr_dtype['attr_val'] = bytes('float16' if output_dtype[module] is torch.float16 else 'float',
                                       encoding='utf-8')
        result[name].append(attr_dtype)
    return result
