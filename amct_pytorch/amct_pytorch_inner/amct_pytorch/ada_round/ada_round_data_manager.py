#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ada round data manager

"""
import torch

from amct_pytorch.utils.model_util import ModuleHelper


class StopFowardException(Exception):
    '''set exception to stop the forward process'''
    def __init__(self, name):
        super().__init__()
        self.name = name


class AdaRoundDataManager:
    '''
    Function: manage the ada round data.
    APIs: get_input_data, get_module_output_data
    '''
    def __init__(self, model):
        self.model = model

    def _register_input_hook(self, input_data, name):
        ''' hooker for cache layer's input data'''
        def hook(module, inputs, outputs):
            input_data.append(inputs[0].detach())
            raise StopFowardException(name)
        return hook

    def _register_output_hook(self, output_data, name):
        ''' hooker for cache layer's output data'''
        def hook(module, inputs, outputs):
            output_data.append(outputs.detach())
            raise StopFowardException(name)
        return hook

    def get_input_data(self, model_input, module_name):
        '''
        Function: get module input data through inference
        Parameters:
            model_input: model input
            module_name: name of module which you want to get input data
        Return:
            module input of module_name
        '''
        module_helper = ModuleHelper(self.model)
        module = module_helper.get_module(module_name)
        input_data = []
        hook = module.register_forward_hook(self._register_input_hook(input_data, module_name))
        try:
            with torch.no_grad():
                if isinstance(model_input, tuple):
                    self.model.forward(*model_input)
                else:
                    self.model.forward(model_input)
        except StopFowardException:
            pass
        hook.remove()
        return input_data[0]

    def get_output_data(self, model_input, module_name):
        '''
        Function: get module output data through inference
        Parameters:
            model_input: model input
            module_name: name of module which you want to get output data
        Return:
            module output of module_name
        '''
        module_helper = ModuleHelper(self.model)
        module = module_helper.get_module(module_name)
        output_data = []
        hook = module.register_forward_hook(self._register_output_hook(output_data, module_name))
        try:
            if isinstance(model_input, tuple):
                self.model.forward(*model_input)
            else:
                self.model.forward(model_input)
        except StopFowardException:
            pass
        hook.remove()
        module_output = output_data[0]
        return module_output