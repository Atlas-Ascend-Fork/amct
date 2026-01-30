#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

distill data sample process

"""
import torch


class DistillSampleBase():
    @staticmethod
    def get_model_input_data(samples):
        '''
        Function: get model's input data, user needs to complete the implementation
        Parameter: samples: samples from dataloader
        Return: model's input data, list, tuple, torch.Tensor
        '''
        raise NotImplementedError("get_model_input_data needs to be implemented for distill!")


class ModelSingleTensorInput(DistillSampleBase):
    @staticmethod
    def get_model_input_data(samples):
        '''default model input data: single torch.Tensor'''
        single_tensor = samples
        if isinstance(samples, (list, tuple)):
            if len(samples) == 0:
                raise ValueError('default sample instance get empty sample,'
                    ' please check the distill sample func')
            single_tensor = samples[0]

        if not isinstance(single_tensor, torch.Tensor):
            raise ValueError('default sample instance get no tensor value,'
                ' please check the train_loader')

        return single_tensor
