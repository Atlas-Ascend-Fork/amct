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

import os
import torch
from ...amct_pytorch.common.utils import files as files_util
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.configuration.distill_config import parse_distill_config
from ...amct_pytorch.distill.distill_sample import ModelSingleTensorInput

LR = 'lr'
PARAMS = 'params'


class StopFowardException(Exception):
    '''set exception to stop the forward process'''
    def __init__(self, name):
        super().__init__()
        self.name = name


class DistillHelper():
    """ Helper to do Distill"""
    def __init__(self, model_t, model_s, config_file, loss, sample_instance):
        '''distill helper initialize'''
        self.config_file = DistillHelper.get_config_file(config_file)
        self.model_teacher, self.model_student = DistillHelper.get_models(model_t, model_s)
        self.config = parse_distill_config(self.config_file, self.model_teacher)
        self.loss = torch.nn.MSELoss() if loss is None else loss
        self.sample_helper = ModelSingleTensorInput() if sample_instance is None else sample_instance

    @property
    def batch_num(self):
        '''get batch number from config'''
        return self.config.get('batch_num', 1)

    @property
    def is_dump(self):
        '''get dump flag from config'''
        return self.config.get('data_dump', False)

    @property
    def distill_groups(self):
        '''get distill groups info from config'''
        return self.config.get('distill_group')

    @property
    def model_t(self):
        ''' teacher model '''
        return self.model_teacher

    @property
    def model_s(self):
        ''' student model '''
        return self.model_student

    @property
    def sample_ins(self):
        ''' sample '''
        return self.sample_helper

    @staticmethod
    def run_model_one_batch(model, samples):
        '''
        Function: run model forward once with one batch sample
        Inputs:
            model: torch model
            samples: model input data
        Outputs:
            result: None
        '''
        with torch.no_grad():
            try:
                if isinstance(samples, torch.Tensor):
                    model(samples)
                else:
                    model(*samples)
            except StopFowardException:
                pass

    @staticmethod
    def get_distill_modules(model, layer_names):
        """
        Function: get modules from model based on layer names
        Inputs:
            model: torch model
            layer_names: layer names
        Outputs:
            result: modules, list
        """
        modules = []
        model_helper = ModuleHelper(model)
        for name in layer_names:
            if model_helper.named_module_dict.get(name):
                mod = model_helper.get_module(name)
                modules.append(mod)
            else:
                raise RuntimeError('layer {} get module failed'.format(name))

        return modules

    @staticmethod
    def get_config_file(config_file):
        """
        Function: get config file if the config_file is valid
        Inputs:
            config_file: configuration file
        Outputs:
            result: file path, str
        """
        files_util.is_valid_name(config_file, 'config_file')
        config_file = os.path.realpath(config_file)
        if not os.path.exists(config_file):
            raise OSError('file ({}) does not exist!'.format(config_file))

        return config_file

    @staticmethod
    def gen_optimizer_per_group(distill_modules, optimizer, lr=1e-5):
        """Generate optimizer."""
        if optimizer is not None:
            opt_params_0 = []
            for module in distill_modules:
                for param in module.parameters():
                    opt_params_0.append(param)
            lr = optimizer.param_groups[0][LR]
            opt_gen = type(optimizer)(opt_params_0, lr=lr)
            return opt_gen

        opt_params_0 = []
        opt_params_act = []
        opt_params_wts = []
        for module in distill_modules:
            for name, param in module.named_parameters():
                if name in ["acts_clip_max", "acts_clip_min"]:
                    # activation quant
                    opt_params_act.append(param)
                elif name in ['wts_scales', 'wts_offsets']:
                    # wts quant
                    opt_params_wts.append(param)
                else:
                    # other param
                    opt_params_0.append(param)

        opt_params = []
        if len(opt_params_0) != 0:
            opt_params.append({PARAMS: opt_params_0})
        if len(opt_params_act) != 0:
            opt_params.append({PARAMS: opt_params_act, LR: 0.1})
        if len(opt_params_wts) != 0:
            opt_params.append({PARAMS: opt_params_wts, LR: 0.0001})
        if len(opt_params) == 0:
            raise RuntimeError('gen default optimzer get no optimize param')

        optimizer = torch.optim.AdamW(opt_params, lr=lr)
        LOGGER.logd('generate default AdamW optimizer')
        return optimizer

    @staticmethod
    def get_models(model_t, model_s):
        '''get model_t and models if the models are valid'''
        if isinstance(model_t, (torch.nn.parallel.DistributedDataParallel,)):
            model_t = model_t.module

        if isinstance(model_s, (torch.nn.parallel.DistributedDataParallel,)):
            model_s = model_s.module

        ModuleHelper(model_t).check_amct_op()
        ModuleHelper(model_s).check_amct_distill_op()

        return model_t, model_s

    def get_distill_modules_loss(self, modules, input_data, target):
        """
        Function: get modules loss value
        Inputs:
            modules: modules
            input_data: modules input data
            target: modules output target
        Outputs:
            result: loss value
        """
        with torch.enable_grad():
            x = input_data
            for module in modules:
                try:
                    x = module.forward(x)
                except Exception as exception:
                    raise RuntimeError(
                        'module forward failed, please check the distill group config') from exception
            if x.numel() != target.numel():
                raise RuntimeError(
                    'shape error, please check train_loader, x {}, target {}'
                    .format(x.numel(), target.numel()))

            # broadcast the target shape
            target = target.reshape(x.shape)
            loss_val = self.loss(x, target)
        return loss_val

    def do_calibration(self, train_loader):
        ''' do calibration for ifmr '''
        if len(train_loader) == 0:
            raise ValueError('train_loader length is 0, please check the train_loader')

        run_batch = 0
        while run_batch < self.batch_num:
            for samples in train_loader:
                model_input = self.get_model_input(samples)
                DistillHelper.run_model_one_batch(self.model_s, model_input)
                run_batch = run_batch + 1
                if run_batch >= self.batch_num:
                    LOGGER.logi(
                        'distill do calibration success. batch num: {}'.format(self.batch_num))
                    return

    def get_model_input(self, samples):
        '''get model input samples, only support tensor and tuple'''
        sp = self.sample_helper.get_model_input_data(samples)
        if isinstance(sp, (list, tuple)):
            if not all(isinstance(x, torch.Tensor) for x in sp):
                raise RuntimeError('expect tensor in list/tuple, please check the sample_instance')
            return sp
        elif isinstance(sp, torch.Tensor):
            return sp

        raise RuntimeError('expect tensor/list/tuple, but got {}, please check the sample_instance'.format(type(sp)))
