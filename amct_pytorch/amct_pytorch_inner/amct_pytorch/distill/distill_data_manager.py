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
import numpy as np

from ...amct_pytorch.common.utils.files import create_path
from ...amct_pytorch.common.utils.files import delete_dir
from ...amct_pytorch.common.utils.files import check_files_exist
from ...amct_pytorch.distill.distill_helper import DistillHelper
from ...amct_pytorch.distill.distill_helper import StopFowardException
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.model_util import ModuleHelper


TMP_PATH = 'tmp'
DATA_DUMP_INPUT_PREFIX = 'data_dump_input_'
DATA_DUMP_OUTPUT_PREFIX = 'data_dump_output_'


class DistillDataManager():
    '''
    Function: mange the distill data.
    APIs: cache_input_data_hooker, get_norm_min_data,
          get_output_data_by_inferring, get_input_data_by_inferring
          dump_data, release
          load_model_input_dump_data, load_input_dump_data, load_output_dump_data
    '''
    def __init__(self, sample_instance):
        ''' distill data manager initialize
            is_dump: dump data or not, bool
            delete_dump_path: need delete tmp path or not, bool
            dump_files: dump file path
            device: cpu/gpu, torch.device
            device_name: device name
        '''
        self.is_dump = False
        self.delete_dump_path = False
        self.dump_files = []
        self.device = torch.device('cpu')
        self.device_name = 'cpu'
        self.sample_ins = sample_instance
        self.model_input_files = {}

    @staticmethod
    def cache_input_data_hooker(data, name):
        ''' hooker for cache layer's input data'''
        def hook(module, inputs, outputs):
            data.append(inputs[0].detach())
            raise StopFowardException(name)
        return hook

    @staticmethod
    def get_norm_min_data(data_t, data_s):
        """
        Function: get layer's output data based on inferring model
        Inputs:
            data_t: data input 0
            data_s: data input 1
        Outputs:
            result: input data, list
        """
        norm_t = (data_t**2).mean()
        norm_s = (data_s**2).mean()
        norm_diff = ((data_t - data_s)**2).mean()

        if norm_diff > min(norm_t, norm_s) * 0.5:
            LOGGER.logd('data diff is larger than threshold', 'DistillDataManager')
            return data_t
        else:
            return data_s

    @staticmethod
    def get_output_data_by_inferring(model, group, sample):
        """
        Function: get layer's output data based on inferring model
        Inputs:
            model: model
            group: group layers
            model_input: input data of the model
        Outputs:
            result: output data, tensor
        """
        modules = DistillHelper.get_distill_modules(model, group)
        x = sample
        with torch.no_grad():
            for module in modules:
                try:
                    x = module.forward(x)
                except RuntimeError as e:
                    raise RuntimeError('module forward failed, exception is: {}'.format(e)) from e

        return x

    def dump_data(self, model, groups, epochs, data_loader):
        """
        Function: dump data to file for each epoch/step
        Inputs:
            model: model
            groups: group layers to dump
            epochs: epoch num
            data_loader: DataLoader for model
        Outputs:
            None
        """
        if not os.path.exists(TMP_PATH):
            create_path(TMP_PATH)
            self.delete_dump_path = True
            LOGGER.logd('create tmp path success.', 'DistillDataManager')

        for epoch in range(epochs):
            for step, samples in enumerate(data_loader):
                model_input = self.sample_ins.get_model_input_data(samples)
                # dumpp model input data to file
                self._dump_model_input(epoch, step, model_input)

                # dump input data to file
                self._dump_group_input_data(model, groups, epoch, step, model_input)

                # dump output data to file
                self._dump_group_output_data(model, groups, epoch, step)

        self.is_dump = True
        LOGGER.logi('dump data success.', 'DistillDataManager')

    def release(self):
        '''remove tmp file/dir, restore cls member'''
        for file in self.dump_files:
            os.remove(file)
        if self.delete_dump_path:
            delete_dir(TMP_PATH)
        self.delete_dump_path = False
        self.is_dump = False

    def load_model_input_dump_data(self, epoch, step):
        """
        Function: load model input data from dump file
        Inputs:
            epoch: epoch index
            step: step index
        Outputs:
            result: model input data, tensor
        """
        file_key = self._gen_file_path('model_input', epoch, step)
        model_input = []
        for file_name in self.model_input_files.get(file_key):
            if not os.path.exists(file_name):
                raise RuntimeError('model input file not exists. {}'.format(file_name))
            sample = torch.tensor(np.load(file_name))
            model_input.append(sample.to(self.device))

        return model_input

    def load_input_dump_data(self, group, epoch, step):
        """
        Function: load input data from dump file
        Inputs:
            group: group layers
            epoch: epoch index
            step: step index
        Outputs:
            result: input data, tensor
        """
        file_name = self._gen_file_path(group[0], epoch, step)
        if not os.path.exists(file_name):
            raise RuntimeError('group input file not exists. {}'.format(file_name))

        in_tensor = torch.tensor(np.load(file_name))
        return in_tensor.to(self.device)

    def load_output_dump_data(self, group, epoch, step):
        """
        Function: load output data from dump file
        Inputs:
            group: target layer
            epoch: epoch index
            step: step index
        Outputs:
            result: output data, tensor
        """
        file_name = self._gen_file_path(group[0], epoch, step, is_in=False)
        if not os.path.exists(file_name):
            raise RuntimeError('group output file not exists. {}'.format(file_name))

        out_tensor = torch.tensor(np.load(file_name))
        return out_tensor.to(self.device)

    def get_input_data_by_inferring(self, model, group, sample):
        """
        Function: get layer's input data based on inferring model
        Inputs:
            model: model
            group: group layers
            model_input: input data of the model
        Outputs:
            result: input data, list
        """
        tensor = []
        module_helper = ModuleHelper(model)
        module = module_helper.get_module(group[0])
        hook = module.register_forward_hook(self.cache_input_data_hooker(tensor, group[0]))
        DistillHelper.run_model_one_batch(model, sample)
        hook.remove()

        if len(tensor) == 0:
            raise RuntimeError('layer {} get input data failed'.format(group[0]))

        return tensor[0]

    def _dump_model_input(self, epoch, step, samples):
        '''dump model's input data file'''
        multi_file = []
        file_key = self._gen_file_path('model_input', epoch, step)
        if isinstance(samples, torch.Tensor):
            file_name = self._gen_file_path('model_input', epoch, step)
            np.save(file_name, samples.cpu())
            multi_file = [file_name]
        else:
            for index, sample in enumerate(samples):
                file_name = self._gen_file_path(f'model_input_{index}', epoch, step)
                np.save(file_name, sample.cpu())
                multi_file.append(file_name)

        self.model_input_files[file_key] = multi_file
        self.dump_files.extend(multi_file)
        LOGGER.logd('dump model input data success.', 'DistillDataManager')

    def _dump_input_data_hooker(self, layer_name, epoch, step):
        ''' hooker for dump layer's input data'''
        def hook(module, inputs, outputs):
            file_name = self._gen_file_path(layer_name, epoch, step)
            np.save(file_name, inputs[0].cpu())
            self.dump_files.append(file_name)
            LOGGER.logd('dump layer {} input data success.'.format(layer_name), 'DistillDataManager')

        return hook

    def _dump_group_input_data(self, model, groups, epoch, step, sample):
        ''' dump group's input data to file'''
        hooks = []
        module_helper = ModuleHelper(model)
        # regist hook function, for each epoch & step
        for group in groups:
            input_layer = module_helper.get_module(group[0])
            hook_in = input_layer.register_forward_hook(
                self._dump_input_data_hooker(group[0], epoch, step))
            hooks.append(hook_in)
        # forward to hook input data
        DistillHelper.run_model_one_batch(model, sample)
        for hook in hooks:
            hook.remove()

    def _dump_group_output_data(self, model, groups, epoch, step):
        ''' dump group's output data to file'''
        for group in groups:
            # get group module input data
            input_data = self.load_input_dump_data(group, epoch, step)
            # infer to get group output data
            output_data = DistillDataManager.get_output_data_by_inferring(model, group, input_data)

            # save data
            file_name = self._gen_file_path(group[0], epoch, step, is_in=False)
            np.save(file_name, output_data.cpu())
            self.dump_files.append(file_name)
            LOGGER.logd('dump layer {} output data success.'.format(group[0]), 'DistillDataManager')

    def _gen_file_path(self, layer_name, epoch, step, is_in=True):
        '''generate dump file path'''
        dump_suffix = f'{layer_name}_{epoch}_{step}_{self.device_name}.npy'
        if is_in:
            file_name = DATA_DUMP_INPUT_PREFIX + dump_suffix
        else:
            file_name = DATA_DUMP_OUTPUT_PREFIX + dump_suffix

        return os.path.join(TMP_PATH, file_name)
