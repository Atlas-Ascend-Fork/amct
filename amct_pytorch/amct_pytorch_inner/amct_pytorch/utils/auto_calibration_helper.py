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
from collections import OrderedDict
import numpy as np

from torch import Tensor
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.common.utils import files as files_util


class AutoCalibrationHelper:
    """ helper for accuracy_based_auto_calibration
    """
    def __init__(self,
                 fused_module,
                 fake_quant_module,
                 quant_layers,
                 record_file,
                 temp_dir,
                 sensitivity):
        self.fused_module = fused_module
        self.fake_quant_module = fake_quant_module
        self.quant_layers = quant_layers
        self.record_file = record_file
        self.temp_dir = temp_dir
        self.sensitivity = sensitivity
        self.fakequant_module_helper = ModuleHelper(fake_quant_module)
        self.original_module_helper = ModuleHelper(fused_module)
        self.cosine_similarity_records = OrderedDict()
        self.shape_info = OrderedDict()

    def get_original_module(self, layer_name):
        """get the bn fused module from fused module"""
        module = self.original_module_helper.get_module(layer_name)
        return module

    def get_fakequant_module(self, layer_name):
        """get the fakequant module from fake_quant_module"""
        module = self.fakequant_module_helper.get_module(layer_name)
        return module

    def calc_ranking_info(self):
        """ calculate the ranking info of model"""
        real_temp_dir = os.path.realpath(self.temp_dir)
        if not os.path.isdir(real_temp_dir):
            raise RuntimeError(
                'amct_log/temp dir {} does not exists!'.format(
                    real_temp_dir))
        self.analyze_layer()
        return self.cosine_similarity_records, self.shape_info

    def analyze_layer(self):
        """
        Function: compare each module layer before and after quant.
        """
        for module_name in self.quant_layers:
            fm_file_list = files_util.find_dump_file(
                data_dir=os.path.realpath(self.temp_dir),
                name_prefix="{}_activation".format(module_name))
            if len(fm_file_list) == 0:
                raise RuntimeError(
                    "Can not find dump file for layer {}".format(module_name))
            cos_sim_list = []
            if self.shape_info.get(module_name) is None:
                self.shape_info[module_name] = {'input_shape': [], 'output_shape': []}
            for fm_file_path in fm_file_list:
                np_feature_map = files_util.parse_dump_data(fm_file_path, with_type=True)
                input_tensor = Tensor(np_feature_map)
                # generate single layer model
                single_module = self.get_original_module(module_name).cpu()
                fake_single_module = self.get_fakequant_module(module_name).cpu()
                original_output = single_module(input_tensor)
                fake_quant_output = fake_single_module(input_tensor)
                if original_output.shape != fake_quant_output.shape:
                    raise ValueError("shape of original output differ from shape of fake quant for layer {}"
                    .format(module_name))
                cos_sim = self.sensitivity.compare(
                    original_output.cpu().detach().numpy().flatten(),
                    fake_quant_output.cpu().detach().numpy().flatten())
                cos_sim_list.append(cos_sim)
                self.shape_info[module_name]['input_shape'].append(input_tensor.shape)
                self.shape_info[module_name]['output_shape'].append(original_output.shape)
            self.cosine_similarity_records[module_name] = np.mean(cos_sim_list)
            LOGGER.logi(
                "******** sensitivity of module {} is {} ********".format(
                    module_name, np.mean(cos_sim_list)))
        LOGGER.logi(
            '******** sensitivity_records ******** ', 'AutoCalibrationHelper')
