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

import amct_pytorch.amct_pytorch_inner.amct_pytorch.common.cmd_line_utils.data_handler as data_handler
import amct_pytorch.amct_pytorch_inner.amct_pytorch.common.cmd_line_utils.arguments_handler as args_handler
from ...amct_pytorch.common.utils.check_params import check_params

from ...amct_pytorch.common.auto_calibration import AutoCalibrationEvaluatorBase
from ...amct_pytorch.utils.log import LOGGER


class ModelEvaluator(AutoCalibrationEvaluatorBase):
    '''
    Evaluator: an evaluator based on dataset.
    '''
    @check_params(
        input_shape=str,
        data_dir=str,
        data_types=str
    )
    def __init__(self, data_dir, input_shape, data_types):
        """
        input_shape: string, the input shape to feed into the model.
        data_dir: string, data bin path.
        data_types: string, data bin type.
        batch_num: int, batch_num to do calibration.
        """
        super().__init__()
        self._preprocess(input_shape, data_dir, data_types)

    @staticmethod
    def _preprocess_input_shape(input_shape):
        """
        Shape of input data. Separate multiple nodes with semicolons (;).
        Use double quotation marks (") to enclose each argument.
        E.g.: "input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"
        """
        if input_shape is None:
            raise ValueError("input shape in TorchModelEvaluator can not be None.")
        input_dict = args_handler.process_data_shape(input_shape)
        return input_dict

    @staticmethod
    def _preprocess_data_dir(data_dir):
        """
        The path to the processed binary datasets.
        For a multi-input model, different input data must be stored in different directories.
        Names of all files in each directory must be sorted in ascending lexicographic order.
        Use double quotation marks (") to enclose each argument.
        E.g.: "data/input1/;data/input2/"
        """
        if data_dir is None:
            raise ValueError("data_dir in TorchModelEvaluator can not be None.")
        data_paths = args_handler.process_multi_data_path(data_dir)
        return data_paths

    @staticmethod
    def _preprocess_data_types(data_types):
        """
        The dtype of the input data. Separate multiple nodes with semicolons (;).
        Use double quotation marks (") to enclose each argument.
        E.g.: "float32;float64"
        """
        if data_types is None:
            raise ValueError("data_types in TorchModelEvaluator can not be None.")
        values = data_types.split(';')
        return values

    @staticmethod
    def _preprocess_batch_num(batch_num):
        """The number of data batches used to run PTQ calibration. Default is set to 1."""
        if batch_num <= 0:
            raise ValueError("Invalid batch_num {} has been given, should be greater or equal to 0."
            "Please check it.".format(batch_num))
        return batch_num

    @check_params(
        modified_model=torch.nn.Module,
        batch_num=int
    )
    def calibration(self, modified_model, batch_num):
        '''
        Function: calibration
        '''
        batch_num = self._preprocess_batch_num(batch_num)
        self.evaluate(modified_model, batch_num)
        LOGGER.logi("Calibration Done.", "ModelEvaluator")

    @check_params(
        modified_model=torch.nn.Module,
        iterations=int
    )
    def evaluate(self, modified_model, iterations):
        '''
        Function: evaluate the modified_model with given iterations
        '''
        iterations = self._preprocess_batch_num(iterations)

        modified_model.eval()

        for data_map in data_handler.load_data(
                self.input_shape,
                self.data_dir,
                self.data_types,
                iterations
            ):
            with torch.no_grad():
                for value in data_map.values():
                    input_tensors = [torch.tensor(value)]
                    modified_model(*(input_tensors))

        LOGGER.logi("Evaluate Done.", "ModelEvaluator")

    def _preprocess(self, input_shape, data_dir, data_types):
        # key is name, value is each dim
        self.input_shape = self._preprocess_input_shape(input_shape)
        # a list of path
        self.data_dir = self._preprocess_data_dir(data_dir)
        # a list of data type
        self.data_types = self._preprocess_data_types(data_types)
