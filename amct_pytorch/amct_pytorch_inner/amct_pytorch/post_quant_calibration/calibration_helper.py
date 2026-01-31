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
import math
import random
import torch

from amct_pytorch.common.utils.check_params import check_params
from amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.utils.weight_quant_api import apply_lut_quantize_weight
from amct_pytorch.utils.vars import LUT_QUANT_MODULE


class CalibrationHelperBase:
    """
    Function: post quant calibration helper base class
    APIs:
        get_calibration_blocks
        model_forward
        block_forward
        get_module_input
        get_module_output
        set_quant_output
        quantization_layer_filter
    """
    @check_params(model=torch.nn.Module,
                  calibration_block=(type(None), str),
                  lr=float,
                  epochs=int,
                  layer_filter=bool,
                  error_ratio_threshold=float,
                  error_average_threshold=float)
    def __init__(self,
                 model,
                 calibration_data,
                 forward_func,
                 calibration_block=None,
                 block_forward_func=None,
                 lr=1e-3,
                 epochs=1,
                 layer_filter=False,
                 error_ratio_threshold=0.2,
                 error_average_threshold=0.1):
        """
        Function: init objective.
        Args:
            model: calibration model
            calibration_data: input data for calibration
            forward_func: model forward function
            calibration_block: calibration as a whole
            block_forward_func: block forward function
            lr: learning rate
            epochs: iteration num
            layer_filter: whether filter quant layer
            error_ratio_threshold & error_average_threshold: filter threshold
        """
        self.model = model
        self.module_helper = ModuleHelper(model)
        self.calibration_data = calibration_data
        self.forward_func = forward_func
        self.calibration_block = LUT_QUANT_MODULE if calibration_block is None else tuple([calibration_block])
        self.block_forward_func = block_forward_func
        self.lr = lr
        self.epochs = epochs
        self.layer_filter = layer_filter
        self.error_ratio_threshold = error_ratio_threshold
        self.error_average_threshold = error_average_threshold

        self._check_values()

    @staticmethod
    def _clac_error(a, b, atol=0.0025, rtol=0.1):
        """
        Function: calculate error between a & b
        """
        abs_error = (a - b).abs()
        check_ok = abs_error <= (atol + rtol * b.abs())
        error_ratio = 1 - check_ok.sum() / a.numel()
        abs_b = b.abs() + 1e-6
        average_error = torch.min((abs_error / abs_b).sum(), abs_error.sum()) / a.numel()
        return error_ratio, average_error

    def get_calibration_blocks(self):
        """
        Function: get all calibration blocks
        Inputs:
            None
        Outputs:
            calibration_blocks: {name:mod}
        """
        calibration_blocks = dict()
        for block_name, block in self.model.named_modules():
            if type(block).__name__ not in self.calibration_block:
                continue
            for _, mod in block.named_modules():
                if type(mod).__name__ in LUT_QUANT_MODULE:
                    calibration_blocks[block_name] = block
                    break
        return calibration_blocks

    def model_forward(self):
        """
        Function: forward whole model
        """
        self.forward_func(self.model, self.calibration_data)

    def block_forward(self, block, *inputs, **kwargs):
        """
        Function: forward block
        """
        if self.block_forward_func is None:
            outputs = block(*inputs, **kwargs)
        else:
            outputs = self.block_forward_func(block, *inputs, **kwargs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs

    @torch.no_grad()
    def get_module_input_output(self, name, module):
        """
        Function: get input data and kwargs of module
        Inputs:
            name: name of module
            module: torch.nn.Module
        Outputs:
            inputs: input data
            module_kwargs: key work args
            outputs: output data
        """
        inputs = list()
        module_kwargs = dict()
        outputs = list()

        class Catcher(torch.nn.Module):
            def __init__(self, ori_module):
                super().__init__()
                self.ori_module = ori_module

            def forward(self, *input_data, **kwargs):
                inputs.append(input_data)
                module_kwargs.update(kwargs)
                output_data = self.ori_module(*input_data, **kwargs)
                if isinstance(output_data, tuple):
                    output_data = output_data[0]
                outputs.append(output_data)
                raise ValueError

        module = Catcher(module)
        if name == '':
            self.model = module
        else:
            self.module_helper.replace_module_by_name(self.model, name, module)
        try:
            self.model_forward()
        except ValueError:
            pass
        if name == '':  # module is the model
            self.model = module.ori_module
        else:
            self.module_helper.replace_module_by_name(self.model, name, module.ori_module)

        return inputs[0], module_kwargs, outputs[0]

    def set_quant_output(self, quant_output):
        """
        Function: set lut quant module output quant value
        """
        for _, mod in self.model.named_modules():
            if type(mod).__name__ in LUT_QUANT_MODULE:
                mod.set_quant_output(quant_output)

    @torch.no_grad()
    def quantization_layer_filter(self):
        """
        Function: filter quant layer and delete record
        """
        filter_layers = dict()
        for name, mod in self.model.named_modules():
            if type(mod).__name__ not in LUT_QUANT_MODULE:
                continue
            ori_weight = mod.weight
            quant_weight = apply_lut_quantize_weight(ori_weight, mod.lut)
            error_ratio, average_error = self._clac_error(ori_weight, quant_weight)
            if error_ratio > self.error_ratio_threshold and average_error > self.error_average_threshold:
                filter_layers[name] = mod

        if len(filter_layers) == 0:
            return

        # delete record of filter layer
        record_module = next(iter(filter_layers.items()))[1].post_quant_record_module
        records = record_module.records.record
        filter_indices = [i for i, record in enumerate(records) if record.key in filter_layers]
        for i in reversed(filter_indices):
            del records[i]
        record_module._write_record_file()

    def _check_values(self):
        has_quant_module = False
        has_calibration_block = False
        for _, mod in self.model.named_modules():
            if type(mod).__name__ in LUT_QUANT_MODULE:
                has_quant_module = True
            if type(mod).__name__ in self.calibration_block:
                has_calibration_block = True
            if has_quant_module and has_calibration_block:
                break
        if not has_quant_module:
            raise ValueError("Not found LUT quant module in model.")
        if not has_calibration_block:
            raise ValueError("Not found {} in model.".format(self.calibration_block[0]))

        if not callable(self.forward_func):
            raise ValueError("forward_func is uncallable.")

        if self.block_forward_func is not None and not callable(self.block_forward_func):
            raise ValueError("block_forward_func is uncallable.")

        if self.lr <= 0:
            raise ValueError("learning rate({}) should be greater than 0.".format(self.lr))

        if self.epochs <= 0:
            raise ValueError("epochs({}) should be greater than 0.".format(self.epochs))

        if self.error_ratio_threshold <= 0:
            raise ValueError("error_ratio_threshold({}) should be greater than 0.".format(self.error_ratio_threshold))

        if self.error_average_threshold <= 0:
            raise ValueError("error_average_threshold({}) should be greater than 0.".format(
                self.error_average_threshold))


class LLMHelper(CalibrationHelperBase):
    def __init__(self,
                 model,
                 calibration_data,
                 forward_func=None,
                 calibration_block=None,
                 block_forward_func=None,
                 lr=1e-3,
                 epochs=1,
                 layer_filter=False,
                 error_ratio_threshold=0.2,
                 error_average_threshold=0.1):
        """
        Function: init objective.
        Args:
            model: calibration model
            calibration_data: input data for calibration
            forward_func: model forward function
            calibration_block: calibration as a whole
            block_forward_func: block forward function
            lr: learning rate
            epochs: iteration num
            layer_filter: whether filter quant layer
            error_ratio_threshold & error_average_threshold: filter threshold
        """
        if forward_func is None:
            forward_func = self.default_forward_func
        super().__init__(model, calibration_data, forward_func, calibration_block, block_forward_func,
            lr, epochs, layer_filter, error_ratio_threshold, error_average_threshold)

    def default_forward_func(self, model, calibration_data):
        """
        Function: default forward method of LLM
        """
        model(calibration_data)


class DPMHelper(CalibrationHelperBase):
    @check_params(time_step=int)
    def __init__(self,
                 model,
                 calibration_data,
                 forward_func,
                 time_step,
                 calibration_block=None,
                 block_forward_func=None,
                 lr=1e-3,
                 epochs=1,
                 layer_filter=False,
                 error_ratio_threshold=0.2,
                 error_average_threshold=0.1):
        """
        Function: init objective.
        Args:
            model: calibration model
            calibration_data: input data for calibration
            forward_func: model forward function
            time_step: time step for sampling
            calibration_block: calibration as a whole
            block_forward_func: block forward function
            lr: learning rate
            epochs: iteration num
            layer_filter: whether filter quant layer
            error_ratio_threshold & error_average_threshold: filter threshold
        """
        super().__init__(model, calibration_data, forward_func, calibration_block, block_forward_func,
            lr, epochs, layer_filter, error_ratio_threshold, error_average_threshold)

        if time_step <= 0:
            raise ValueError("time_step({}) should be greater than 0.".format(time_step))
        self.time_step = time_step

    def model_forward(self):
        """
        Function: forward whole model
        """
        sample_step = self._get_sample_step(self.time_step)
        return self.forward_func(self.model, self.calibration_data, sample_step)

    @staticmethod
    def _get_sample_step(time_step):
        """
        Function: use NDTC to calculate sample step
        """
        # time_step less than 7 occurs randint's args illegal
        if time_step < 7:
            return random.randint(0, time_step - 1)

        half = time_step // 2
        top_index = half + 2

        left = random.randint(0, 1)
        if left > 0:
            rand_index = random.randint(0, half * half - 1)
            index = int(math.sqrt(rand_index)) + 1
            sample_step = index + 2
        else:
            remain = half - (time_step - top_index) + 1
            rand_index = random.randint(remain * remain, half * half - 1)
            index = int(math.sqrt(rand_index)) + 1
            sample_step = time_step + remain - index
        return sample_step
