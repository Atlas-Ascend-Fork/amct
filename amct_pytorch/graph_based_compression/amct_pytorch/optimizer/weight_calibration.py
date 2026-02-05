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
from google.protobuf import text_format
import torch
import numpy as np

from ...amct_pytorch.configuration.configuration import Configuration
from ...amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from ...amct_pytorch.proto import scale_offset_record_pb2
from ...amct_pytorch.custom_op.arq.arq import weight_cali_tensor

from ...amct_pytorch.common.utils.record_file_operator import record_weights_scale_offset
from ...amct_pytorch.common.utils.record_file_operator import \
    read_activation_scale_offset
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.vars import QUANTIZABLE_TYPES
from ...amct_pytorch.configuration.check import GraphChecker
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.weight_quant_api import adjust_deconv_weight_shape
from ...amct_pytorch.utils.weight_quant_api import adjust_conv_weight_shape
from ...amct_pytorch.ada_round.ada_round_optimize import replace_adaround_module
from ...amct_pytorch.utils.module_info import ModuleInfo
CONV2D = 'Conv2d'
CONV3D = 'Conv3d'
CONV1D = 'Conv1d'


class WeightsCalibrationPass(BaseModuleFusionPass):
    """
    Function: Do caliration for weight, sacle and offset for weight will
        be found and weight is fake_quantized.
    APIs: set_up, tear_down, match_pattern, do_pass
    """
    def __init__(self, record_dict=None, weight_fakequant=True):
        """
        Function: init object
        Parameter:
        weight_fakequant: bool, whether to use the fake quant weight.
                For AMC Feature, set to False.

        Return: None
        """
        BaseModuleFusionPass.__init__(self)
        self.records = scale_offset_record_pb2.ScaleOffsetRecord()
        self.conf = Configuration()
        self.record_file_path = self.conf.get_record_file_path()
        self.weight_fakequant = weight_fakequant
        self.record_dict = {} if record_dict is None else record_dict

    @staticmethod
    def apply_balance_scale_to_weight(object_module, object_node, tensor_balance_factor, weight_tensor):
        """
        mul tensor_balance_factor to weight tensor
        Parameters:
            object_module: module to be processed
            object_node: node to be processed
            tensor_balance_factor: a list of tensor balance factor
            weight_tensor: node weight tensor
        Return: True: matched
                False: mismatch
        """
        weight_dtype = weight_tensor.dtype
        device = weight_tensor.device
        if weight_dtype is torch.float16:
            tensor_balance_factor = np.array(tensor_balance_factor, np.float16)
        else:
            tensor_balance_factor = np.array(tensor_balance_factor, np.float32)

        weight_tensor = weight_tensor.cpu()
        broadcast_shape = [1] * len(weight_tensor.shape)
        _, cin_axis = ModuleInfo.get_wts_cout_cin(object_module)
        broadcast_shape[cin_axis] = -1
        tensor_balance_factor = tensor_balance_factor.reshape(broadcast_shape)

        if type(object_module).__name__ in (CONV2D, CONV3D, CONV1D) and object_module.groups > 1:
            group = object_module.groups
            weight_tensor = adjust_conv_weight_shape(group, weight_tensor)
            weight_tensor = weight_tensor.transpose(0, 1)

        weight_tensor = weight_tensor * tensor_balance_factor

        if type(object_module).__name__ in (CONV2D, CONV3D, CONV1D) and object_module.groups > 1:
            group = object_module.groups
            weight_tensor = weight_tensor.transpose(0, 1)
            weight_tensor = adjust_conv_weight_shape(group, weight_tensor)
        return weight_tensor.to(device)
    
    @staticmethod
    def _graph_weight_set_process(object_node, object_module, weight):
        """
        Function: set weight data to graph.

        Args:
        object_node: graph object node.
        object_module: module to process.
        weight: new weigh to set into the graph.

        Return:
        None.
        """
        weight_param = QuantOpInfo.get_weight_node(object_node)
        weight_helper = TensorProtoHelper(weight_param.proto, weight_param.model_path)

        calied_weight_raw = weight
        # transpose torch.nn.Linear.weight in (Cout, Cin) to ONNX (Cin, Cout).
        if object_node.type == 'MatMul' and type(object_module).__name__ == 'Linear':
            need_transpose = True
            # Trans op may already be inserted in exported ONNX model
            if object_node.has_attr('with_weights_trans'):
                if object_node.get_attr('with_weights_trans'):
                    need_transpose = False
            if need_transpose:
                calied_weight_raw = calied_weight_raw.transpose(1, 0)

        calied_weight_raw = calied_weight_raw.flatten().cpu().numpy()
        if weight.dtype is torch.float16:
            set_data_type = 'FLOAT16'
        else:
            set_data_type = 'FLOAT'

        weight_helper.clear_data()
        weight_helper.set_data(calied_weight_raw, set_data_type)

    def set_up(self):
        """
        Function: read the scale and offset from Configuration's file.
        Inputs: None
        Returns: None
        """
        with open(self.record_file_path, 'r') as record_read_file:
            pbtxt_string = record_read_file.read()
            text_format.Merge(pbtxt_string, self.records)

    def match_pattern(self, module, name, graph=None):
        """
        Function:Match the module to be quantized in model
        Parameters:
            module: module to be matched
            name: module's name
            graph: graph structure, not necessary
        Return: True: matched
                False: mismatch
        """
        if type(module).__name__ not in QUANTIZABLE_TYPES:
            return False
        if name not in self.conf.get_quant_config():
            return False
        return True

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual Insert IFMR module
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        # Step0: do avgpooling calibration only by kernel shape
        object_node = graph.get_node_by_name(object_name)
        if object_node.type == 'AveragePool':
            self._averagegpool_process(object_name)
            return

        # Step1: do weight calibration by algo
        # get weights' information for quantization
        layer_config = self.conf.get_layer_config(object_name)
        wts_param = layer_config.get('weight_quant_params')
        dmq_param = layer_config.get('dmq_balancer_param')
        calied_weight = self._weight_calibration_process(
            object_module=object_module,
            object_node=object_node,
            wts_param=wts_param,
            dmq_param=dmq_param,
            model=model
        )
        object_module.weight.data = calied_weight

        # step2: set weights data to graph.
        self._graph_weight_set_process(
            object_node=object_node,
            object_module=object_module,
            weight=object_module.weight.data)

        LOGGER.logd('Do layer \'{}[{}]\' weights calibration success!'\
                    .format(object_node.name, wts_param.get('wts_algo')), \
                    'WeightsCalibrationPass')

    def tear_down(self):
        """
        Function: write the scale and offset to Configuration's file.
        Inputs: None
        Returns: None
        """
        with open(self.record_file_path, "w") as record_write_file:
            record_write_file.write(
                text_format.MessageToString(self.records, as_utf8=True))

    def _averagegpool_process(self, object_name):
        """
        Function: process average pool. write weight scale and offset.
        AveragePool do not need weight calibration.

        Args:
        object_name: name of object_module.
        """
        scale = [1.0]
        offset = [0]
        record_weights_scale_offset(self.records, object_name, scale,
                                    offset)
        LOGGER.logd(
            'Do layer:\'{}\' weights calibration success!'.format(
                object_name), 'WeightsCalibrationPass')

    def _weight_calibration_process(
        self,
        object_module,
        object_node,
        wts_param,
        dmq_param,
        model):
        """
        Function: do weight calibration by specific algorithm.
        write scale and offset to records.

        Args:
        object_module: module to process.

        Return:
        calied_weight: fake quant weight.
        Notes, `auto_mixed_precision_search` is original weight. NOT FAKE.
        """
        data_tensor = object_module.weight.data
        calied_weight = data_tensor
        tensor_balance_factor = None
        # broadcast tensor_balance_factor to weight shape and apply to weight
        if dmq_param:
            if not self.record_dict.get(object_node.name) or \
                not self.record_dict.get(object_node.name).get('tensor_balance_factor'):
                raise ValueError("config indicates dmq_balancer in layer: %s, " \
                                    "but no tensor_balance_factor found in record." \
                                    "please check quant_preprocess and calibration is done!" % object_node.name)

            tensor_balance_factor = self.record_dict.get(object_node.name).get('tensor_balance_factor')
            data_tensor = WeightsCalibrationPass.apply_balance_scale_to_weight(
                object_module, object_node, tensor_balance_factor, data_tensor)

        # find scale and offset
        if type(object_module).__name__ in ('ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'):
            group = object_module.groups
            data_tensor = adjust_deconv_weight_shape(group, data_tensor)
        scale = [1.0]
        offset = [0]
        if wts_param.get('wts_algo') == 'arq_quantize':
            scale, offset, calied_weight = weight_cali_tensor(data_tensor,
                                                              wts_param)
        elif wts_param.get('wts_algo') == 'ada_quantize':
            if tensor_balance_factor is not None:
                tensor_balance_factor = torch.tensor(tensor_balance_factor, dtype=data_tensor.dtype)
                if type(object_module).__name__ in ('ConvTranspose2d', CONV2D):
                    tensor_balance_factor = tensor_balance_factor.reshape([1, -1, 1, 1])
            adaround_module = replace_adaround_module(object_node.name, model, data_tensor, tensor_balance_factor)
            scale, offset = adaround_module.get_scale_offset()
            calied_weight = data_tensor
        else:
            weights_len = data_tensor.numel()

        # FOR AMC Feature, weight do not need fake quant.
        if not self.weight_fakequant:
            calied_weight = data_tensor

        if type(object_module).__name__ in ('ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'):
            group = object_module.groups
            calied_weight = adjust_deconv_weight_shape(group, data_tensor)

        # save the quantize information
        record_weights_scale_offset(self.records, object_node.name, scale,
                                    offset, wts_param.get('num_bits'))
        return calied_weight

