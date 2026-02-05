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
import shutil
from shutil import copyfileobj
from io import BytesIO
import pathlib

import torch # pylint: disable=E0401
import onnx

from onnx import onnx_pb # pylint: disable=E0401
from onnx.onnx_pb import AttributeProto
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.graph.graph import Graph
from ...amct_pytorch.configuration.check import GraphChecker
from ...amct_pytorch.common.utils import files as files_util
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.custom_op.quant_identity.quant_identity import \
    MarkedQuantizableModule
from ...amct_pytorch.common.utils.util import version_higher_than
from ...amct_pytorch.utils.vars import TORCH_VERSION
from ...amct_pytorch.utils.vars import PRUNABLE_TYPES
from ...amct_pytorch.utils.vars import PASSIVE_PRUNABLE_TYPES
from ...amct_pytorch.optimizer.conv_bn_fusion_pass import ConvBnFusionPass
from ...amct_pytorch.common.utils.prune_record_attr_util import AttrProtoHelper

SUPPORT_TYPES = (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d, torch.nn.ReLU6, torch.nn.ReLU,
    torch.nn.LeakyReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.Softmax)

TMP_PATH = os.path.realpath(os.path.join(os.getcwd(), 'amct_temp'))


class Parser:
    """ Helper of onnx file
    """
    def __init__(self, ckpt_path):
        self.__ckpt_path = ckpt_path

    @staticmethod
    def parse_proto(proto_file):
        """ parse the onnx pb pb file to """
        if isinstance(proto_file, str):
            with open(proto_file, 'rb') as onnx_pb_file:
                onnx_pb_str = onnx_pb_file.read()
                model = onnx_pb.ModelProto()
                model.ParseFromString(onnx_pb_str)
                LOGGER.logd(f'Parse onnx model from {proto_file} success.')
                return model
        elif isinstance(proto_file, BytesIO):
            onnx_pb_str = proto_file.getvalue()
            model = onnx_pb.ModelProto()
            model.ParseFromString(onnx_pb_str)
            LOGGER.logd(f'Parse onnx model from {proto_file} success.')
            return model
        else:
            raise TypeError(f'Not support proto type: "{type(proto_file)}"')

    @staticmethod
    def parse_net_to_graph(proto_file):
        """" parse the onnx pb graph to inner graph,
        """
        if isinstance(proto_file, str):
            proto_file_path = pathlib.Path(proto_file)
            # parse by proto
            if not proto_file_path.is_file():
                raise ValueError('The input onnx pb file is not exist.')
        if isinstance(proto_file, str):
            model_path = os.path.dirname(proto_file)
        else:
            model_path = os.path.realpath('./amct_temp')
        model = Parser.parse_proto(proto_file)
        graph = Graph(model, model_path)
        return graph

    @staticmethod
    def export_onnx(model,
                    args,
                    onnx_file,
                    export_setting=None):
        """
        Function: Save nn.module to onnx
        Inputs: model: an instance of torch.nn.Module
                args: tuple, the input data.
                onnx_file: a string, file path to save onnx file
                export_setting: a dict, some args for torch.onnx.export
        Returns: torch_out: model's output from args
        """
        if isinstance(onnx_file, str):
            files_util.create_file_path(onnx_file)

        if not isinstance(args, (tuple, torch.Tensor)):
            raise RuntimeError('input data type must be tuple or torch.Tensor!')

        if export_setting is None:
            export_setting = {}
        else:
            Parser.validate_export_setting(export_setting)
        if torch.__version__ == '2.1.0':
            export_setting['opset_version'] = 16
        else:
            export_setting['opset_version'] = 11
        if version_higher_than(torch.__version__, '1.12.0'):
            export_setting['keep_initializers_as_inputs'] = True
        if version_higher_than(torch.__version__, '1.5.0') and \
            not version_higher_than(torch.__version__, '1.11.0'):
            export_setting['enable_onnx_checker'] = False

        try:
            model = ModuleHelper.deep_copy(model)
        except RuntimeError as exception:
            LOGGER.logw(str(exception), "Parser::export_onnx")

        # add quantize mark op to user model
        model_helper = ModuleHelper(model)
        for name, mod in model.named_modules():
            if _is_rename(name, mod):
                parent_module = model_helper.get_parent_module(name)
                marked_module = MarkedQuantizableModule(mod, name)
                setattr(parent_module, name.split('.')[-1], marked_module)

        torch_out = _export_to_onnx(model, args, onnx_file, export_setting)

        # remove quantize mark op to user model
        model_helper = ModuleHelper(model)
        for name, mod in model.named_modules():
            if isinstance(mod, MarkedQuantizableModule):
                parent_module = model_helper.get_parent_module(name)
                original_module = mod.sub_module
                setattr(parent_module, name.split('.')[-1], original_module)

        # set file's permission 640
        if isinstance(onnx_file, str):
            os.chmod(onnx_file, files_util.FILE_MODE)

        return torch_out

    @staticmethod
    def write_node_attrs_extracted_from_onnx(graph, proto_file, attr_names):
        """
        Function: Read attributes from onnx model and write it into graph's node_proto
        Inputs: graph: model graph
                proto_file: str, onnx file path
                attr_names: list, attribute names read from onnx model to graph
        """
        with open(proto_file, 'rb') as onnx_pb_file:
            onnx_pb_str = onnx_pb_file.read()
            proto_data = onnx_pb.ModelProto()
            proto_data.ParseFromString(onnx_pb_str)
        for node in proto_data.graph.node:
            node_proto = graph.get_node_by_name(node.name)
            for attr in node.attribute:
                if attr.name in attr_names:
                    _write_attr_to_node_proto(node_proto, attr)

    @staticmethod
    def validate_export_setting(export_setting):
        """
        Function: check the export_setting passed in by user are valid
        Inputs: export_setting: dict, use to export onnx
        """
        input_output_sett = ['input_names', 'output_names']
        for setting_key in input_output_sett:
            if not export_setting.get(setting_key):
                continue
            for setting in export_setting.get(setting_key):
                if not isinstance(setting, str):
                    raise RuntimeError('{} type must be list(string)'.format(setting_key))
        if not export_setting.get('dynamic_axes'):
            return

        for key, value in export_setting.get('dynamic_axes').items():
            if not isinstance(key, str) or not isinstance(value, (dict, list)):
                raise RuntimeError('dynamic_axes type is invalid,'
                        'type must be dict<string, dict<python:int, string>> or dict<string, list(int)>')
            Parser.check_dynamic_axes_sub_item(value)

    @staticmethod
    def check_dynamic_axes_sub_item(value):
        """
        Function: check the values of dynamic_axes in export_setting are valid
        Inputs: value: dict or list, the values of dynamic_axes
        """
        if isinstance(value, dict):
            for x, y in value.items():
                if not isinstance(x, int) or not isinstance(y, str):
                    raise RuntimeError('dynamic_axes type is invalid,'
                        'type must be dict<string, dict<python:int, string>> or dict<string, list(int)>')
                if x < 0:
                    raise RuntimeError('dynamic_axes value is invalid,'
                        'The int value of axis indicators cannot be a negative number.')
        else:
            for item in value:
                if not isinstance(item, int):
                    raise RuntimeError('dynamic_axes type is invalid,'
                        'type must be dict<string, dict<python:int, string>> or dict<string, list(int)>')
                if item < 0:
                    raise RuntimeError('dynamic_axes value is invalid,'
                        'The int value of axis indicators cannot be a negative number.')


def _write_attr_to_node_proto(node_proto, attr):
    """
    Function: Save nn.module to onnx
    Inputs: node_proto: onnx.AttributeProto
            attr: str, attribute need to be written to node proto
    """
    attr_value = getattr(attr,
                    AttrProtoHelper.map_value_location(attr.type))
    if attr.type == AttributeProto.AttributeType.STRINGS:
        attr_value = [byte_string.decode('utf-8') \
            for byte_string in attr_value]
    elif attr.type == AttributeProto.AttributeType.STRING:
        attr_value = attr_value.decode('utf-8')
    node_proto.set_attr(attr.name, attr_value)


def _is_rename(name, mod):
    """
    Function: sub function for Parser.export_onnx, whether a mod need to keep its name in module
    :param:name: a string, mod's name
    :param:mod: torch.nn.module
    :return:bool, keep or not
    """
    if isinstance(mod, SUPPORT_TYPES):
        return True
    if GraphChecker.check_quantize_type(name, mod):
        return True
    if GraphChecker.check_retrain_type(name, mod):
        return True
    if ConvBnFusionPass.is_fusionable_conv(mod, name):
        return True
    # check prune limitations
    mod_type = type(mod).__name__
    if mod_type in PRUNABLE_TYPES or mod_type in PASSIVE_PRUNABLE_TYPES:
        return True
    return False


def _export_to_onnx(model, args, onnx_file, export_setting):
    """
    Function: sub function for Parser.export_onnx, export model to onnx
    :param:model: an instance of torch.nn.Module
    :param:args: tuple, the input data.
    :param:onnx_file: a string, file path to save onnx file
    :param:export_setting: a dict, some args for torch.onnx.export
    :return:torch_out, a numpy array
    """
    export_success = True
    try:
        torch_out = torch.onnx.export(model, args, onnx_file, **export_setting)
    except Exception as exception:
        if '2G' in str(exception):
            torch_out = _export_oversize_model(model, args, onnx_file, export_setting)
        else:
            raise RuntimeError("The model cannot export to onnx, "
                                "exception is: {}".format(exception)) from exception
    else:
        if isinstance(onnx_file, BytesIO) and len(onnx_file.getvalue()) == 0:
            export_success = False
        if isinstance(onnx_file, str) and os.path.getsize(onnx_file) == 0:
            export_success = False

    if not export_success:
        raise RuntimeError('Model cannot be quantized for it cannot be export to onnx! onnx file len is 0')

    return torch_out


def _export_oversize_model(model, args, onnx_file, export_setting):
    onnx_local_file = '{}/temp.onnx'.format(TMP_PATH)
    if not os.path.exists(TMP_PATH):
        os.mkdir(TMP_PATH)
    torch_out = torch.onnx.export(model, args, onnx_local_file, **export_setting)
    f = open(onnx_local_file, 'rb')
    copyfileobj(f, onnx_file)
    onnx_file.seek(0)
    return torch_out
