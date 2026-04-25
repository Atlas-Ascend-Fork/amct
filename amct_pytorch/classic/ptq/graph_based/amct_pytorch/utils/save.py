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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['save_onnx_model', 'generate_onnx_file_name', 'split_dir_prefix']

import os
import numpy as np
import onnx
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.common.utils import files as files_util
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper


# Limitation of model is 2GB
MAXIMUM_PROTOBUF = 2000000000


def convert_external_data_format(onnx_model, graph_proto, file_name, model_type):
    """
    Function: save external data in file.
    Inputs:
        onnx_model: onnx.onnx_ml_pb2.ModelProto, a model to convert.
        graph_proto: IR Graph
        file_name: a string, a file(.onnx) to save net's information.
        model_type: the model is deploy, fakequant or other model
    Returns: None
    """
    tensor_size = list()
    ori_tensor_external = list()
    # original model path
    model_path = graph_proto.model_path
    
    # Sort all tensors
    for index, initializer in enumerate(onnx_model.graph.initializer):
        tensor_size.append((index, initializer.ByteSize()))
        if initializer.data_location == 1:
            ori_tensor_external.append(index)

    tensor_size.sort(key=lambda x: x[1], reverse=True)

    # save external_data of input model in output path
    for index in ori_tensor_external:
        initializer = onnx_model.graph.initializer[index]  
        external_file = _gen_external_file_name(initializer, file_name, model_type)
        TensorProtoHelper(initializer, model_path).save_external_data(
            os.path.join(os.path.dirname(file_name), external_file))

    for index, _ in tensor_size:
        # check 2 GB limit
        if onnx_model.ByteSize() < MAXIMUM_PROTOBUF:
            break

        initializer = onnx_model.graph.initializer[index]
        external_file = _gen_external_file_name(initializer, file_name, model_type)
        TensorProtoHelper(initializer, model_path).save_external_data(
            os.path.join(os.path.dirname(file_name), external_file))


def _gen_external_file_name(initializer, file_name, model_type):
    """
    Function: generate external filename
    Inputs:
        initializer: TensorProto.
        file_name: a string, the model name.
        model_type: model is deploy, fakequant or other model
    Returns: a string, external filename
    """
    if model_type == 'Fakequant':
        external_file = initializer.name + '_fakequant.external'
    elif model_type == 'Deploy':
        external_file = initializer.name + '_deploy.external'
    else:
        external_file = initializer.name + '.external'
    return external_file

        
def save_onnx_model(graph_proto, file_name, model_type=None, node_info=None, deleted_attr=None):
    """
    Function: save graph_proto in file.
    Inputs:
        graph_proto: IR Graph, the graph_proto to be saved.
        file_name: a string, a file(.onnx) to save net's information.
        model_type: deploy or fakequant or others
    Returns: None
    """
    file_realpath = os.path.realpath(file_name)
    # save to file
    files_util.create_file_path(file_realpath, check_exist=True)
    dump_model = graph_proto.dump_proto()
    if node_info:
        dump_model = _write_node_info(dump_model, node_info)
    if deleted_attr:
        delete_customized_attr(dump_model, deleted_attr)
    convert_external_data_format(dump_model, graph_proto, file_realpath, model_type)
    with open(file_realpath, 'wb') as fid:
        fid.write(dump_model.SerializeToString())
    # set file's permission 640
    os.chmod(file_realpath, files_util.FILE_MODE)

    LOGGER.logi(f"The model file is saved in {file_realpath}", module_name='Utils')


def _write_node_info(onnx_model, node_info):
    """
    Function: set node info as attribute in the exported onnx model file.
    Inputs:
        onnx_model: exported onnx model.
        node_info: node info which will be written in onnx model as node attr.
    Outputs:
        onnx_model: exported onnx model.
    """
    for node_proto in onnx_model.graph.node:
        if node_proto.name not in node_info:
            continue
        for _, attr in enumerate(node_info.get(node_proto.name)):
            AttributeProtoHelper(node_proto).set_attr_value(attr.get('attr_name'),
                                                            attr.get('attr_type'),
                                                            attr.get('attr_val'))
    return onnx_model


def delete_customized_attr(onnx_model, deleted_attr):
    for node in onnx_model.graph.node:
        for idx, attr in enumerate(node.attribute):
            if attr.name in deleted_attr:
                del node.attribute[idx]


def generate_onnx_file_name(save_dir, save_prefix, save_type):
    ''' Generate model's name. '''
    if save_type == 'Deploy':
        file_suffix = 'deploy_model.onnx'
    else:
        file_suffix = 'fake_quant_model.onnx'

    if save_prefix != '':
        ckpt_file = os.path.join(save_dir, '_'.join([save_prefix,
                                                     file_suffix]))
    else:
        ckpt_file = os.path.join(save_dir, file_suffix)

    return ckpt_file


def split_dir_prefix(save_path):
    ''' split save_path to save_dir and save_prefix'''
    if save_path == '':
        save_prefix = ''
        save_dir = os.path.realpath(save_path)
    elif save_path != '' and save_path[-1] == '/':
        save_prefix = ''
        save_dir = os.path.realpath(save_path)
    else:
        save_dir, save_prefix = os.path.split(os.path.realpath(save_path))
    files_util.is_valid_save_prefix(save_prefix)

    return save_dir, save_prefix


def dump_ifmr_input_tensor(dump_data, dump_dir, layer_name, batch_idx):
    """ dump the input data"""
    if not os.path.exists(dump_dir):
        raise RuntimeError("{} does not exists.".format(dump_dir))
    layer_name = layer_name.replace("/", "_")
    npy_dump_file_path = "{}/{}_ifmr_layer_{}.npy".format(
        dump_dir, layer_name, batch_idx)
    np.save(npy_dump_file_path, dump_data)
