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
from io import BytesIO
import torch

import amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer as opt
from ..amct_pytorch.configuration.distill_config import create_default_distill_config
from ..amct_pytorch.configuration.distill_config import create_distill_config_from_proto
from ..amct_pytorch.configuration.distill_config import parse_distill_config
from ..amct_pytorch.parser.parser import Parser
from ..amct_pytorch.utils.log import LOGGER
from ..amct_pytorch.common.utils.check_params import check_params
from ..amct_pytorch.common.utils import files as files_util
from ..amct_pytorch.common.utils.log_base import LOG_FILE_DIR
from ..amct_pytorch.common.utils.record_file_operator import \
    ScaleOffsetRecordHelper
from ..amct_pytorch.utils.model_util import ModuleHelper
from ..amct_pytorch.distill.distill_data_manager import DistillDataManager
from ..amct_pytorch.distill.distill_helper import DistillHelper
from ..amct_pytorch.distill.distill_sample import DistillSampleBase
from ..amct_pytorch.quantize_tool import _generate_model
from ..amct_pytorch.quantize_tool import _check_config_consistency
from ..amct_pytorch.proto import scale_offset_record_pb2
from ..amct_pytorch.parser.parse_record_file import RecordFileParser

MODULE_NAME = 'Distill'


@check_params(config_file=str,
              model=torch.nn.Module,
              input_data=(torch.Tensor, tuple),
              config_defination=(type(None), str))
def create_distill_config(
        config_file,
        model,
        input_data,
        config_defination=None):
    """
    Function: Create distill quantize configuration json file for amct_pytorch
        tool
    Parameter: config_file: file path of quantize configuration json file
               model: user mode instance of Torch.nn.Module
               input_data: used to compile model, can be ramdom data
               config_defination: simple config file from user to set
    Return: None
    """
    config_file = os.path.realpath(config_file)
    files_util.is_valid_name(config_file, 'config_file')

    if isinstance(model, (torch.nn.parallel.DistributedDataParallel,)):
        model = model.module

    ModuleHelper(model).check_amct_op()

    # parse to graph
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx)
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)

    # create config file for distill quantization
    if config_defination is None:
        create_default_distill_config(config_file, graph)
    else:
        config_proto_file = os.path.realpath(config_defination)
        files_util.is_valid_name(config_proto_file, 'config_defination')
        if not os.path.exists(config_proto_file):
            raise FileNotFoundError("Not found config_defination file {}".format(config_proto_file))
        create_distill_config_from_proto(
            config_file,
            graph,
            config_proto_file)


@check_params(config_file=str,
              model=torch.nn.Module,
              input_data=(torch.Tensor, tuple))
def create_distill_model(config_file, model, input_data):
    """
    Function: Modify user's model for compressed in train process.
    Parameter: config_file: compressed quantize configuration json file
               model: user pytorch model's model file
               input_data: used to compile model, can be ramdom data
    Return: model: modified pytorch model for distill.
    """
    config_file = DistillHelper.get_config_file(config_file)

    try:
        model_copy = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "create_distill_model")
        model_copy = model

    # create a reference to the de-parallel model, which all modification will be applied on.
    if isinstance(model_copy, (torch.nn.parallel.DistributedDataParallel,)):
        model = model_copy.module
    else:
        model = model_copy

    ModuleHelper(model).check_amct_op()
    # parse to graph
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx)
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)
    distill_config = parse_distill_config(config_file, model)

    # quantize
    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.InsertQatPass(distill_config))
    optimizer.do_optimizer(model, graph)

    return model_copy


def _get_data(data_manager, distill_helper, group, index, sample):
    '''get input data of student and output data of teacher'''
    (epoch, step) = index
    if distill_helper.is_dump:
        input_data_t = data_manager.load_input_dump_data(
            group, epoch, step)
        output_data_t = data_manager.load_output_dump_data(
            group, epoch, step)

        dump_sample = data_manager.load_model_input_dump_data(epoch, step)
        input_data_s = data_manager.get_input_data_by_inferring(
            distill_helper.model_s, group, dump_sample)
    else:
        # load input data of teacher model and student model
        input_data_t = data_manager.get_input_data_by_inferring(
            distill_helper.model_t, group, sample)
        output_data_t = data_manager.get_output_data_by_inferring(
            distill_helper.model_t, group, input_data_t)
        input_data_s = data_manager.get_input_data_by_inferring(
            distill_helper.model_s, group, sample)

    input_data_s = data_manager.get_norm_min_data(input_data_t, input_data_s)
    return input_data_s, output_data_t


@check_params(model=torch.nn.Module,
              compress_model=torch.nn.Module,
              config_file=str,
              train_loader=torch.utils.data.DataLoader,
              epochs=int,
              lr=float,
              sample_instance=(type(None), DistillSampleBase),
              loss=(type(None), torch.nn.modules.loss._Loss),
              optimizer=(type(None), torch.optim.Optimizer))
def distill(model, compress_model, config_file, train_loader, epochs=1,
    lr=1e-3, sample_instance=None, loss=None, optimizer=None):
    '''
    Function: distill the compressed model
    Parameter: model: golden model
               compress_model: compress model to distill
               config_file: distill config file
               train_loader: data loader
               epochs: distill epochs
               lr: learning rate
               sample_instance: sample process instance
               loss: distill loss function
               optimizer: distill optimizer
    Return: distilled compress model
    '''
    # check input parameter
    if epochs < 1:
        raise ValueError('invalid param epochs {}'.format(epochs))

    distill_helper = DistillHelper(
        model, compress_model, config_file, loss, sample_instance)
    distill_helper.do_calibration(train_loader)

    data_manager = DistillDataManager(distill_helper.sample_ins)
    if distill_helper.is_dump:
        data_manager.dump_data(
            distill_helper.model_t, distill_helper.distill_groups, epochs, train_loader)

    # distill each group
    group_size = len(distill_helper.distill_groups)
    LOGGER.logi('distill group num: {}'.format(group_size), MODULE_NAME)
    for group_index, group in enumerate(distill_helper.distill_groups):
        # get student distill modules
        distill_modules = distill_helper.get_distill_modules(distill_helper.model_s, group)
        distill_opt = distill_helper.gen_optimizer_per_group(distill_modules, optimizer, lr)
        for epoch in range(epochs):
            loss_per_epoch = 0
            for step, sample in enumerate(train_loader):
                sample = distill_helper.get_model_input(sample)
                index = (epoch, step)
                input_data_s, output_data_t = _get_data(
                    data_manager, distill_helper, group, index, sample)

                # student model forward & loss
                loss_val = distill_helper.get_distill_modules_loss(
                    distill_modules, input_data_s, output_data_t)
                distill_opt.zero_grad()
                loss_val.backward()
                distill_opt.step()
                loss_per_epoch += loss_val.item()
                LOGGER.logd('step {}/{} loss {}'.format(step + 1, len(train_loader), loss_val), MODULE_NAME)
            loss_per_epoch = loss_per_epoch / epochs
            LOGGER.logi('epoch {}/{} distill group {} finished, loss {}'
                .format(epoch + 1, epochs, group_index + 1, loss_per_epoch), MODULE_NAME)
        LOGGER.logi('group {}/{} distill group finished'.format(group_index + 1, group_size), MODULE_NAME)
    data_manager.release()

    LOGGER.logi('distill model success.')
    return distill_helper.model_s


@check_params(model=torch.nn.Module,
              save_path=str,
              input_data=(torch.Tensor, tuple),
              record_file=(type(None), str),
              input_names=(list, type(None)),
              output_names=(list, type(None)),
              dynamic_axes=(dict, type(None)))
def save_distill_model(
        model,
        save_path,
        input_data,
        record_file=None,
        input_names=None,
        output_names=None,
        dynamic_axes=None):
    """
    Function: save compressed model to fakequant_onnx_file and
        deploy_onnx_file.
    Parameter:
        model: compressed model
        save_path: a string, the path where to store model and model's name.
        input_data: used to compile model, can be ramdom data
        record_file: temporary file to store scale and offset
        input_names: list of strings, names to assign to the input nodes of
            the graph, in order
        output_names: names to assign to the output nodes of the graph
            in order
        dynamic_axes: a dictionary to specify dynamic axes of input/output
    Return: None.
    """
    try:
        model_copy = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(str(exception), "save_distill_model")
        model_copy = model

    # create a reference to the de-parallel model, which all modification will be applied on.
    if isinstance(model_copy, (torch.nn.parallel.DistributedDataParallel,)):
        model = model_copy.module
    else:
        model = model_copy

    # check inputs
    if record_file is None:
        record_file = os.path.join(LOG_FILE_DIR, 'scale_offset_record.txt')
    os.makedirs(os.path.split(record_file)[0], exist_ok=True)
    files_util.check_files_exist([record_file])
    ModuleHelper(model).check_amct_distill_op()

    model.eval()
    record_helper = ScaleOffsetRecordHelper(scale_offset_record_pb2.ScaleOffsetRecord)

    # preprocess distill model
    _preprocess_distill_model(model, input_data, record_helper)

    # save model to fakequant and deploy
    modfied_onnx_file = BytesIO()
    Parser.export_onnx(model, input_data, modfied_onnx_file,
                       {'input_names': input_names,
                        'output_names': output_names,
                        'dynamic_axes': dynamic_axes})

    graph = Parser.parse_net_to_graph(modfied_onnx_file)

    record_helper.dump(record_file)
    # parse record_file
    record_parser = RecordFileParser(record_file, graph, 'model after distill')

    if record_parser.is_records_empty():
        raise RuntimeError(
            "record_file is empty, no layers to be saved. "
            "please ensure distill is finished by checking information!")
    records, _ = record_parser.parse()

    _generate_model(graph, records, save_path)


def _preprocess_distill_model(model, input_data, record_helper):
    """
    Function: process compressed model before saving as follows:
            1. delete compressed module
            2. fuse bn
    Inputs:
        model: distill model
        input_data: used to compile model, can be ramdom data
        record_helper: ScaleOffsetRecordHelper class to record quant factor
    Returns:
        model_copy: a model processed
    """
    # delete distill pass
    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.DeleteQatPass(record_helper))
    optimizer.add_pass(opt.RepalceSyncBNPass())
    optimizer.do_optimizer(model, None)

    # fuse bn in model
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx)
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)

    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.ConvBnFusionPass(None, record_helper))
    optimizer.do_optimizer(model, graph)
