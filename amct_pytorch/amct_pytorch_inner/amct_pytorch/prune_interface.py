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
from ..amct_pytorch.common.utils.check_params import check_params
from ..amct_pytorch.common.utils import files as files_util

from ..amct_pytorch.configuration.retrain_config import RetrainConfig
from ..amct_pytorch.parser.parser import Parser
from ..amct_pytorch.parser.parse_record_file import RecordFileParser
from ..amct_pytorch.utils.model_util import ModuleHelper
from ..amct_pytorch.utils.log import LOGGER
from ..amct_pytorch.prune.pruner_helper import PruneHelper

from ..amct_pytorch.quantize_tool import _generate_model
from ..amct_pytorch.quantize_tool import _preprocess_retrain_model
from ..amct_pytorch.quantize_tool import _copy_modified_model
from ..amct_pytorch.quantize_tool import _modify_original_model_to_quant
from ..amct_pytorch.utils.model_util import load_pth_file
from ..amct_pytorch.utils.save import split_dir_prefix
from ..amct_pytorch.utils.save import generate_onnx_file_name
from ..amct_pytorch.utils.vars import TORCH_VERSION
from ..amct_pytorch.common.utils.util import version_higher_than
from ..amct_pytorch.utils.singleton_record import SingletonScaleOffsetRecord
from ..amct_pytorch.custom_op.selective_prune.selective_prune \
    import create_selective_prune_record
from ..amct_pytorch.custom_op.selective_prune.selective_prune \
    import restore_selective_prune_record


@check_params(model=torch.nn.Module,
              config_defination=str,
              record_file=str)
def create_prune_retrain_model(model, input_data, config_defination, record_file):
    """
    Function: create_prune_retrain_model API. create a pruned model
    of given 'model'.

    Parameter:
    config_file: retrain quantize configuration json file
    model: user pytorch model's model file
    record_file: temporary file to store scale and offset
    input_data: used to compile model, can be ramdom data

    Return: model: modified pytorch model for prune.
    """
    ModuleHelper(model).check_amct_op()
    try:
        model = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "create_quant_retrain_config")

    record_file = files_util.create_empty_file(record_file, check_exist=True)
    SingletonScaleOffsetRecord().reset_singleton(record_file)

    # parse to graph
    model_onnx = BytesIO()
    Parser.export_onnx(model, input_data, model_onnx)
    graph = Parser.parse_net_to_graph(model_onnx)

    graph.add_model(model)
    # parse config
    RetrainConfig.init(graph, config_defination,
                        enable_retrain=True,
                        enable_prune=True)

    # do prune
    prune_helper = PruneHelper(graph, input_data, record_file)
    prune_helper.create_prune_model()

    # do selective prune
    create_selective_prune_record(graph)
    _modify_original_model_to_prune(model, graph)

    return model


@check_params(model=torch.nn.Module,
              record_file=str,
              config_defination=str,
              pth_file=str,
              state_dict_name=(str, type(None)))
def restore_prune_retrain_model(
    model,
    input_data,
    record_file,
    config_defination,
    pth_file,
    state_dict_name=None):
    """
    Function: restore_prune_retrain_model API. Restore pruned model
    based on record_file.

    Parameter:
    model: user pytorch model's model file
    record_file: temporary file to store scale and offset
    input_data: used to compile model, can be ramdom data
    config_defination: used to make prune config.
    pth_file: user prune training checkpoint file path.
    state_dict_name: key value of weight parameter in pth_file.

    Return: model: modified pytorch model for prune.
    """
    ModuleHelper(model).check_amct_op()
    try:
        model = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "restore_prune_retrain_model")

    record_file = os.path.realpath(record_file)
    SingletonScaleOffsetRecord().reset_singleton(record_file)
    # parse to graph
    model_onnx = BytesIO()
    Parser.export_onnx(model, input_data, model_onnx)
    graph = Parser.parse_net_to_graph(model_onnx)
    graph.add_model(model)

    # parse config
    RetrainConfig.init(graph, config_defination,
                        enable_retrain=True,
                        enable_prune=True)

    prune_helper = PruneHelper(graph, input_data, record_file)
    prune_helper.restore_prune_model()

    restore_selective_prune_record()
    _modify_original_model_to_prune(model, graph)

    model = load_pth_file(model, pth_file, state_dict_name)

    return model


@check_params(
    model=torch.nn.Module,
    save_path=str,
    input_names=(list, type(None)),
    output_names=(list, type(None)),
    dynamic_axes=(dict, type(None))
)
def save_prune_retrain_model(
    model,
    save_path,
    input_data,
    input_names=None,
    output_names=None,
    dynamic_axes=None
):
    """
    Function:
    Save modified prune model to fake quant onnx file and deploy onnx file

    Parameter:
    model: selective prune model.
    save_path: a string, the path where to store model and model's name.
    input_data: used to compile model, can be random data.
    input_names: list of strings, names to assign to the input nodes of
                 the graph, in order.
    output_names: names to assign to the output nodes of the graph, in order.
    dynamic_axes: a dictionary to specify dynamic axes of input/output

    Return: None.
    """
    # check input
    export_setting = {'input_names': input_names,
                      'output_names': output_names,
                      'dynamic_axes': dynamic_axes}
    _check_param_onnx_export(**export_setting)

    # step1. preprocess prune model, restore raw module and set prune wts
    model_copy = ModuleHelper.deep_copy(model)
    _preprocess_prune_model(model_copy)
    # step2. save corresponding onnx file.
    _inner_save_plain_prune_model(
        model=model_copy,
        save_path=save_path,
        input_data=input_data,
        **export_setting
    )


def _modify_original_to_compressed_model(model, input_data, config_defination, record_file, prune_call_mode):
    # step 1. parse
    # parse to graph
    model_onnx = BytesIO()
    Parser.export_onnx(model, input_data, model_onnx)
    graph = Parser.parse_net_to_graph(model_onnx)
    graph.add_model(model)

    # parse compressed config
    RetrainConfig.init(graph,
                        config_defination,
                        enable_retrain=True,
                        enable_prune=True)

    # step 2. do filter prune
    if RetrainConfig.enable_prune:
        prune_helper = PruneHelper(graph, input_data, record_file)
        if prune_call_mode == "create":
            prune_helper.create_prune_model()
            create_selective_prune_record(graph)
        elif prune_call_mode == "restore":
            prune_helper.restore_prune_model()
            restore_selective_prune_record()

    model_onnx = BytesIO()
    Parser.export_onnx(model, input_data, model_onnx)
    graph = Parser.parse_net_to_graph(model_onnx)
    graph.add_model(model)

    # step 3. do selective prune
    if RetrainConfig.enable_prune:
        _modify_original_model_to_prune(model, graph)

    # step 4. retrain quantize
    if RetrainConfig.enable_retrain:
        _modify_original_model_to_quant(model, record_file, graph)

    return model


@check_params(
    model=torch.nn.Module,
    config_defination=str,
    record_file=str,
)
def create_compressed_retrain_model(
    model,
    input_data,
    config_defination,
    record_file):
    """
    Function:
    create_compressed_retrain_model API.
    create a prund and retrain quantize model of given 'torch model'.

    Parameter:
    model: user pytorch model's model file.
    input_data: used to compile model, can be random data.
    config_defination: simple compressed config file from user to set.
    record_file: temporary file to store producer, consumer and scale, offset.

    Return: model: modified pytorch model for retrain.
    """
    ModuleHelper(model).check_amct_op()
    try:
        model = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "create_compressed_retrain_model")

    config_defination = os.path.realpath(config_defination)
    files_util.is_valid_name(record_file, 'record_file')
    record_file = files_util.create_empty_file(record_file, check_exist=True)
    SingletonScaleOffsetRecord().reset_singleton(record_file)

    model = _modify_original_to_compressed_model(
        model,
        input_data,
        config_defination,
        record_file,
        "create"
    )

    return model


@check_params(
    model=torch.nn.Module,
    config_defination=str,
    record_file=str,
    pth_file=str,
    state_dict_name=(str, type(None))
)
def restore_compressed_retrain_model(
    model,
    input_data,
    config_defination,
    record_file,
    pth_file,
    state_dict_name=None
    ):
    """
    Function:
    restore_compressed_retrain_model API. Restore compressed model
    based on record_file.

    Parameter:
    model: user pytorch model's model file.
    input_data: used to compile model, can be random data.
    config_defination: used to make prune and retrain config.
    record_file: temporary file to store producer, consumer and scale, offset.
    pth_file: user quant aware training checkpoint file path.
    state_dict_name: key value of weight parameter in pth_file.
    """
    ModuleHelper(model).check_amct_op()
    try:
        model = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "restore_compressed_retrain_model")

    record_file = os.path.realpath(record_file)
    SingletonScaleOffsetRecord().reset_singleton(record_file)
    pth_file = os.path.realpath(pth_file)

    model = _modify_original_to_compressed_model(
        model,
        input_data,
        config_defination,
        record_file,
        "restore"
    )

    model = load_pth_file(model, pth_file, state_dict_name)

    return model


@check_params(
    model=torch.nn.Module,
    record_file=str,
    save_path=str,
    input_names=(list, type(None)),
    output_names=(list, type(None)),
    dynamic_axes=(dict, type(None))
)
def save_compressed_retrain_model(
    model,
    record_file,
    save_path,
    input_data,
    input_names=None,
    output_names=None,
    dynamic_axes=None
):
    """
    Function:
    IF ONLY HAS PRUNE, save modified model to deploy and fake quant onnx file.
    The two files are the same.
    OTHERWISE, save modified prune and retrain quantize to fake quant onnx file
    and deploy onnx file, the execution steps are same with quant.

    Parameter:
    model: prune and retrain model.
    record_file: temporary file to store producer, consumer and scale, offset.
    save_path: a string, the path where to store model and model's name.
    input_data: used to compile model, can be random data.
    input_names: list of strings, names to assign to the input nodes of
                 the graph, in order.
    output_names: names to assign to the output nodes of the graph, in order.
    dynamic_axes: a dictionary to specify dynamic axes of input/output

    Return: None.
    """
    # check inputs
    export_setting = {'input_names': input_names,
                      'output_names': output_names,
                      'dynamic_axes': dynamic_axes}
    _check_param_onnx_export(**export_setting)

    files_util.is_valid_name(record_file, 'record_file')
    record_file = os.path.realpath(record_file)

    # copy model to a new one
    model_copy = _copy_modified_model(model, record_file)
    # step1. preprocess selective prune model
    _preprocess_prune_model(model_copy)
    # step2. check if only prune.
    only_prune = check_only_prune(model_copy, record_file)
    # step3. save corresponding onnx file.
    if only_prune:
        _inner_save_plain_prune_model(
            model=model_copy,
            save_path=save_path,
            input_data=input_data,
            **export_setting
        )
    else:
        _inner_save_compressed_retrain_model(
            model=model_copy,
            record_file=record_file,
            save_path=save_path,
            input_data=input_data,
            **export_setting
        )


def _inner_save_plain_prune_model(
    model,
    save_path,
    input_data,
    **export_setting,
):
    """
    Function:
    Save modified prune model to deploy and fake quant onnx file.
    The two files are the same.
    """
    save_dir, save_prefix = split_dir_prefix(save_path)
    files_util.create_path(save_dir)
    deploy_file = generate_onnx_file_name(save_dir, save_prefix, 'Deploy')
    fake_quant_file = generate_onnx_file_name(save_dir, save_prefix, 'Fakequant')

    if torch.__version__ == '2.1.0':
        export_setting['opset_version'] = 16
    else:
        export_setting['opset_version'] = 11
    if version_higher_than(torch.__version__, '1.12.0'):
        export_setting['keep_initializers_as_inputs'] = True
    if version_higher_than(torch.__version__, '1.5.0') and \
        not version_higher_than(torch.__version__, '1.11.0'):
        export_setting['enable_onnx_checker'] = False

    files_util.check_files_exist([deploy_file])
    torch.onnx.export(model, input_data, deploy_file, **export_setting)
    # set file's permission 640
    os.chmod(deploy_file, files_util.FILE_MODE)
    LOGGER.logi('Get only prune model, save deploy onnx file to {}'.format(deploy_file),
                'save_compressed_retrain_model')

    files_util.check_files_exist([fake_quant_file])
    torch.onnx.export(model, input_data, fake_quant_file, **export_setting)
    # set file's permission 640
    os.chmod(fake_quant_file, files_util.FILE_MODE)
    LOGGER.logi('Get only prune model, save fake quant onnx file to {}'.format(fake_quant_file),
                'save_compressed_retrain_model')


def _inner_save_compressed_retrain_model(
    model,
    record_file,
    save_path,
    input_data,
    **export_setting
):
    """
    Function:
    save modified prune and retrain quantize to fake quant onnx file
    and deploy onnx file, the execution steps are same with quant.
    """
    # create a reference to the de-parallel model, which all modification will be applied on.
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel,)):
        model_deparallel = model.module
    else:
        model_deparallel = model

    # preprocess retrain model
    _preprocess_retrain_model(model_deparallel, input_data)

    # save model to fake quant and deploy
    modified_onnx_file = BytesIO()
    Parser.export_onnx(model_deparallel, input_data, modified_onnx_file, export_setting)

    graph = Parser.parse_net_to_graph(modified_onnx_file)

    record_parser = RecordFileParser(
        record_file,
        graph,
        'modified_onnx_file after prune, retrain',
        enable_prune=False,
        enable_quant=True
    )
    if record_parser.is_records_empty():
        raise RuntimeError(
            "record_file is empty, no layers to be quantized. Please "
            "confirm the process of retrain quantification: whether "
            "the inference process is omitted after the training!")
    records, _ = record_parser.parse()

    _generate_model(graph, records, save_path)


def check_only_prune(model, record_file):
    """
    Function:
    Check if model has no prune and record_file has no quant factor.

    Parameters:
    model: prune and retrain model. Notes may not retrain.
    record_file: temporary file to store producer, consumer and scale, offset.

    Return:
    bool. True or False.
    True: model has no amct retrain op and record_file has no quant factor.
    False: otherwise
    """
    # step1. check model, should have no amct retrain op.
    model_has_quant = True
    try:
        ModuleHelper(model).check_amct_retrain_op()
    except RuntimeError as exception:
        LOGGER.logi(str(exception), "save_compressed_retrain_model")
        model_has_quant = False
    # step2. check record_file, should have records about prune.
    record_parser = RecordFileParser(
        record_file=record_file,
        graph=None,
        model_name='check only prune',
        enable_quant=False,
        enable_prune=True
    )
    if not model_has_quant and not record_parser.is_records_empty():
        return True
    return False


def _modify_original_model_to_prune(model, graph):
    """
    Function:
    modify the original model to selective prune retrain.
    used both in prune
    Parameter:
    model: original model.
    graph: inner graph.
    Return:
    a modified model.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = 'cpu'

    optimizer = opt.GraphOptimizer()
    optimizer.add_pass(opt.InsertRetrainPrunePass(device))
    optimizer.do_optimizer(graph, model)


def _preprocess_prune_model(model):
    """
    Function: process prune model before saving as follows:
            delete selective prune module
    Inputs:
        model: selective prune model
    Returns:
        model_copy: a model processed
    """
    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.DeleteRetrainPrunePass())
    optimizer.do_optimizer(model, None)


def _check_type_in_container(container, data_type):
    for element in container:
        if not isinstance(element, data_type):
            raise RuntimeError("container element type error {}".format(data_type))


def _check_negative_integers(lst):
    for item in lst:
        if item < 0:
            raise RuntimeError('dynamic_axes value is invalid,'
                'The int value of axis indicators cannot be a negative number.')


def _check_param_onnx_export(**export_setting):
    if not export_setting.get('dynamic_axes'):
        return
    dynamic_axes = export_setting.get('dynamic_axes')
    if not isinstance(dynamic_axes, dict):
        raise RuntimeError("dynamic_axes type must be dict")

    for key in dynamic_axes.keys():
        if not isinstance(key, str):
            raise RuntimeError("dynamic_axes key must be str")

    for val in dynamic_axes.values():
        if not isinstance(val, dict) and not isinstance(val, list):
            raise RuntimeError("dynamic_axes value must be dict or list")

        if isinstance(val, list):
            _check_type_in_container(val, int)
            _check_negative_integers(val)

        if isinstance(val, dict):
            _check_type_in_container(val.keys(), int)
            _check_negative_integers(val.keys())
            _check_type_in_container(val.values(), str)
