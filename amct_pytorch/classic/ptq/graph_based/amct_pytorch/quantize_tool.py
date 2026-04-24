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

import amct_pytorch.graph_based_compression.amct_pytorch.optimizer as opt
from ..amct_pytorch.ada_round.ada_round_groups import get_ada_round_groups
from ..amct_pytorch.utils.log import LOGGER
from ..amct_pytorch.common.utils.check_params import check_params
from ..amct_pytorch.common.utils import files as files_util
from ..amct_pytorch.configuration.configuration import Configuration
from ..amct_pytorch.custom_op.recorder.recorder import Recorder
from ..amct_pytorch.configuration.retrain_config import RetrainConfig

from ..amct_pytorch.parser.parser import Parser
from ..amct_pytorch.parser.parse_record_file import RecordFileParser
from ..amct_pytorch.utils.save import save_onnx_model
from ..amct_pytorch.utils.save import generate_onnx_file_name
from ..amct_pytorch.utils.save import split_dir_prefix
from ..amct_pytorch.utils.model_util import ModuleHelper
from ..amct_pytorch.utils.model_util import load_pth_file
from ..amct_pytorch.utils.model_util import get_node_output_info
from ..amct_pytorch.utils.singleton_record import SingletonScaleOffsetRecord
from ..amct_pytorch.utils.vars import NUM_BITS
from ..amct_pytorch.ada_round.ada_round_optimize import optimize_alpha


@check_params(config_file=str,
              model=torch.nn.Module,
              skip_layers=(list, type(None)),
              batch_num=int,
              activation_offset=bool,
              config_defination=(type(None), str))
def create_quant_config(
        config_file,
        model,
        input_data,
        skip_layers=None,
        batch_num=1,
        activation_offset=True,
        config_defination=None):
    """
    Function: Create quantize configuration json file for amct_pytorch tool
    Parameter: config_file: file path of quantize configuration json file
               model: user mode instance of Torch.nn.Module
               input_data: used to compile model, can be ramdom data
               skip_layers: list of layers that not do quantize, default empty
               batch_num: number of batch that used for calibration
               activation_offset: whether activation quantize with offset
               config_defination: simply config file from user to set
    Return: None
    """
    # cope inputs
    config_file = os.path.realpath(config_file)

    try:
        model = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "create_quant_config")

    ModuleHelper(model).check_amct_op()

    # parse to graph
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx)
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)

    # create config file for quantizetion
    Configuration.create_quant_config(config_file, graph, skip_layers,
                                      batch_num, activation_offset,
                                      config_defination)
    LOGGER.logi(f'Create quant config file {config_file} success.')


@check_params(config_file=str,
              record_file=str,
              model=torch.nn.Module)
def quantize_preprocess(
        config_file,
        record_file,
        model,
        input_data):
    """
    Function: Quantization input model: According to the quantization
              configuration file, insert dmq_balancer op at the specified
              position of torch.nn.Module.
    Inputs:
        config_file: a string, Name of quantized configuration file (including
                   path information).
        record_file: a string, the name of file recording quantization factor.
        graph: a torch.nn.Module.
        input_data: used to compile model, can be ramdom data
    Returns:
        None
    """
    files_util.is_valid_name(config_file, 'config_file')
    files_util.is_valid_name(record_file, 'record_file')
    config_file = os.path.realpath(config_file)
    if not os.path.exists(config_file):
        raise OSError(f'file ({config_file}) does not exist!')
    file_realpath = files_util.create_empty_file(record_file, check_exist=True)
    SingletonScaleOffsetRecord().reset_singleton(file_realpath)

    model = ModuleHelper.deep_copy(model)

    ModuleHelper(model).check_amct_op()

    # parse to graph
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx)
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)

    # check dmq_balancer config
    Configuration().init(config_file, file_realpath, graph)
    quant_config = Configuration().get_quant_config()
    Configuration().check_quant_config_dmq_balancer(quant_config)

    torch_recorder = Recorder(file_realpath, enable_dmq_balancer=True)
    optimizer = opt.ModelOptimizer()
    # BN fusion
    if Configuration().get_fusion_switch():
        optimizer.add_pass(opt.ConvBnFusionPass(Configuration))
    # do dmq_balancer
    optimizer.add_pass(opt.InsertDMQBalancerPass(torch_recorder))
    optimizer.do_optimizer(model, graph)
    return model


@check_params(config_file=str,
              modfied_onnx_file=str,
              record_file=str,
              model=torch.nn.Module,
              input_names=(list, type(None)),
              output_names=(list, type(None)),
              dynamic_axes=(dict, type(None)))
def quantize_model(
        config_file,
        modfied_onnx_file,
        record_file,
        model,
        input_data,
        input_names=None,
        output_names=None,
        dynamic_axes=None):
    """
    Function: Modify user's model for calibration in inference process.
    Parameter: config_file: quantize configuration json file
               modfied_onnx_file: a string, the file export from model after
               fusion.
               record_file: temporary file to store scale and offset
               model: user pytorch model's model file
               input_data: used to compile model, can be ramdom data
               input_names: list of strings, names to assign to the
                   input nodes of the graph, in order
               output_names: names to assign to the
                   output nodes of the graph, in order
               dynamic_axes: a dictionary to specify dynamic axes of
                   input/output
    Return: model: modified pytorch model for calibration inference.
    """
    model = inner_quantize_model(config_file=config_file,
                                 modfied_onnx_file=modfied_onnx_file,
                                 record_file=record_file,
                                 model=model,
                                 input_data=input_data,
                                 input_names=input_names,
                                 output_names=output_names,
                                 dynamic_axes=dynamic_axes,
                                 dump_config=None)

    return model


def inner_quantize_model(
        config_file,
        modfied_onnx_file,
        record_file,
        model,
        input_data,
        input_names=None,
        output_names=None,
        dynamic_axes=None,
        dump_config=None,
        weight_fakequant=True):
    """
    Function: Modify user's model for calibration in inference process.
    Parameter: config_file: quantize configuration json file
               modfied_onnx_file: a string, the file export from model after
               fusion.
               record_file: temporary file to store scale and offset
               model: user pytorch model's model file
               input_data: used to compile model, can be ramdom data
               input_names: list of strings, names to assign to the
                   input nodes of the graph, in order
               output_names: names to assign to the
                   output nodes of the graph, in order
               dynamic_axes: a dictionary to specify dynamic axes of
                   input/output
               dump_config: class, contains dump_dir and batch_num.
                   dump_dir: the dump dir of input data tensor.
               weight_fakequant: for amc feature, weight do not need fake quant.
    Return: model: modified pytorch model for calibration inference.
    """
    config_file = os.path.realpath(config_file)
    modfied_onnx_file = os.path.realpath(modfied_onnx_file)
    SingletonScaleOffsetRecord().reset_singleton(record_file)

    try:
        model = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "quantize_model")

    ModuleHelper(model).check_amct_op()

    # parse to graph
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx,
                       {'input_names': input_names,
                        'output_names': output_names,
                        'dynamic_axes': dynamic_axes})
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)

    Configuration().init(config_file, record_file, graph)
    quant_config = Configuration().get_quant_config()
    if Configuration().check_dmq_balancer_enable(quant_config):
        if os.path.exists(record_file):
            record_parser = RecordFileParser(record_file, graph, modfied_onnx_file)
        if not os.path.exists(record_file) or record_parser.is_records_empty():
            raise RuntimeError(
                "config_file indicates dmq_balancer, but record_file is empty. "
                "please check quant_preprocess and calibration is done!")
        records, _ = record_parser.parse()
    else:
        record_file = files_util.create_empty_file(record_file, check_exist=True)
        records = {}

    # do weight quantize
    _inner_quantize_weight(model, graph, weight_fakequant, records, input_data)
    node_info = get_node_output_info(model, input_data)
    save_onnx_model(graph, modfied_onnx_file, None, node_info)

    # quantize
    _inner_quantize_data(model, graph, record_file, dump_config, records)

    return model


def _inner_quantize_weight(model, graph, weight_fakequant, records, input_data):
    """
    Function: do weight quantize process.

    Args:
    model: torch.nn.Module, user's model to quantize.
    graph: inner graph.
    weight_fakequant: for amc feature, weight do not need fake quant.
    """
    # fuse and export modified onnx
    optimizer = opt.ModelOptimizer()
    # add bn fusion pass only when get_fusion_switch set to true
    if Configuration().get_fusion_switch():
        optimizer.add_pass(opt.ConvBnFusionPass(Configuration))
    optimizer.add_pass(opt.WeightsCalibrationPass(records, weight_fakequant=weight_fakequant))

    optimizer.do_optimizer(model, graph)
    
    # for adaRound feature, Finetune the rounding mode.
    groups = get_ada_round_groups(graph, Configuration().get_quant_config())
    optimize_alpha(model, input_data, groups, graph)


def _inner_quantize_data(model, graph, record_file, dump_config, records):
    """
    Function: do data quantize process.

    Args:
    model: torch.nn.Module, user's model to quantize.
    graph: inner graph.
    record_file: string, record file real path.
    dump_config: DumpConfig class.
    """
    # quantize
    torch_recorder = Recorder(record_file)
    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.InsertCaliQuantPass(torch_recorder, records, dump_config))
    optimizer.do_optimizer(model, graph)


@check_params(modfied_onnx_file=str, record_file=str, save_path=str)
def save_model(modfied_onnx_file, record_file, save_path):
    """
    Function: save modfied_onnx_file to fakequant_onnx_file and
        deploy_onnx_file.
    Parameter:
        modfied_onnx_file: a string, the file export from quantize_model
        record_file: a string, path of file containing the scale and offset.
        save_path: a string, the path where to store model and model's name.
    Return: None

    """
    # check inputs
    files_util.is_valid_name(modfied_onnx_file, 'modfied_onnx_file')
    files_util.is_valid_name(record_file, 'record_file')
    modfied_onnx_file = os.path.realpath(modfied_onnx_file)
    record_file = os.path.realpath(record_file)
    SingletonScaleOffsetRecord().reset_singleton(record_file)

    graph = Parser.parse_net_to_graph(modfied_onnx_file)

    # parse record_file
    record_parser = RecordFileParser(record_file, graph, modfied_onnx_file)
    if record_parser.is_records_empty():
        raise RuntimeError(
            "record_file is empty, no layers to be quantized. Please "
            "ensure calibration is finished by checking information!")
    records, _ = record_parser.parse()

    Parser.write_node_attrs_extracted_from_onnx(graph, modfied_onnx_file, ['op_data_type'])
    quant_config = Configuration().get_quant_config()
    _generate_model(graph, records, save_path)


@check_params(config_file=str,
              model=torch.nn.Module,
              config_defination=(type(None), str))
def create_quant_retrain_config(
        config_file,
        model,
        input_data,
        config_defination=None):
    """
    Function: Create retrain quantize configuration json file for amct_caffe
        tool
    Parameter: config_file: file path of quantize configuration json file
               model: user mode instance of Torch.nn.Module
               input_data: used to compile model, can be ramdom data
               config_defination: simply config file from user to set
    Return: None
    """
    # cope inputs
    config_file = os.path.realpath(config_file)

    try:
        model = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "create_quant_retrain_config")

    if isinstance(model, (torch.nn.parallel.DistributedDataParallel,)):
        model = model.module

    ModuleHelper(model).check_amct_op()

    # parse to graph
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx)
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)

    if config_defination is None:
        RetrainConfig.create_default_retrain_config(config_file, graph)
    # create config file for quantizetion
    else:
        RetrainConfig.create_quant_retrain_config(
            config_file,
            graph,
            config_defination)
    LOGGER.logi(f'Create quant retrain config file {config_file} success.')


@check_params(config_file=str,
              model=torch.nn.Module,
              record_file=str)
def create_quant_retrain_model(
        config_file,
        model,
        record_file,
        input_data):
    """
    Function: Modify user's model for retrain in train and inference process.
    Parameter: config_file: retrain quantize configuration json file
               model: user pytorch model's model file
               record_file: temporary file to store scale and offset
               input_data: used to compile model, can be ramdom data
    Return: model: modified pytorch model for retrain.
    """
    config_file = os.path.realpath(config_file)
    record_file = files_util.create_empty_file(record_file, check_exist=True)
    SingletonScaleOffsetRecord().reset_singleton(record_file)

    try:
        model_copy = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "create_quant_retrain_model")
        model_copy = model

    # create a reference to the de-parallel model, which all modification will be applied on.
    if isinstance(model_copy, (torch.nn.parallel.DistributedDataParallel,)):
        model_deparallel = model_copy.module
    else:
        model_deparallel = model_copy

    _modify_original_model(model_deparallel, input_data, config_file, record_file)

    return model_copy


@check_params(config_file=str,
              model=torch.nn.Module,
              record_file=str,
              pth_file=str,
              state_dict_name=(str, type(None)))
def restore_quant_retrain_model(
        config_file,
        model,
        record_file,
        input_data,
        pth_file,
        state_dict_name=None):
    """
    Function: Modify user's model and restore retrain network from last
        checkpoint.
    Parameter: config_file: retrain quantize configuration json file
               model: user pytorch model's model file
               record_file: temporary file to store scale and offset
               input_data: used to compile model, can be ramdom data
               pth_file: user quant aware training checkpoint file path
               state_dict_name: key value of weight parameter in pth_file
    Return: model: modified pytorch model for retrain.
    """
    config_file = os.path.realpath(config_file)
    pth_file = os.path.realpath(pth_file)
    record_file = files_util.create_empty_file(record_file, check_exist=True)
    SingletonScaleOffsetRecord().reset_singleton(record_file)

    try:
        model = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "restore_quant_retrain_model")

    _modify_original_model(model, input_data, config_file, record_file)

    model = load_pth_file(model, pth_file, state_dict_name)

    return model


@check_params(config_file=str,
              model=torch.nn.Module,
              record_file=str,
              save_path=str,
              input_names=(list, type(None)),
              output_names=(list, type(None)),
              dynamic_axes=(dict, type(None)))
def save_quant_retrain_model(
        config_file,
        model,
        record_file,
        save_path,
        input_data,
        input_names=None,
        output_names=None,
        dynamic_axes=None):
    """
    Function: save modfied_onnx_file to fakequant_onnx_file and
        deploy_onnx_file.
    Parameter:
        config_file: retrain quantize configuration json file
        model: retrain model
        record_file: temporary file to store scale and offset
        save_path: a string, the path where to store model and model's name.
        input_data: used to compile model, can be ramdom data
        input_names: list of strings, names to assign to the input nodes of
            the graph, in order
        output_names: names to assign to the output nodes of the graph
            in order
        dynamic_axes: a dictionary to specify dynamic axes of input/output
    Return: None.
    """
    # check inputs
    files_util.is_valid_name(config_file, 'config_file')
    files_util.is_valid_name(record_file, 'record_file')
    config_file = os.path.realpath(config_file)
    record_file = os.path.realpath(record_file)
    SingletonScaleOffsetRecord().reset_singleton(record_file)
    ModuleHelper(model).check_amct_retrain_op()

    # copy model to a new one
    model_copy = _copy_modified_model(model, record_file)

    # create a reference to the de-parallel model, which all modification will be applied on.
    if isinstance(model_copy, (torch.nn.parallel.DistributedDataParallel,)):
        model_copy = model_copy.module

    # preprocess retrain model
    _preprocess_retrain_model(model_copy, input_data, config_file)

    # save model to fakequant and deploy
    modfied_onnx_file = BytesIO()
    Parser.export_onnx(model_copy, input_data, modfied_onnx_file,
                       {'input_names': input_names,
                        'output_names': output_names,
                        'dynamic_axes': dynamic_axes})

    graph = Parser.parse_net_to_graph(modfied_onnx_file)

    record_parser = RecordFileParser(
        record_file, graph, 'modfied_onnx_file after retrain')
    if record_parser.is_records_empty():
        raise RuntimeError(
            "record_file is empty, no layers to be quantized. Please "
            "confirm the process of retrain quantification: whether "
            "the inference process is omitted after the training!")
    records, _ = record_parser.parse()

    _generate_model(graph, records, save_path)


def _copy_modified_model(model, record_file):
    """
    Function: deep copy the original model to model_copy.
    Inputs:
        model: original model
        record_file: temporary file to store scale and offset
    Returns:
        model: a model processed
    """
    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.SetRecorderPass())
    optimizer.do_optimizer(model, None)
    try:
        model_copy = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        model_copy = model
        LOGGER.logw(exception, "save_quant_retrain_model")
    else:
        torch_recorder = Recorder(record_file)
        optimizer = opt.ModelOptimizer()
        optimizer.add_pass(opt.SetRecorderPass(torch_recorder))
        optimizer.do_optimizer(model, None)
    return model_copy


def parameters(module):
    """Returns an iterator over module parameters."""
    def _parameters(recurse=True):
        for name, param in module.named_parameters(recurse=recurse):
            if 'acts_comp_reuse' not in name:
                yield param
    return _parameters


def _modify_original_model(model, input_data, config_file, record_file):
    """
    Function: Modify the original model to quantify retraining.
    Inputs:
        model: original model
        input_data: used to compile model, can be ramdom data
        config_file: retrain quantize configuration json file
        record_file: temporary file to store scale and offset
    Returns:
        model: a model processed
    """
    ModuleHelper(model).check_amct_op()

    # parse to graph
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx)
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)
    RetrainConfig.init_retrain(config_file, record_file, graph)

    _modify_original_model_to_quant(model, record_file, graph)


def _modify_original_model_to_quant(model, record_file, graph):
    """
    Function:
    modify the original model to quant retrain.
    used both in quant and compress.
    Parameter:
    model: original model.
    record_file: temporary file to store quant or compressed.
    graph: inner graph.
    Return:
    a modified model.
    """
    # quantize
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = 'cpu'
    torch_recorder = Recorder(record_file)
    optimizer = opt.GraphOptimizer()
    optimizer.add_pass(opt.InsertRetrainPass(device))
    optimizer.add_pass(opt.ShareActCompPass())
    optimizer.add_pass(opt.InsertRetrainQuantPass(torch_recorder, device))
    optimizer.do_optimizer(graph, model)

    # Rewrite the parameters function to eliminate the shared variables
    for item in model.named_modules():
        item[1].parameters = parameters(item[1])


def _preprocess_retrain_model(model, input_data, config_file=None):
    """
    Function: process retrain model before saving as follows:
            1. delete retain module
            2. fuse bn
    Inputs:
        model: retrain model
        input_data: used to compile model, can be ramdom data
        config_file: retrain quantize configuration json file
    Returns:
        model_copy: a model processed
    """
    # delete retrain pass
    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.DeleteRetrainPass())
    optimizer.add_pass(opt.RepalceSyncBNPass())
    optimizer.do_optimizer(model, None)

    # fuse bn in model
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx)
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)
    if config_file is not None:
        retrain_config = RetrainConfig().parse_retrain_config(config_file, graph)
        if not _check_config_consistency(retrain_config,
                                        RetrainConfig.retrain_config):
            raise RuntimeError(
                "The config_file is inconsistent with the config_file input by "
                "the create_quant_retrain_model or restore_quant_retrain_model "
                "interface!")

    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.ConvBnFusionPass(RetrainConfig))
    optimizer.do_optimizer(model, graph)


def _check_config_consistency(retrain_config, single_instance_config):
    ''' Verify the consistency between retrain_config and
        single_instance_config.
    '''
    def _check_item_consistency(value1, value2):
        ''' Verify the consistency between value1 and value2. '''
        if not isinstance(value1, type(value2)):
            return False
        if isinstance(value1, dict):
            for key, value in value1.items():
                if key not in value2.keys():
                    return False
                consistency = \
                    _check_item_consistency(value, value2.get(key))
                if not consistency:
                    return False
        else:
            if value1 != value2:
                return False
        return True

    if retrain_config.keys() != single_instance_config.keys():
        return False

    consistency = True
    for key, value in retrain_config.items():
        consistency = _check_item_consistency(
            value, single_instance_config.get(key))
        if not consistency:
            return False
    return consistency


def _generate_model(graph, records, save_path):
    ''' Generate deploy and fakequant onnx model. '''
    save_dir, save_prefix = split_dir_prefix(save_path)
    files_util.create_path(save_dir)

    # do common pass
    optimizer = opt.GraphOptimizer()
    optimizer.add_pass(opt.GemmTransBOptimizePass(records))
    optimizer.add_pass(opt.ApplyDMQBalancerPass(records))
    optimizer.add_pass(opt.InsertQuantPass(records))
    optimizer.add_pass(opt.InsertWeightQuantPass(records))
    optimizer.add_pass(opt.InsertDequantPass(records))
    optimizer.add_pass(opt.MultQuantOptimizerPass(records))
    optimizer.add_pass(opt.InsertBiasQuantPass(records))
    optimizer.add_pass(opt.QuantFusionPass(records))
    optimizer.do_optimizer(graph)

    # copy graph after common pass done
    graph_copy = graph.deep_copy()

    # generate and save deploy model
    optimizer = opt.GraphOptimizer()
    optimizer.add_pass(opt.ReplaceRNNPass(records))
    optimizer.do_optimizer(graph)
    deploy_file = generate_onnx_file_name(save_dir, save_prefix, 'Deploy')
    save_onnx_model(graph, deploy_file, 'Deploy', None, ['op_data_type'])

    # generate and save fakequant model
    optimizer = opt.GraphOptimizer()
    optimizer.add_pass(opt.InsertRNNFakeQuantPass(records))
    optimizer.add_pass(opt.ReplaceQuantPass(records))
    optimizer.add_pass(opt.ReplaceWeightQuantPass(records)) # AscendWeightQuant op only
    optimizer.add_pass(opt.WeightFakequantPass(records)) # int8 case only
    optimizer.add_pass(opt.ReplaceDequantPass(records))
    optimizer.add_pass(opt.ReplaceAntiQuantPass(records))
    optimizer.add_pass(opt.ReplaceBiasQuantPass(records))
    optimizer.do_optimizer(graph_copy)
    fakequant_file = generate_onnx_file_name(save_dir, save_prefix, 'Fakequant')
    save_onnx_model(graph_copy, fakequant_file, 'Fakequant', None, ['op_data_type'])


def inner_generate_fakequant_module(graph, model, records, numbits):
    """ generate the fake quant pytorch module"""
    optimizer = opt.ModelOptimizer()
    if Configuration().get_fusion_switch():
        optimizer.add_pass(opt.ConvBnFusionPass(Configuration))
    optimizer.add_pass(opt.InsertFakequantConvPass(records, numbits))
    optimizer.add_pass(opt.InsertFakequantConvTransposePass(records, numbits))
    optimizer.add_pass(opt.InsertFakequantLinearPass(records, numbits))
    optimizer.add_pass(opt.InsertFakequantAvgPool2dPass(records, numbits))
    optimizer.add_pass(opt.WeightFakequantModulePass(records, numbits))
    optimizer.add_pass(opt.BiasFakequantModulePass(records, numbits))
    optimizer.do_optimizer(model=model, graph=graph)
    return model


def generate_fakequant_module(
        model,
        config_file,
        record_file,
        input_data,
        input_names=None,
        output_names=None,
        dynamic_axes=None):
    """ generate the fake quant pytorch module"""
    files_util.is_valid_name(record_file, 'record_file')
    record_file = os.path.realpath(record_file)
    config_file = os.path.realpath(config_file)

    files_util.is_valid_name(record_file, 'record_file')

    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx,
                       {'input_names': input_names,
                        'output_names': output_names,
                        'dynamic_axes': dynamic_axes})

    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)
    Configuration().init(config_file, record_file, graph)

    # parse record file
    record_parser = RecordFileParser(
        record_file, graph, 'modified_onnx_file after calibration')
    if record_parser.is_records_empty():
        raise RuntimeError(
            "record_file is empty, no layers to be quantized. Please "
            "ensure calibration is finished by checking information!")
    records, _ = record_parser.parse()

    # For AMC Feature, each layer can be same dst_type.
    # optimize code follow up.
    numbits = fetch_num_bits()

    model = inner_generate_fakequant_module(graph, model, records, numbits)

    return model


def fetch_num_bits():
    """
    Function: fetch num bits from config json.
    """
    # need to optimize, current each layer is same.
    quant_config = Configuration().get_quant_config()
    layer_name = Configuration.get_layers_name(quant_config)
    activation_quant_params = quant_config[layer_name[0]]['activation_quant_params']
    weight_quant_params = quant_config[layer_name[0]]['weight_quant_params']
    if activation_quant_params[NUM_BITS] != weight_quant_params[NUM_BITS]:
        raise RuntimeError('Current activation and weight num bits should be same!')
    return activation_quant_params[NUM_BITS]


def inner_fuse_bn(
        model,
        config_file,
        record_file,
        input_data,
        input_names=None,
        output_names=None,
        dynamic_axes=None):
    """ generate the bn fused pytorch module"""
    files_util.is_valid_name(record_file, 'record_file')
    record_file = os.path.realpath(record_file)
    config_file = os.path.realpath(config_file)

    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data, tmp_onnx,
                       {'input_names': input_names,
                        'output_names': output_names,
                        'dynamic_axes': dynamic_axes})

    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)
    Configuration().init(config_file, record_file, graph)

    optimizer = opt.ModelOptimizer()
    if Configuration().get_fusion_switch():
        optimizer.add_pass(opt.ConvBnFusionPass(Configuration))
    optimizer.do_optimizer(model=model, graph=graph)
    return model


def add_dump_operations(model, dump_config):
    """
    Function: ONLY dump the input data of QUANTIZABLE_TYPES.
    Notes that torch_recorder, graph both can be None. DO NOT NEED BNFusion.

    Args:
    dump_config: DumpConfig class, contains dump_dir, batch_num.

    Return:
    None.
    """
    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.InsertCaliQuantPass(torch_recorder=None, dump_config=dump_config, mode='dump'))
    optimizer.do_optimizer(model, graph=None)
