#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Functions and classes to handle amct command line arguments.

"""

import os
import argparse
from ...utils.log import LOGGER
from ..utils.vars_util import FEATURE_LIST


class AMCTHelpFormatter(argparse.HelpFormatter):
    '''
    helper calss
    '''
    def _split_lines(self, text, width):
        '''
        Function: _split_lines
        '''
        return super()._split_lines(text, width) + ['']


class ArgumentBase:
    '''
    Argument Base calss
    '''
    def __init__(self):
        pass


####################
#   Required args  #
####################
class Model(ArgumentBase):
    '''
    Model calss
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser.add_argument('--model', type=str, default=None, action=GetRealPath,
            required=True, help='The path to the input model. original model for calibration, qat model for convert')
        super().__init__()


class OutputNodeNames(ArgumentBase):
    '''
    OutputNodeNames
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser.add_argument('--outputs', type=str, action=ProcessMultiNodeNames,
                required=True, help='The name for the output tensors of the original model. E.g.:output0:0;outoput1:0')
        super().__init__()


class SavePath(ArgumentBase):
    '''
    required class
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser.add_argument('--save_path', type=str, action=GetRealPath,
            required=True, help='The path to save the results, which should contain the prefix of the result model.'
            ' For example: \"./results/model_prefix\"')
        super().__init__()


####################
#   optional args  #
####################
class InputsShapes(ArgumentBase):
    '''
    optiaonal class
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser.add_argument('--input_shape', type=str, default=None, action=ProcessDataShape,
            help='Shape of input data. Separate multiple nodes with semicolons (;).'
            ' Use double quotation marks (") to enclose each argument.'
            ' E.g.: "input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"')
        super().__init__()


class ModelWeights(ArgumentBase):
    '''
    optiaonal class
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser.add_argument('--weights', type=str, default=None, action=GetRealPath,
            required=True, help='model weights (only for caffe model).'
            ' Use double quotation marks (") to enclose each argument.')
        super().__init__()


class DataTypes(ArgumentBase):
    '''
    optiaonal class
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser.add_argument('--data_types', type=str, default=None, action=ProcessDataType,
            help='The dtype of the input data. Separate multiple nodes with semicolons (;).'
            ' Use double quotation marks (") to enclose each argument.'
            ' E.g.: "float32;float64"')
        super().__init__()


class DataPaths(ArgumentBase):
    '''
    optiaonal class
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser.add_argument('--data_dir', type=str, default=None, action=ProcessMultiDataPath,
            help='The path to the processed binary datasets.'
            ' For a multi-input model, different input data must be stored in different directories.'
            ' Names of all files in each directory must be sorted in ascending lexicographic order.'
            ' Use double quotation marks (") to enclose each argument.'
            ' E.g.: "data/input1/;data/input2/"')
        super().__init__()


class ConfigFilePath(ArgumentBase):
    '''
    optiaonal class
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser.add_argument(
            '--calibration_config', type=str, action=GetRealPath, default=None,
            help='The path to the user customized .cfg config_defination file. Default is set to None.')
        super().__init__()


class BatchNum(ArgumentBase):
    '''
    optiaonal class
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser.add_argument('--batch_num', type=int, default=None, action=NonNegativeNum,
            help='The number of data batches used to run PTQ calibration. Default is set to 1.')
        super().__init__()


class EvaluatorPath(ArgumentBase):
    '''
    optiaonal class
    '''
    def __init__(self, parser):
        self.parser = parser
        self.parser = parser.add_argument('--evaluator', type=str, default=None, action=GetRealPath,
            help='Python script contains evaluator based on base class "Evaluator".')
        super().__init__()


class GPUID(ArgumentBase):
    '''
    GPUID: gpu id.
    '''

    def __init__(self, parser):
        self.parser = parser
        self.parser = parser.add_argument('--gpu_id', type=int, default=None, action=NonNegativeNum,
            help='use gpu accelerateing.')
        super().__init__()


class Feature(ArgumentBase):
    '''
    feature: specify which features to enable.
    '''

    def __init__(self, parser):
        self.parser = parser
        self.parser = parser.add_argument('--feature', type=str, default='ptq', action=ParseFeature,
            help='Specify features enabled. Only \"approximate\" and \"ptq\" are optional features. '
                 'Default is set to ptq. E.g.: \"approximate|ptq\"')
        super().__init__()


class ParseFeature(argparse.Action):
    '''
    optiaonal class
    '''
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        features = ['ptq']
        if values is not None:
            features = values.split('|')
            for feature in features:
                if not feature:
                    raise ValueError('Feature is empty. '
                                     'Only support feature in {}.'.format(FEATURE_LIST))
                if feature not in FEATURE_LIST:
                    raise ValueError('Feature {} is not supported. '
                                     'Only support feature in {}.'.format(feature, FEATURE_LIST))
            if not features:
                raise ValueError('No feature is enabled.')
        setattr(namespace, self.dest, features)


class GetRealPath(argparse.Action):
    '''
    optiaonal class
    '''
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            values = os.path.realpath(values)
        setattr(namespace, self.dest, values)


class ProcessMultiNodeNames(argparse.Action):
    '''
    optiaonal class
    '''
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            node_names = values.split(';')
            values = node_names
        setattr(namespace, self.dest, values)


class ProcessDataShape(argparse.Action):
    '''
    optiaonal class
    '''
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            values = process_data_shape(values)
        setattr(namespace, self.dest, values)


class ProcessDataType(argparse.Action):
    '''
    optiaonal class
    '''
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            values = values.split(';')
        setattr(namespace, self.dest, values)


class ProcessMultiDataPath(argparse.Action):
    '''
    optiaonal class
    '''
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            values = process_multi_data_path(values)
        setattr(namespace, self.dest, values)


class NonNegativeNum(argparse.Action):
    '''
    NonNegativeNum: gpu id.
    '''

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            if values < 0:
                raise ValueError("Invalid batch_num {} was given, please check.".format(values))
        setattr(namespace, self.dest, values)


def batch_num_check(args_var):
    '''
    Function: check if the arguments match some special limitations
    Arguments:
    args_var: the parsed arguments
    Return: None
    '''

    if args_var.batch_num is not None and args_var.calibration_config is not None:
        raise ValueError("batch num and calibration config are exclusive, only one can be configured.")

    if args_var.batch_num is None and args_var.calibration_config is None:

        LOGGER.logw('both batch num and config file are not set, set to default value: 1')

    if args_var.batch_num is None:
        args_var.batch_num = 1


def calibration_args_checker(args_var):
    '''
    Function: check if the arguments match some special limitations
    Arguments:
    args_var: the parsed arguments
    Return: None
    '''

    if args_var.evaluator is None:
        if args_var.input_shape is None:
            raise ValueError("'--input_shape' is required when '--evaluator' option is not given.")
        if args_var.data_dir is None:
            raise ValueError("'--data_dir' is required when '--evaluator' option is not given.")
        if args_var.data_types is None:
            raise ValueError("'--data_types' is required when '--evaluator' option is not given.")

        shape_len = len(args_var.input_shape)
        path_len = len(args_var.data_dir)
        type_len = len(args_var.data_types)

        if shape_len != path_len or shape_len != type_len or path_len != type_len:
            raise ValueError("The number of arguments for '--input_shape', '--data_dir'"
            " and '--data_types' must be the same!")
    else:
        if args_var.input_shape is not None:
            raise ValueError("'--input_shape' is not required when '--evaluator' option is given.")
        if args_var.data_dir is not None:
            raise ValueError("'--data_dir' is not required when '--evaluator' option is given.")
        if args_var.data_types is not None:
            raise ValueError("'--data_types' is not required when '--evaluator' option given.")

    batch_num_check(args_var)


def process_data_shape(values):
    """ Process data_shape to several shape info.
    For example, "input_name1:n1,c1,h1,w1;input_name2:n1,c1,h1,w1" to
    {"input_name1": [n1,c1,h1,w1], "input_name2": [n1,c1,h1,w1]},

    or "input_name1:0:n1,c1,h1,w1;input_name2:0:0:n1,c1,h1,w1" to
    {"input_name1:0": [n1,c1,h1,w1], "input_name2:0:0": [n1,c1,h1,w1]}

    Args:
        values (string): data_shape like "input_name1:n1,c1,h1,w1;input_name2:n1,c1,h1,w1"
                                or "input_name1:0:n1,c1,h1,w1;input_name2:0:0:n1,c1,h1,w1"

    Raises:
        ValueError: format is invalid

    Returns:
        dict: new shape info, key is input_name and value is shape(list)
    """
    data_shapes_str = values.split(';')
    input_dict = {}
    for data_item in data_shapes_str:
        data_list = data_item.split(':')
        data_name = ':'.join(data_list[:-1])
        data_shape_str = data_list[-1]
        dims = data_shape_str.split(',')
        if len(data_list) < 2 or len(data_name) == 0 or len(dims) == 0:
            raise ValueError(
                "Invalid input_shape. Input name and shapes of each input node"
                " should be ':' split. E.g.: input_name1:n1,c1,h1,w1"
                " or input_name2:0:n1,c1,h1,w1")
        data_shape = [int(shape_dim) for shape_dim in dims]
        input_dict[data_name] = data_shape

    return input_dict


def process_multi_data_path(values):
    """Process data_path to several path info.

    Args:
        values (string): data_path like "input_name1:n1,c1,h1,w1;input_name2:n1,c1,h1,w1"

    Returns:
        list of string: new path info.
    """
    data_paths = values.split(';')
    data_paths = [os.path.realpath(data_path) for data_path in data_paths]
    return data_paths
