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

__all__ = ["RecordFileParserBase", "RecordManager"]

import os
import collections
import numpy as np
from google.protobuf import text_format # pylint: disable=E0401

from .util import is_invalid
from ..config.field import WTS_SUPPORT_NUM_BITS, ACT_SUPPORT_NUM_BITS
from ..utils.vars_util import DATA_OFFSET_RANGE, DATA_OFFSET_RANGE_INT16
from ..utils.vars_util import RNN_TENSOR_NUM, RNN_LAYER_TYPE
from ...utils.log import LOGGER

SCALE_RANGE = [0.0, float("inf")]
CLUSTER_CENTER_RANGE = [-128, 127]
CLUSTER_CENTER_LEN_RANGE = [16, 32]
WEIGHT_OFFSET_RANGE = [0, 0]
SHIFT_N_RANGE = [1, 16]
TENSOR_BALANCE_FACTOR_RANGE = [np.finfo(np.float32).eps, 1 / np.finfo(np.float32).eps]
DATA_OFFSET = 'data_offset'
WEIGHT_SCALE = 'weight_scale'
WEIGHT_OFFSET = 'weight_offset'
SHIFT_N = 'shift_n'
CLUSTER_CENTER = 'cluster_center'
H_OFFSET = 'h_offset'
RECURRENCE_WEIGHT_OFFSET = 'recurrence_weight_offset'
GRAPH_QUERIER = 'graph_querier'
GRAPH_CHECKER = 'graph_checker'
OP_DATA_TYPE = 'op_data_type'
DATA_SCALE = 'data_scale'
INT4 = 'INT4'


RecordFileParsedResult = collections.namedtuple('RecordFileParsedResult',
                                                ['layers_params', 'skip_fusion_layers'])


class RecordFileParserBase():
    """
    Function: Parse the information of compression from record_file.
    APIs: read_record_file, parse
    """
    def __init__(self, # pylint: disable=R0913
                 record_file,
                 graph,
                 model_name,
                 config,
                 records=None,
                 enable_quant=True,
                 enable_prune=False,
                 prune_idx=None):
        """
        Function: init object
        Inputs:
        record_file: a string, the file to parse.
        graph: the graph corresponding to record_file.
        model_name: a string, the model's name.
        config: a dict, including capacity, records_pb2, op_quirer,
        graph_querier, graph_checker
        """
        self.record_file = os.path.realpath(record_file)
        self.graph = graph
        self.model_name = os.path.realpath(model_name)
        self.capacity = config.get('capacity')
        if records is None:
            self.records = config.get('records_pb2').ScaleOffsetRecord()
        else:
            # tensorflow pass the InnerScaleOffsetRecord() to here
            self.records = records
        self.config = config
        self.enable_quant = enable_quant
        self.enable_prune = enable_prune
        self.prune_idx = prune_idx
        self.read_record_file(self.record_file, self.records)

    @staticmethod
    def read_record_file(record_file, records):
        ''' read record file to a proto '''
        record_file = os.path.realpath(record_file)
        with open(record_file, 'r') as fid:
            pbtxt_string = fid.read()
            try:
                text_format.Merge(pbtxt_string, records)
            except text_format.ParseError as e:
                raise RuntimeError(
                    "the record_file{%s} cannot be parsered, please ensure "\
                    "it matches with scale_offset_record.proto!"
                    % (record_file)) from e

    def is_records_empty(self):
        ''' check records  '''
        if self.enable_quant and list(self.records.record) == []:
            return True
        if self.enable_prune \
            and list(self.records.prune_record) == []:
            return True
        return False

    def parse(self, enable_shift_n=True): # pylint: disable=R0912
        """
        Function: Parse the information of compression.
        Inputs:
        enable_shift_n: a bool, whether the shift_bit if configurable.
        Returns:
        layers_params: a dictionary, the information of compression
        containing items as follows:
        data_scale_value: an array, quant factor data's scale.
        data_offset_value: an array, quant factor data's offset.
        shift_n_value: an number, quant factor N.
        weight_scale_value: an array, quant factor weight's scale.
        weight_offset_value: an array, quant factor weight offset.
        dst_type: a string, indicates bit_num(INT4 or INT8).
        skip_fusion_layers: a list containing layers not to do fusion.
        """
        if not self.enable_quant and not self.enable_prune:
            raise ValueError("Can not parse record file."
                            "because enable are both False.")
        if self.enable_quant and self.enable_prune:
            record_file_parsed_result_quant = self.parse_quant(enable_shift_n)
            record_file_parsed_result_prune = self.parse_prune()
            return record_file_parsed_result_quant, record_file_parsed_result_prune
        if self.enable_prune and not self.enable_quant:
            record_file_parsed_result = self.parse_prune()
        else:
            record_file_parsed_result = self.parse_quant(enable_shift_n)

        return record_file_parsed_result

    def parse_prune(self):
        '''parse record file for prune'''
        for record in self.records.prune_record:
            try:
                record.producer
            except AttributeError as e:
                raise RuntimeError('Cannot find "producer" in record file.') from e
            try:
                record.consumer
            except AttributeError as e:
                raise RuntimeError('Cannot find "consumer" in record file.') from e
            node_names = [node.name for node in record.producer] + \
                [node.name for node in record.consumer]
            try:
                for node_name in node_names:
                    if hasattr(self.graph, 'get_operation_by_name'):
                        self.graph.get_operation_by_name(node_name)
                    else:
                        self.graph.get_node_by_name(node_name)
            except KeyError as e:
                raise RuntimeError(
                    "find %s failure in %s. Please match %s and %s." %
                    (node_name, self.model_name, self.model_name,
                     self.record_file)) from e
        return RecordFileParsedResult(self.records.prune_record, None)


    def parse_quant(self, enable_shift_n):
        '''parse record file for quantization'''
        layers_params = collections.OrderedDict()
        skip_fusion_layers = []
        quantizable_layers = self.config[GRAPH_QUERIER].get_support_quant_layers(self.graph)
        if hasattr(self.config[GRAPH_QUERIER], 'get_support_qat_layer2type') and \
            not hasattr(self.graph, 'get_operations'):
            quantizable_layers += list(self.config[GRAPH_QUERIER].get_support_qat_layer2type(self.graph).keys())
        for record in self.records.record:
            # check key and value exist
            if not record.HasField('key'):
                raise RuntimeError('Cannot find "key" in record file.')
            if not record.HasField('value'):
                raise RuntimeError(
                    "cannot find 'value' in %s in record_file." % (record.key))
            node, is_quantize_tensor = self.find_layer_in_graph(record)

            recorder = RecordManager(record, node, self.capacity,
                                     self.config.get('op_quirer'), self.prune_idx)
            if recorder.get_skip_fusion():
                if node.type in self.capacity.get('FUSE_TYPES'):
                    skip_fusion_layers.append(record.key)

            if node.name in quantizable_layers:
                # parse value to np.array and store in a dictionary
                layer_params = recorder.parse_quant_value(enable_shift_n)
                if layer_params and node.type in RNN_LAYER_TYPE:
                    recorder.parse_quant_value_rnn(layer_params)

                if layer_params:
                    # check placeholder
                    if self.config.get(GRAPH_CHECKER) is not None:
                        self.config[
                            GRAPH_CHECKER].check_quantize_placeholder(
                                self.graph, [record.key])
                    layers_params[record.key] = layer_params
                tensor_balance_factor = recorder.parse_tensor_balance_factor_value()
                if tensor_balance_factor:
                    if not layers_params.get(record.key):
                        layers_params[record.key] = collections.OrderedDict()
                    layers_params[record.key]['tensor_balance_factor'] = tensor_balance_factor
            elif is_quantize_tensor:
                # check tensor valid
                if self.config.get(GRAPH_CHECKER) is not None:
                    layer_name, input_index = record.key.split(":")
                    self.config[GRAPH_CHECKER].check_tensor_quant(self.graph, [
                        {'layer_name': layer_name,
                        'input_index': int(input_index)}])
                layer_params = recorder.parse_quant_value(False, False)
                if layer_params:
                    layers_params[record.key] = layer_params
            else:
                raise RuntimeError("layer %s cannot be quantized "\
                    % (node.name))
        return RecordFileParsedResult(layers_params, skip_fusion_layers)

    def find_layer_in_graph(self, record):
        """ find the layer in graph """
        is_quantize_tensor = False
        try:
            if hasattr(self.graph, 'get_operation_by_name'):
                try:
                    node = self.graph.get_operation_by_name(record.key)
                except ValueError:
                    node = self.graph.get_operation_by_name(record.key.split(':')[0])
                    is_quantize_tensor = True
            else:
                key_name = record.key.split(':')
                node = self.graph.get_node_by_name(key_name[0])
                if len(key_name) > 1:
                    is_quantize_tensor = True
        except Exception as e:
            raise RuntimeError(
                "find %s failure in %s. Please match %s and %s." %
                (record.key, self.model_name, self.model_name,
                    self.record_file)) from e
        return node, is_quantize_tensor


class RecordManager():
    """
    Function: Parse a record
    APIS: get_key, get_value, get_do_fusion, parse_value
    """
    def __init__(self, record, node, capacity, op_quirer, prune_idx=None):
        ''' init object '''
        self.record = record
        self.node = node
        self.capacity = capacity
        self.op_quirer = op_quirer
        self.shift_n_types = capacity.get('SHIFT_N_TYPES')
        self.no_weight_quant_types = capacity.get('NO_WEIGHT_QUANT_TYPES')
        self.prune_idx = prune_idx
        self.no_dmq_balancer_types = ['AvgPool', 'AveragePool']
        if not self.shift_n_types:
            self.shift_n_types = []
        if not self.no_weight_quant_types:
            self.no_weight_quant_types = []

    @staticmethod
    def is_in_range(value_array, min_val, max_val, included=True):
        ''' whether value_array is in the range of [min_val, max_val]'''
        if included:
            is_in_range = (value_array >= min_val) & \
                            (value_array <= max_val)
        else:
            is_in_range = (value_array > min_val) & \
                            (value_array < max_val)
        return is_in_range.all()

    @staticmethod
    def get_range(dst_type):
        ''' Get the range based on dst_type(act_type or wts_type) of layers_params '''
        dst_type_dict = {INT4: DATA_OFFSET_RANGE,
                         'INT6': DATA_OFFSET_RANGE,
                         'INT7': DATA_OFFSET_RANGE,
                         'INT8': DATA_OFFSET_RANGE,
                         'INT16': DATA_OFFSET_RANGE_INT16
                        }
        return dst_type_dict.get(dst_type, DATA_OFFSET_RANGE)

    @staticmethod
    def check_rnn_layer_params_range(layer_params, layer_name):
        '''
        Check the range of value in layers_params
        from record_file for rnn operator exclusive params.
        '''
        # offset_h
        data_offset_range = RecordManager.get_range(layer_params['act_type'])
        if is_invalid(layer_params.get(H_OFFSET)) or \
            not RecordManager.is_in_range(layer_params.get(H_OFFSET),
                            data_offset_range[0],
                            data_offset_range[1]):
            raise ValueError("offset_h's value should be in %s in layer %s." %
                             (data_offset_range, layer_name))
        # offset_r
        if is_invalid(layer_params.get(RECURRENCE_WEIGHT_OFFSET)) or \
            not RecordManager.is_in_range(layer_params.get(RECURRENCE_WEIGHT_OFFSET),
                            WEIGHT_OFFSET_RANGE[0],
                            WEIGHT_OFFSET_RANGE[1]):
            raise ValueError("offset_r's value should be 0 in layer %s." %
                             (layer_name))
        # scale_d
        if is_invalid(layer_params.get('h_scale')) or \
            not RecordManager.is_in_range(layer_params.get('h_scale'),
                            SCALE_RANGE[0],
                            SCALE_RANGE[1],
                            included=False):
            raise ValueError("scale_h's value shoule be in positivate range "
                             "of float32 in layer %s." % (layer_name))
        # scale_r
        if is_invalid(layer_params.get('recurrence_weight_scale')) or \
            not RecordManager.is_in_range(layer_params.get('recurrence_weight_scale'),
                            SCALE_RANGE[0],
                            SCALE_RANGE[1],
                            included=False):
            raise ValueError("scale_r's value shoule be in positivate range "
                             "of float32 in layer %s" % (layer_name))

    @staticmethod
    def check_cluster_center(cluster_center_array):
        """ check whether the cluster center is in [-128, 127]"""
        if len(cluster_center_array) not in CLUSTER_CENTER_LEN_RANGE:
            raise ValueError(
                "The length of cluster center is 16 or 32, but is {}".format(len(cluster_center_array)))
        for value in cluster_center_array:
            if value < CLUSTER_CENTER_RANGE[0] or value > CLUSTER_CENTER_RANGE[1]:
                raise ValueError("Cluster center should between [-128, 127], but is {}".format(value))

    @classmethod
    def check_layer_params_range(cls, layer_params, layer_name):
        ''' Check the range of value in layers_params from record_file.'''
        # offset_d
        data_offset_range = RecordManager.get_range(layer_params['act_type'])
        if is_invalid(layer_params.get(DATA_OFFSET)) or \
            not RecordManager.is_in_range(layer_params.get(DATA_OFFSET),
                            data_offset_range[0],
                            data_offset_range[1]):
            raise ValueError("offset_d's value should be in %s in layer %s." %
                             (data_offset_range, layer_name))
        # offset_w
        if is_invalid(layer_params.get(WEIGHT_OFFSET)) or \
            not RecordManager.is_in_range(layer_params.get(WEIGHT_OFFSET),
                            WEIGHT_OFFSET_RANGE[0],
                            WEIGHT_OFFSET_RANGE[1]):
            raise ValueError("offset_w's value should be 0 in layer %s." %
                             (layer_name))
        # scale_d
        if is_invalid(layer_params.get(DATA_SCALE)) or \
            not RecordManager.is_in_range(layer_params.get(DATA_SCALE),
                            SCALE_RANGE[0],
                            SCALE_RANGE[1],
                            included=False):
            raise ValueError("scale_d's value shoule be in positivate range "
                             "of float32 in layer %s." % (layer_name))
        # scale_w
        if is_invalid(layer_params.get(WEIGHT_SCALE)) or \
            not RecordManager.is_in_range(layer_params.get(WEIGHT_SCALE),
                            SCALE_RANGE[0],
                            SCALE_RANGE[1],
                            included=False):
            raise ValueError("scale_w's value shoule be in positivate range "
                             "of float32 in layer %s" % (layer_name))

        # shift_bit
        def _is_shift_n_in_range(shift_bit):
            if RecordManager.is_in_range(shift_bit, 0, 0) or RecordManager.is_in_range(
                    shift_bit, SHIFT_N_RANGE[0], SHIFT_N_RANGE[1]):
                return True
            return False

        if is_invalid(layer_params.get(SHIFT_N)) or \
            not _is_shift_n_in_range(layer_params.get(SHIFT_N)):
            raise ValueError("shift_bit's value shoule be in %s or all be 0 "
                             "in layer %s" % (SHIFT_N_RANGE, layer_name))

    def get_key(self):
        ''' get key '''
        return self.record.key

    def get_value(self):
        ''' get value '''
        return self.record.value

    def get_skip_fusion(self):
        ''' get skip_fusion '''
        return self.get_value().skip_fusion

    def get_pruned_quant_factor(self, scale_w_array, offset_w_array, shift_n_array, enable_shift_n):
        """
        Function:Cut the quant factors with regards to the pruned channel
        """
        prune_channel = self.prune_idx.get(self.node.name, None)
        factor_length = scale_w_array.size
        channel_wise = (factor_length > 1)
        if prune_channel and channel_wise:
            mask_length = len(prune_channel)
            if factor_length != mask_length:
                raise RuntimeError('The quant factor length {} does not match ' \
                    'the prune mask length {}'.format(factor_length, mask_length))

            prune_channel = [(i == 1) for i in prune_channel]
            scale_w_array = scale_w_array[prune_channel]
            offset_w_array = offset_w_array[prune_channel]
            if enable_shift_n:
                shift_n_array = shift_n_array[prune_channel]

        return scale_w_array, offset_w_array, shift_n_array

    def parse_quant_value_rnn(self, layer_params):
        ''' Parse quant value(scale,offset,shift_bit,dst_type) from record '''
        # read scale_d and offset_d
        layer_params['h_scale'] = self.get_act_quant_factor('scale_h', np.float32)
        layer_params[H_OFFSET] = self.get_act_quant_factor('offset_h', np.float32)

        # read scale_w, offset_w and shift_bit
        scale_r_array = self.get_wts_quant_factor('scale_r', np.float32)
        offset_r_array = self.get_wts_quant_factor('offset_r', np.int32)

        scale_r_array, offset_r_array, _ = \
            self.adjust_scale_offset_shiftn(scale_r_array, offset_r_array,
                                            np.array([]))
        layer_params['recurrence_weight_scale'] = scale_r_array
        layer_params[RECURRENCE_WEIGHT_OFFSET] = offset_r_array
        RecordManager.check_rnn_layer_params_range(layer_params, self.get_key())

        # astype to necessary type
        if layer_params['act_type'] == 'INT16':
            layer_params[H_OFFSET] = \
                layer_params[H_OFFSET].astype(np.int16)
        else:
            layer_params[H_OFFSET] = \
                layer_params[H_OFFSET].astype(np.int8)
        layer_params[RECURRENCE_WEIGHT_OFFSET] = \
            layer_params[RECURRENCE_WEIGHT_OFFSET].astype(np.int8)

        return layer_params

    def parse_quant_value(self, enable_shift_n=True, enable_wts_quant=True):
        ''' Parse quant value(scale,offset,shift_bit,dst_type) from record '''
        if self.check_quant_value_empty():
            return None
        # read value(scale,offset,shift_bit) from record
        layer_params = collections.OrderedDict()
        # read scale_d and offset_d
        layer_params[DATA_SCALE] = self.get_act_quant_factor('scale_d', np.float32)
        layer_params[DATA_OFFSET] = self.get_act_quant_factor('offset_d', np.int32)
        layer_params['dst_type'] = self.get_dst_type()
        layer_params['act_type'] = self.get_act_type()
        layer_params['wts_type'] = self.get_wts_type()
        layer_params['fakequant_precision_mode'] = self.get_fakequant_precision_mode()
        # read op_dtype
        self.get_op_dtype(layer_params)
        # read scale_w, offset_w and shift_bit
        scale_w_array = self.get_wts_quant_factor('scale_w', np.float32,
            desirable=self.node.type not in self.no_weight_quant_types and enable_wts_quant)
        offset_w_array = self.get_wts_quant_factor('offset_w', np.int32,
            desirable=self.node.type not in self.no_weight_quant_types and enable_wts_quant)
        shift_n_array = self.get_shift_n(desirable=(
            enable_shift_n and self.node.type in self.shift_n_types))

        if self.prune_idx is not None:
            scale_w_array, offset_w_array, shift_n_array = self.get_pruned_quant_factor( \
                scale_w_array, offset_w_array, shift_n_array, enable_shift_n)

        # optional cluster center
        cluster_center_array = self.get_cluster_center()

        if enable_shift_n or enable_wts_quant:
            scale_w_array, offset_w_array, shift_n_array = \
                self.adjust_scale_offset_shiftn(scale_w_array, offset_w_array,
                                                shift_n_array)
        layer_params[WEIGHT_SCALE] = scale_w_array
        layer_params[WEIGHT_OFFSET] = offset_w_array
        layer_params[SHIFT_N] = shift_n_array

        # optional cluster center
        if cluster_center_array is not None:
            RecordManager.check_cluster_center(cluster_center_array)
            layer_params[CLUSTER_CENTER] = cluster_center_array

        # check value
        RecordManager.check_layer_params_range(layer_params, self.get_key())

        # astype to necessary type
        if layer_params['act_type'] == 'INT16':
            layer_params[DATA_OFFSET] = layer_params[DATA_OFFSET].astype(np.int16)
        else:
            layer_params[DATA_OFFSET] = layer_params[DATA_OFFSET].astype(np.int8)
        layer_params[WEIGHT_OFFSET] = layer_params[WEIGHT_OFFSET].astype(np.int8)
        layer_params[SHIFT_N] = layer_params.get(SHIFT_N).astype(np.int8)

        # optional cluster center
        if cluster_center_array is not None:
            layer_params[CLUSTER_CENTER] = layer_params.get(CLUSTER_CENTER).astype(np.int8)

        return layer_params

    def parse_tensor_balance_factor_value(self):
        ''' Parse tensor_balance_factor value from record '''
        # read tensor_balance_factor
        tensor_balance_factor_array = self.get_tensor_balance_factor(desirable=(
            self.node.type not in self.no_dmq_balancer_types))
        if tensor_balance_factor_array is None:
            return None
        # check tensor_balance_factor length
        tensor_balance_factor_length_expect = self.get_possible_input_length()
        if tensor_balance_factor_array.size != tensor_balance_factor_length_expect:
            raise ValueError("tensor_balance_factor's length: {} should be equal to"\
                "the channel length of inputs of layer {}: {}."\
                "Cannot support to quantize {}".format(tensor_balance_factor_array.size, self.node.name,
                tensor_balance_factor_length_expect, self.node.name))
        # check tensor_balance_factor value
        if not (tensor_balance_factor_array >= TENSOR_BALANCE_FACTOR_RANGE[0]).all() or \
            not (tensor_balance_factor_array <= TENSOR_BALANCE_FACTOR_RANGE[1]).all():
            raise ValueError("tensor_balance_factor's value shoule be in range[FLT_EPSILON, 1/FLT_EPSILON] "
                             "of float32 in layer {}".format(self.get_key()))

        return tensor_balance_factor_array.tolist()

    def check_quant_value_empty(self):
        '''check whether the value in record is empty '''
        value = self.get_value()
        if value.scale_d or value.offset_d or value.scale_w or \
            value.offset_w or value.shift_bit:
            return False
        if hasattr(value, 'scale_h') and \
            (value.scale_h or value.offset_h or value.scale_r or value.offset_r):
            return False
        return True

    def get_dst_type(self):
        '''get dst type'''
        if hasattr(self.record.value, 'dst_type') and self.record.value.HasField('dst_type'):
            if 'AvgPool' in self.record.key and self.record.value.dst_type == INT4:
                raise RuntimeError('AvgPool in INT4 not supported. Please Check record file.')
            if self.record.value.dst_type in ('INT8', INT4):
                return self.record.value.dst_type
            raise RuntimeError('Do not support dst_type {} in record file'.format( \
                                self.record.value.dst_type))
        return 'UNSET'

    def get_act_type(self):
        '''get act type'''
        if hasattr(self.record.value, 'act_type') and self.record.value.HasField('act_type'):
            support_act_type = ['INT' + str(num_bits) for num_bits in ACT_SUPPORT_NUM_BITS]
            if self.record.value.act_type in support_act_type:
                return self.record.value.act_type
            raise RuntimeError('Do not support act_type {} in record file'.format( \
                               self.record.value.act_type))
        LOGGER.logw('act_type does not exist in the record file ')
        return 'UNSET'

    def get_wts_type(self):
        '''get wts type'''
        if hasattr(self.record.value, 'wts_type') and self.record.value.HasField('wts_type'):
            if self.record.value.wts_type in (INT4, 'INT6', 'INT7', 'INT8'):
                return self.record.value.wts_type
            raise RuntimeError('Do not support wts_type {} in record file'.format( \
                               self.record.value.wts_type))
        LOGGER.logw('wts_type does not exist in the record file ')
        return 'UNSET'

    def get_op_dtype(self, layer_params):
        '''get op dtype'''
        if hasattr(self.record.value, OP_DATA_TYPE) and self.record.value.HasField(OP_DATA_TYPE):
            if self.record.value.op_data_type not in ['FLOAT32', 'FLOAT16']:
                raise RuntimeError(f'Do not support op_data_type {self.record.value.op_data_type}')
            layer_params[OP_DATA_TYPE] = self.record.value.op_data_type

    def get_act_quant_factor(self, name, dtype):
        '''get activation quant factor on name'''
        if not self.record.value.HasField(name):
            raise RuntimeError(
                "cannot find {} of layer {} in record_file, "\
                "please check data calibration process!".format(name, self.get_key()))
        result = list()
        result.append(getattr(self.record.value, name))
        result_array = np.array(result, dtype=dtype)[0]
        return result_array

    def get_wts_quant_factor(self, name, dtype, desirable=True):
        result = []
        if desirable:
            if not getattr(self.record.value, name, None):
                raise RuntimeError(
                    "cannot find {} of layer {} in record_file".format(name, self.get_key()))
            result.extend(getattr(self.record.value, name))
        else:
            if getattr(self.record.value, name, None):
                raise RuntimeError(
                    "find undesirable {} of layer {} in record_file".format(name, self.get_key()))
            default_value = 1.0 if 'scale' in name else 0.0
            result.extend([default_value])

        result_array = np.array(result, dtype=dtype)

        return result_array

    def get_cluster_center(self):
        ''' optional value, get cluster_center '''
        record = self.record
        cluster_center = []
        if not hasattr(record.value, 'cluster'):
            cluster_center_array = None
        elif not record.value.cluster:
            cluster_center_array = None
        else:
            cluster_center.extend(record.value.cluster)
            cluster_center_array = np.array(cluster_center, dtype=np.int32)

        return cluster_center_array

    def get_shift_n(self, desirable=True):
        ''' get shift_n '''
        shift_bit = []
        if desirable:
            if not self.record.value.shift_bit:
                raise RuntimeError(
                    "cannot find shift_bit of layer %s in record_file" %
                    (self.get_key()))
            shift_bit.extend(self.record.value.shift_bit)
        else:
            if self.record.value.shift_bit:
                raise RuntimeError(
                    "shift_bit is not support, please delete shift_bit "
                    "of layer %s in record_file" % (self.get_key()))
            shift_bit = []

        shift_n_array = np.array(shift_bit, dtype=np.uint32)

        return shift_n_array


    def get_tensor_balance_factor(self, desirable=True):
        ''' get tensor_balance_factor '''
        if not hasattr(self.record.value, 'tensor_balance_factor') or not self.record.value.tensor_balance_factor:
            return None
        if not desirable:
            raise RuntimeError(
                "tensor_balance_factor is not support, please delete tensor_balance_factor "
                "of layer {} in record_file".format(self.get_key()))
        return np.array(self.record.value.tensor_balance_factor, dtype=np.float32)


    def get_fakequant_precision_mode(self):
        '''get fakequant precision mode'''
        if hasattr(self.record.value, 'fakequant_precision_mode') and \
            self.record.value.HasField('fakequant_precision_mode'):
            if self.record.value.fakequant_precision_mode in ['DEFAULT', 'FORCE_FP16_QUANT']:
                return self.record.value.fakequant_precision_mode
            raise RuntimeError('Do not support fakequant_precision_mode {} in record file'.format( \
                               self.record.value.fakequant_precision_mode))
        LOGGER.logd('fakequant_precision_mode does not exist in the record file ')
        return 'DEFAULT'


    def adjust_scale_offset_shiftn(self, scale_w_array, offset_w_array,
                                   shift_n_array):
        '''
        Function: Adjust the shape of scale and offset according to operation.
        if operation is conv2d and scale_w_array has multiple numbers, then
        scale_w_array's shape will be reshaped as [cout, 1, 1, 1]
        Inputs:
        scale_w_array: np.array, scale_w
        offset_w_array: np.array, offset_w
        shift_n_array: np.array, shift_n
        Return:
        scale_w_array: np.array, reshaped scale_w
        offset_w_array: np.array, reshaped offset_w
        shift_n_array: np.array, reshaped shift_n
        '''
        operation = self.node
        err_prefix = "scale_w's length, offset_w's length"
        if not shift_n_array.size:
            shift_n_array = np.array([0] * scale_w_array.size, dtype=np.int8)
        else:
            err_prefix += " and shift_bit's length"
        if scale_w_array.size != offset_w_array.size or \
            scale_w_array.size != shift_n_array.size:
            raise RuntimeError("{} should be equal in layer {}.".format(err_prefix, operation.name))

        if scale_w_array.size == 1:
            channel_wise = False
        else:
            if operation.type in RNN_LAYER_TYPE and \
                scale_w_array.size == RNN_TENSOR_NUM.get(operation.type):
                channel_wise = False
            else:
                channel_wise = True

        # check length and reshape
        shape_expect, scale_length_expect = self.get_possible_length(
            channel_wise)
        if scale_w_array.size != scale_length_expect:
            raise RuntimeError("{}({}) should be equal to the {} when "\
                "channel_wise be {}. Cannot support to quantize {}".format(
                err_prefix, scale_w_array.size, scale_length_expect, channel_wise, operation.name))

        scale_w_array = np.reshape(scale_w_array, shape_expect)
        offset_w_array = np.reshape(offset_w_array, shape_expect)
        shift_n_array = np.reshape(shift_n_array, shape_expect)

        return scale_w_array, offset_w_array, shift_n_array

    def get_possible_length(self, channel_wise=True):
        '''Get possible length for   '''
        operation = self.node
        if operation.type in ('AvgPool', 'AveragePool'):
            shape_expect = []
            scale_length_expect = 1
        else:
            if hasattr(self.op_quirer, 'get_op_scale_shape'):
                quire_func = getattr(self.op_quirer, 'get_op_scale_shape')
            else:
                quire_func = getattr(self.op_quirer, 'get_scale_shape')
            shape_expect, scale_length_expect = quire_func(
                self.node, channel_wise)
        return shape_expect, scale_length_expect

    def get_possible_input_length(self):
        '''Get possible input cin length for   '''
        quire_func = getattr(self.op_quirer, 'get_cin_length')
        tensor_balance_factor_length_expect = quire_func(self.node)
        return tensor_balance_factor_length_expect
