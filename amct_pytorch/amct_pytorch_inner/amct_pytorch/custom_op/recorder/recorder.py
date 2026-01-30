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

Optimizer is manager and executor of graph passes.

"""

import torch
from google.protobuf import text_format
from torch import nn
from pathlib import Path

from amct_pytorch.common.utils import files as files_util
from amct_pytorch.proto import post_quant_record_pb2
from amct_pytorch.common.utils.record_file_operator import \
    record_activation_scale_offset
from amct_pytorch.common.utils.record_file_operator import \
    record_weights_scale_offset
from amct_pytorch.common.utils.record_file_operator import \
    record_shift_bits
from amct_pytorch.common.utils.record_file_operator import \
    read_weights_scale_offset
from amct_pytorch.common.utils.record_file_operator import \
    read_activation_scale_offset
from amct_pytorch.common.utils.record_file_operator import \
    record_dmq_balancer_factor
from amct_pytorch.common.utils.record_file_operator import \
    record_kv_cache_scale_offset
from amct_pytorch.common.utils.record_file_operator import \
    record_quant_factors
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.custom_op.utils import tensor
from amct_pytorch.utils.singleton_record import SingletonScaleOffsetRecord
from amct_pytorch.common.utils.vars_util import DEFAULT
from amct_pytorch.common.utils.util import cast_fp16_precision
from amct_pytorch.utils.vars import QUANT_RESULT_FILE

LAYERS_NUM = 'layers_num'


class Recorder(nn.Module): # pylint: disable=R0902
    """
    Record quant factors to record file.
    """
    def __init__(self, record_file, enable_dmq_balancer=False,
        enable_kv_cache_quant=False, fakequant_precision_mode=DEFAULT):
        """
        Function: init function
        Inputs:
            record_file: a string, the file to record quant factors
        Returns: None
        """
        super().__init__()
        self.record_file = record_file
        self.enable_dmq_balancer = enable_dmq_balancer
        self.enable_kv_cache_quant = enable_kv_cache_quant
        self.register_buffer('quant_layer_num', tensor(0))
        self.quant_layer_names = []
        self.fakequant_precision_mode = fakequant_precision_mode

        self.__read_over = False
        self.__write_over = False
        self.__counts = {
            'layers_num': 0,
            'act_cali_num': 0,
            'wts_cali_num': 0,
            'dmq_balancer_num': 0,
            'kv_cache_quant_num': 0,
        }
        self.__add_record_factors_map = {
            'ifmr': self._add_acts_factors,
            'hfmg': self._add_acts_factors,
            'arq': self._add_wts_factors,
            'dmq_balancer': self._add_dmq_balancer_factors,
            'kv_cache': self._add_kv_cache_factors,
        }

        SingletonScaleOffsetRecord().reset_record()
        self.records = SingletonScaleOffsetRecord().record

    def record_quant_layer(self, layer_name):
        """
        Function: record the number of quantization layers.
        Inputs:
            layer_name: a string, layer name to be quantified
        Returns: None
        """
        if not set(layer_name) < set(self.quant_layer_names):
            self.quant_layer_names.extend(layer_name)
            count_num = len(list(set(self.quant_layer_names)))
            if self.quant_layer_num != count_num: # pylint: disable=E0203
                self.quant_layer_num.mul_(0).add_(1).mul_(count_num)
        self.__counts['act_cali_num'] = 0
        self.__counts['wts_cali_num'] = 0
        self.__counts['dmq_balancer_num'] = 0
        self.__counts['kv_cache_quant_num'] = 0

    def forward(self,
                layers_name,
                factors_type,
                quant_factors):
        """
        Function: forward function
        Inputs:
            layers_name: a string, Name of the layer to which the quant factor belongs
            factors_type: a string, quant factors' type
            quant_factors: a dict, include
                scale_d: float number, data's scale
                offset_d: int number, data's offset
        Returns: None
        """
        if factors_type not in self.__add_record_factors_map.keys():
            raise ValueError("unsupport factors_type %s!" % (factors_type))

        # enable lock
        SingletonScaleOffsetRecord().lock.acquire()
        self._read_record_file()
        self._add_record_factors(layers_name, factors_type, quant_factors)
        self._write_record_file()
        # release lock
        SingletonScaleOffsetRecord().lock.release()

    def get_scales(self, layer_name):
        """
        Function: get scale_w, scale_d of layer_name from record
        Inputs:
            layer_name: string, name of layer
        Returns:
            scale_w: a list, scale_w of layer_name
            scale_d: a number, scale_d of layer_name
        """
        SingletonScaleOffsetRecord().lock.acquire()
        scale_w, _ = read_weights_scale_offset(self.records, layer_name)
        scale_d, _ = read_activation_scale_offset(self.records, layer_name)
        SingletonScaleOffsetRecord().lock.release()
        return scale_w, scale_d

    def backward(self):
        ''' backward function '''
        raise NotImplementedError

    def _read_record_file(self):
        """ read record from file"""
        if not self.__read_over:
            with open(self.record_file, 'r') as fid:
                pbtxt_string = fid.read()
                text_format.Merge(pbtxt_string, self.records)
            if self.quant_layer_num.cpu():
                self.__counts[LAYERS_NUM] = self.quant_layer_num.cpu()
            else:
                self.__counts[LAYERS_NUM] = len(self.records.record)
            self.__read_over = True

    def check_layer_recorded(self, layer_name, record_keyword):
        """ check whether the layer has already be written in record """
        for item in self.records.record:
            if item.key == layer_name and hasattr(item, self.record_keyword):
                return True
        return False

    def _write_record_file(self):
        """ write record to file"""
        if self.__counts.get('act_cali_num') == self.__counts.get(LAYERS_NUM):
            with open(self.record_file, "w") as fid:
                fid.write(
                    text_format.MessageToString(self.records,
                                                as_utf8=True))
            if not self.quant_layer_num.cpu():
                self.__write_over = True
        elif self.enable_dmq_balancer:
            if self.__counts.get('dmq_balancer_num') == self.__counts.get(LAYERS_NUM):
                with open(self.record_file, "w") as fid:
                    fid.write(
                        text_format.MessageToString(self.records,
                                                    as_utf8=True))
                if not self.quant_layer_num.cpu():
                    self.__write_over = True
        elif self.enable_kv_cache_quant:
            if self.__counts.get('kv_cache_quant_num') == self.__counts.get('layers_num'):
                with open(self.record_file, "w") as fid:
                    fid.write(
                        text_format.MessageToString(self.records,
                                                    as_utf8=True))
                if not self.quant_layer_num.cpu():
                    self.__write_over = True

    def _add_record_factors(self, layers_name, factors_type, quant_factors):
        """ add quant factors to record"""
        if self.__write_over:
            raise RuntimeError("The records has been written to record_file!")

        self.__add_record_factors_map.get(factors_type)(layers_name, quant_factors)

    def _add_acts_factors(self, layers_name, quant_factors):
        """ add acts quant factors to record"""
        if self.__counts.get('act_cali_num') >= self.__counts.get(LAYERS_NUM):
            raise RuntimeError(
                "number of data's factors(scale_d, offset_d) has been "
                "samed with number of quantization layers")
        scale_d = quant_factors.get('scale_d')
        scale_h = quant_factors.get('scale_h')
        offset_d = quant_factors.get('offset_d')
        offset_h = quant_factors.get('offset_h')
        num_bits = quant_factors.get('num_bits')
        if scale_d is None or offset_d is None:
            raise ValueError("scale_d and offset_d are necessary")
        for layer_name in layers_name:
            if self.fakequant_precision_mode != DEFAULT:
                scale_d = cast_fp16_precision(scale_d)
                if scale_h is not None:
                    scale_h = cast_fp16_precision(scale_h)
            record_activation_scale_offset(self.records, layer_name,
                                            scale_d, offset_d, num_bits, scale_h, offset_h)
            self.__counts['act_cali_num'] += 1
        LOGGER.logd(
            "Record layer '{}' data's quant factors!".format(layers_name),
            'Recorder')

    def _add_wts_factors(self, layers_name, quant_factors):
        """ add wts quant factors to record"""
        if self.__counts.get('wts_cali_num') >= self.__counts.get(LAYERS_NUM):
            raise RuntimeError(
                "number of data's factors(scale_w, offset_w) has been "
                "samed with number of quantization layers")
        scale_w = quant_factors.get('scale_w')
        scale_r = quant_factors.get('scale_r')
        offset_w = quant_factors.get('offset_w')
        offset_r = quant_factors.get('offset_r')
        num_bits = quant_factors.get('num_bits')
        if scale_w is None or offset_w is None:
            raise ValueError("scale_w and offset_w are necessary")
        for layer_name in layers_name:
            record_weights_scale_offset(self.records, layer_name,
                                        scale_w, offset_w, num_bits, scale_r, offset_r)
            self.__counts['wts_cali_num'] += 1
        LOGGER.logd("Record layer '{}' weight's quant "
                    "factors!".format(layers_name), 'Recorder')

    def _add_dmq_balancer_factors(self, layers_name, quant_factors):
        """ add dmq_balancer factors to record"""
        tensor_balance_factor = quant_factors.get('tensor_balance_factor')
        if tensor_balance_factor is None:
            raise ValueError("tensor_balance_factor is necessary")
        for layer_name in layers_name:
            record_dmq_balancer_factor(self.records, layer_name, tensor_balance_factor)
            self.__counts['dmq_balancer_num'] += 1
        LOGGER.logd(
            "Record layer '{}' tensor_balance_factor!".format(layers_name),
            'Recorder')

    def _add_kv_cache_factors(self, layers_name, quant_factors):
        """ add kv_cache quant factors to record """
        if self.__counts.get('kv_cache_quant_num') >= self.__counts.get('layers_num'):
            raise RuntimeError(
                "number of data's factors(scale, offset) has been "
                "greater than or equal to number of quantization layers")
        scale = quant_factors.get('scale')
        offset = quant_factors.get('offset')
        if scale is None or offset is None:
            raise ValueError("scale and offset are necessary")
        for layer_name in layers_name:
            record_kv_cache_scale_offset(self.records, layer_name, scale, offset)
            self.__counts['kv_cache_quant_num'] += 1
        LOGGER.logd(
            "Record layer '{}' data's quant factors!".format(layers_name),
            'Recorder')


class PostQuantRecorder(nn.Module):
    """
    Record quant factors to record file.
    """
    def __init__(self, record_file):
        """
        Function: init function
        Inputs:
            record_file: a string, the file to record quant factors
        Returns: None
        """
        super().__init__()
        self.record_file = record_file
        self.__read_over = False
        self.records = post_quant_record_pb2.ScaleOffsetRecord()
        self._read_record_file()
        self.quant_result_file = str(Path(record_file).with_name(QUANT_RESULT_FILE))
        self.quant_result_layer_names = list()
        self.quant_result = dict()

    def forward(self,
                layers_name,
                quant_info,
                quant_factors=None):
        """
        Function: forward function
        Inputs:
            layers_name: a string, the file to record quant factors
            quant_factors: a dict, include
                scale_d: float number, data's scale
                scale_w: float number, wts's scale
        Returns: None
        """
        self._add_record_factors(layers_name, quant_info, quant_factors)
        self._write_quant_result()
        self._write_record_file()

    def record_quant_result_layer(self, layer_name):
        """
        Function: record the number of quant result layers.
        Inputs:
            layer_name: list of string.
        Returns: None
        """
        if not set(layer_name) < set(self.quant_result_layer_names):
            self.quant_result_layer_names.extend(layer_name)

    def _read_record_file(self):
        """ read record from file"""
        if not self.__read_over:
            with open(self.record_file, 'r') as fid:
                pbtxt_string = fid.read()
                text_format.Merge(pbtxt_string, self.records)
            self.__read_over = True

    def _write_record_file(self):
        """ write record to file"""
        with open(self.record_file, "w") as fid:
            fid.write(
                text_format.MessageToString(self.records,
                                            as_utf8=True))

    def _write_quant_result(self):
        if len(self.quant_result.keys()) == 0 or len(self.quant_result.keys()) != len(self.quant_result_layer_names):
            return
        self.quant_result_file = files_util.create_empty_file(self.quant_result_file, check_exist=True)
        torch.save(self.quant_result, self.quant_result_file)
        self.quant_result = dict()

    def _add_record_factors(self, layers_name, quant_info, quant_params):
        """ add quant factors to record """
        if quant_params:
            quant_info['quant_result_path'] = self.quant_result_file
            self.quant_result.update(quant_params)

        for layer_name in layers_name:
            record_quant_factors(self.records, layer_name, quant_info)
        LOGGER.logd(
            "Record layer '{}' data's quant factors!".format(layers_name),
            'Recorder')

    
