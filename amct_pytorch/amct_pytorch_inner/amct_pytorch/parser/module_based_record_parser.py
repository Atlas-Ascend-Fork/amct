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
from collections import OrderedDict

import numpy as np
import torch

from amct_pytorch.common.utils.util import version_higher_than
from amct_pytorch.proto import post_quant_record_pb2
from amct_pytorch.common.utils.parse_record_file import RecordFileParserBase
from amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.utils.vars import POST_QUANTIZE_ACT_TYPES, POST_QUANTIZE_WTS_TYPES, POST_QUANTIZE_WTS_ONLY_TYPES
from amct_pytorch.utils.vars import ALLOWED_ROUND_MODE_MAP, QUANT_COMBINATION_MAP
from amct_pytorch.utils.vars import MXFP4_E2M1, MXFP8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2, INT4, INT8

__all__ = ["ModuleBasedRecordParser"]


class ModuleBasedRecordParser():
    """
    Function: Parse the information of post quant from record_file.
    APIs: is_records_empty, parse
    """
    def __init__(self, record_file, module):
        self.record_file = os.path.realpath(record_file)
        self.module = module
        self.records = post_quant_record_pb2.ScaleOffsetRecord()
        RecordFileParserBase.read_record_file(self.record_file, self.records)
        if self.is_records_empty():
            raise RuntimeError(
                "record_file is empty, no layers to be saved. "
                "please ensure calibration is finished by checking information!")

    def is_records_empty(self):
        ''' check records  '''
        return list(self.records.record) == []

    def parse(self):
        """ parse the information of post quant """
        layers_params = OrderedDict()

        for record in self.records.record:
            if not record.HasField('key'):
                raise RuntimeError('Cannot find "key" in record file.')
            if not record.HasField('value'):
                raise RuntimeError(
                    "Cannot find 'value' in {} in record_file.".format(record.key))

            record_manager = ModuleBasedRecordManager(record, self.module)
            layer_params = record_manager.parse_quant_value()
            layers_params[record.key] = layer_params
        return layers_params


class ModuleBasedRecordManager():
    """
    Function: Parse a record
    APIS: get_key, get_value, parse_quant_value
    """
    def __init__(self, record, model):
        self.record = record
        self.model = model
    
    def get_key(self):
        ''' get key '''
        return self.record.key

    def parse_quant_value(self):
        ''' Parse quant factor from record '''
        layer_params = OrderedDict()
        layer_params['round_mode'] = self.get_round_mode()
        layer_params['wts_type'] = self.get_wts_type()
        layer_params['weight_compress_only'] = self.get_weight_compress_only()
        if not layer_params.get('weight_compress_only'):
            layer_params['act_type'] = self.get_act_type()
        if self.record.value.HasField('group_size') and hasattr(self.record.value, 'group_size'):
            layer_params['group_size'] = self.get_group_size()
        
        #parse with quant_result.pt
        if self.record.value.HasField('quant_result_path') and hasattr(self.record.value, 'quant_result_path'):
            layer_params['quant_result_path'] = self.record.value.quant_result_path
            self.check_quant_params_range(layer_params, self.get_key())
            return layer_params

        #parse without quant_result.pt
        if self.is_weight_scale_exist():
            layer_params['weight_scale'] = self.get_quant_factor('scale_w')

        if not layer_params['weight_compress_only']:
            # MXFP8_E4M3FN*MXFP8_E4M3FN/MXFP4_E2M1
            if layer_params['act_type'] != MXFP8_E4M3FN:
                layer_params['data_scale'] = self.get_quant_factor('scale_d')
                data_offset = self.get_quant_factor('offset_d', np.int32, check_existed=False)
                if data_offset is not None:
                    layer_params['data_offset'] = data_offset
        return layer_params

    def get_quant_factor(self, name, dtype=np.float32, check_existed=True):
        ''' get quant factor from record proto by name '''
        if not hasattr(self.record.value, name):
            raise RuntimeError(
                "cannot find {} of layer {} in record_file, "\
                "please check data calibration process!".format(name, self.get_key()))
        result = getattr(self.record.value, name)
        if len(result) == 0:
            if check_existed:
                raise RuntimeError("cannot find {} of layer {} in record_file, "
                "please check data calibration process!".format(name, self.get_key()))
            else:
                return None
        result_array = np.array(result, dtype=dtype)
        return result_array

    def get_weight_compress_only(self):
        ''' get weight compress only '''
        if not self.record.value.HasField('weight_compress_only') or \
            not hasattr(self.record.value, 'weight_compress_only'):
            return False
        return self.record.value.weight_compress_only

    def get_act_type(self):
        '''get act type'''
        if not self.record.value.HasField('act_type') or \
            not hasattr(self.record.value, 'act_type'):
            raise RuntimeError("act_type is not set for layer {}".format(self.get_key()))
        if self.record.value.act_type not in POST_QUANTIZE_ACT_TYPES:
            raise RuntimeError('Do not support act_type {} in record file'.format( \
                               self.record.value.act_type))
        return self.record.value.act_type

    def get_wts_type(self):
        '''get wts type'''
        if not self.record.value.HasField('wts_type') or \
            not hasattr(self.record.value, 'wts_type'):
            raise RuntimeError("wts_type is not set for layer {}".format(self.get_key()))
        if self.record.value.wts_type not in (POST_QUANTIZE_WTS_TYPES + POST_QUANTIZE_WTS_ONLY_TYPES):
            raise RuntimeError('Do not support wts_type {} in record file'.format( \
                               self.record.value.wts_type))
        return self.record.value.wts_type

    def get_round_mode(self):
        ''' get round mode '''
        if not self.record.value.HasField('round_mode') or \
            not hasattr(self.record.value, 'round_mode'):
            raise RuntimeError("round_mode is not set for layer {}".format(self.get_key()))
        return self.record.value.round_mode

    def get_weight_compress_only(self):
        ''' get weight compress only '''
        if not self.record.value.HasField('weight_compress_only') or \
            not hasattr(self.record.value, 'weight_compress_only'):
            return False
        return self.record.value.weight_compress_only
    
    def get_group_size(self):
        ''' get group size '''
        if not self.record.value.HasField('group_size') or \
            not hasattr(self.record.value, 'group_size'):
            return None
        return self.record.value.group_size

    def check_quant_params_range(self, layer_params, layer_name):
        ''' Check the range of value in layers_params '''
        mod = ModuleHelper(self.model).get_module(layer_name)
        layer_type = type(mod).__name__
        wts_type = layer_params.get('wts_type')
        if not layer_params.get('weight_compress_only'):
            act_type = layer_params.get('act_type')
        else:
            act_type = 'NOT_QUANTIZE'
        quant_combination = act_type + ' ' + layer_params.get('wts_type')
        if quant_combination not in QUANT_COMBINATION_MAP:
            raise RuntimeError(
                'Not supported quant type act_type: {}, wts_type: {}'.format(act_type, wts_type))
        allowed_layer_type = QUANT_COMBINATION_MAP.get(quant_combination)
        if layer_type not in allowed_layer_type:
            raise RuntimeError('layer {} not support act_type {} and wts {}'.format(
                layer_name, act_type, wts_type))

        # check round mode for each dtype
        if layer_params.get('round_mode') and \
            layer_params.get('round_mode') not in ALLOWED_ROUND_MODE_MAP.get(layer_params.get('wts_type')):
            raise RuntimeError("{} only support round mode {}".format(layer_params.get(
                'act_type'), ALLOWED_ROUND_MODE_MAP.get(layer_params.get('wts_type'))))

    def is_weight_scale_exist(self):
        '''check weight scale is exist in the algo'''
        if self.get_wts_type() in [MXFP4_E2M1, MXFP8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2]:
            # weight_scale is not exist MXFP4 and MXFP8 
            return False
        return True


def get_layer_quant_params(records, layer_name):
    """
    Function: get single layer quant params from quant_result_path in records
    Inputs:
    records: record
    layer_name: str, name of check layer
    return: dict, include current layer params
    """
    if records.get('quant_result_path') is None:
        raise RuntimeError("quant_result_path not exists in record!".format(layer_name))

    quant_result_path = records.get('quant_result_path')
    if not os.path.exists(quant_result_path):
        raise RuntimeError("quant_result_path {} not exists. Please check your record file.".format(quant_result_path))
    if version_higher_than(torch.__version__, '2.1.0'):
        load_kwargs = {'mmap': True, 'weights_only': False}
    else:
        load_kwargs = {}
    try:
        quant_params = torch.load(quant_result_path, **load_kwargs)
    except Exception as e:
        raise RuntimeError("obtain quant_params params from file failed!") from e
    if quant_params.get(layer_name) is None:
        raise RuntimeError("obtain quant_params param for layer {} failed!".format(layer_name))
    return quant_params.get(layer_name)
