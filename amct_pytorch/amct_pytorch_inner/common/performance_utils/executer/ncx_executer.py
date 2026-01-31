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

import ctypes
import os
import numpy as np

from ....utils.log import LOGGER  # pylint: disable=E0402
from ....lib.load_custom_op import NCX_LIB_PATH  # pylint: disable=E0402

ASCEND_SAMPLER = 'ascend_sampler'
NCA_ERROR_CODE = {
    0: 'NC_ERROR_NONE',
    1: 'NC_ERROR_INVALID_PARAM',
    2: 'NC_ERROR_MEMALLOC_FAILED',
    3: 'NC_ERROR_FAILED',
    4: 'NC_COMPILE_MODEL_FAILED',
    5: 'NC_LOAD_MODEL_FAILED',
    6: 'NC_RUN_MODEL_FAILED',
    7: 'NC_MODEL_NOT_LOADED',
    8: 'NC_INVALID_SESSION',
    9: 'NC_INIT_COMPILER_FAILED',
    10: 'NC_INIT_EXECUTOR_FAILED',
    11: 'NC_SET_DEVICE_FAILED',
    12: 'NC_PARSER_JSON_FAILED',
    13: 'NC_PROF_MANAGER_FAILED',
    100: 'NC_NULL_PTR',
    101: 'FILE_OPEN_FAILED'
}
DATA_TYPE_DICT = {
    'float32': 'float',
    'float16': 'float16',
    'int32': 'int32'
}


class NcxConf(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'NcxConf'
    """
    _fields_ = [("ipLength", ctypes.c_uint), ("confIp", ctypes.c_char_p), \
                ("portLength", ctypes.c_uint), ("confPort", ctypes.c_char_p)]


class SocVersion(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'SocVersion'
    """
    _fields_ = [("length", ctypes.c_uint), ("socVersion", ctypes.c_char_p)]


class ModelInfo(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'ModelInfo'
    """
    _fields_ = [("modelPathlength", ctypes.c_uint),
                ("modelPath", ctypes.c_char_p)]


class ProfilingInfo(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'ProfilingInfo'
    """
    _fields_ = [("resultPathlength", ctypes.c_uint),
                ("resultPath", ctypes.c_char_p)]


class ResultInfo(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'ResultInfo'
    """
    _fields_ = [("model_time", ctypes.c_float), ("error_code", ctypes.c_uint),
                ("model_id", ctypes.c_uint), ("session_id", ctypes.c_char_p)]


class InputOutput(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'InputOutput'
    """
    _fields_ = [("input_num", ctypes.c_uint), ("output_num", ctypes.c_uint)]


class DataBuffer(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'VoidData'
    """
    _fields_ = [("length", ctypes.c_uint),
                ("data", ctypes.POINTER(ctypes.c_void_p)),
                ("type", ctypes.c_char_p)]


class SessionOnAscend(): # pylint: disable=too-many-instance-attributes
    """The class for managing board session on ascend.

    Args:
        ncx_conf (str): a string of ncx config file path.
        lib_path (str): a string of amct_ncx lib path.
    """
    def __init__(self, ncx_conf, lib_path=NCX_LIB_PATH):
        self.data_in_info = None
        self.data_out_info = None
        self.result_info = None
        self.input_output = None
        self.model_info = None
        self.profiling_info = None
        self.sess_closed = True

        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.lib.CreateSessionOnAscend.restype = (ResultInfo)
        self.lib.CloseSessionOnAscend.restype = (ResultInfo)
        self.lib.LoadModelOnAscend.restype = (ResultInfo)
        self.lib.RunModelOnAscend.restype = (ResultInfo)
        self.ncx_conf = \
            NcxConf(len(ncx_conf['ip']),
                    ctypes.c_char_p(bytes(ncx_conf['ip'], 'utf-8')),
                    len(ncx_conf['port']),
                    ctypes.c_char_p(bytes(ncx_conf['port'], 'utf-8')),)
        self._open()

    def __del__(self):
        if not self.sess_closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.sess_closed:
            self.close()

    @staticmethod
    def _get_data_info(data_list):
        data_info = []
        for data_item in data_list:
            if str(data_item.dtype) not in DATA_TYPE_DICT.keys():
                LOGGER.loge('Data type[{}] not supported[RunModel]!'.format(str(data_item.dtype)),
                            ASCEND_SAMPLER)
                raise TypeError('Data type[{}] not supported[RunModel]!'.format(str(data_item.dtype)))
            if not data_item.flags['C_CONTIGUOUS']:
                data_item = np.ascontiguousarray(data_item,
                                                 dtype=data_item.dtype)
            data_ptr = ctypes.cast(data_item.ctypes.data,
                                   ctypes.POINTER(ctypes.c_void_p))
            data_info.append(DataBuffer(
                data_item.size, data_ptr, ctypes.c_char_p(
                    bytes(DATA_TYPE_DICT.get(str(data_item.dtype)),
                          'utf-8'))))
        return data_info

    def close(self):
        """Close the ncx session."""
        LOGGER.logi('start close ncx session!', ASCEND_SAMPLER)
        self.result_info = self.lib.CloseSessionOnAscend(
            self.result_info.session_id)
        if self.result_info.error_code:
            LOGGER.loge('NCA report error when executing [CloseSession]!',
                        ASCEND_SAMPLER)
            raise RuntimeError(NCA_ERROR_CODE.get(self.result_info.error_code))
        LOGGER.logi(
            'sucess close ncx session, session_id[{}]!'.format(
                self.result_info.session_id), ASCEND_SAMPLER)
        self.sess_closed = True

    def load_model(self, model_path):
        """Remote loading om model."""
        LOGGER.logi('start load model, model_path[{}]!'.format(model_path),
                    ASCEND_SAMPLER)
        if not os.path.exists(model_path):
            LOGGER.loge(
                'Model file[{}] does not exit. [LoadModel]!'.format(model_path),
                ASCEND_SAMPLER)
            raise FileExistsError(
                'Model file[{}] does not exit. [LoadModel]!'.format(model_path))
        self.model_info = \
            ModelInfo(len(model_path),
                      ctypes.c_char_p(bytes(model_path, 'utf-8')))

        self.result_info = self.lib.LoadModelOnAscend(
            self.model_info, self.result_info.session_id)
        if self.result_info.error_code:
            LOGGER.loge('NCA report error when executing [LoadModel]!',
                        ASCEND_SAMPLER)
            raise RuntimeError(NCA_ERROR_CODE.get(self.result_info.error_code))
        LOGGER.logi(
            'sucess load model, model_id[{}], session_id[{}]!'.format(
                self.result_info.model_id, self.result_info.session_id),
            ASCEND_SAMPLER)

    def run_model(self, result_path, loop_count, inputs=None, outputs=None):
        """Remote execution om model."""
        LOGGER.logi('start run model, result_path[{}]!'.format(result_path),
                    ASCEND_SAMPLER)
        inputs = [] if inputs is None else inputs
        outputs = [] if outputs is None else outputs
        self.profiling_info = \
            ProfilingInfo(len(result_path),
                          ctypes.c_char_p(bytes(result_path, 'utf-8')))
        self.input_output = InputOutput(len(inputs), len(outputs))

        self.data_in_info = SessionOnAscend._get_data_info(inputs)
        self.data_out_info = SessionOnAscend._get_data_info(outputs)
        self.result_info = self.lib.RunModelOnAscend(
            self.profiling_info,
            self.result_info.model_id, self.result_info.session_id,
            ctypes.c_uint(loop_count), self.input_output, *self.data_in_info,
            *self.data_out_info)
        if self.result_info.error_code:
            LOGGER.loge('NCA report error when executing [RunModel]!',
                        ASCEND_SAMPLER)
            raise RuntimeError(NCA_ERROR_CODE.get(self.result_info.error_code))
        LOGGER.logi(
            'sucess run model, model_id[{}], session_id[{}], model_time[{}]!'.
            format(self.result_info.model_id, self.result_info.session_id,
                   self.result_info.model_time), ASCEND_SAMPLER)

    def _open(self):
        LOGGER.logi('start create ncx session!', ASCEND_SAMPLER)
        self.result_info = self.lib.CreateSessionOnAscend(self.ncx_conf)
        if self.result_info.error_code:
            LOGGER.loge('NCA report error when executing [CreateSession]!',
                        ASCEND_SAMPLER)
            raise RuntimeError(NCA_ERROR_CODE.get(self.result_info.error_code))
        LOGGER.logi(
            'sucess create ncx session, session_id[{}]!'.format(
                self.result_info.session_id), ASCEND_SAMPLER)
        self.sess_closed = False


class NcxExecuter():
    """NCX remote model execution class."""
    def __init__(self):
        pass

    @staticmethod
    def get_om_profiling(config, om_model, csv_file):
        """NCX gets the performance data of OM model running on board."""
        with SessionOnAscend(
                config['distribute_config']) as session_on_ascend:
            session_on_ascend.load_model(om_model)
            session_on_ascend.run_model(csv_file, config['loop_count'])

        if not os.path.exists(csv_file):
            LOGGER.loge('Failed to generate the profiling file!', 'NcxExecuter')
            raise RuntimeError('Failed to generate the profiling file!')

    @staticmethod
    def get_om_output(config, om_model, csv_file, inputs, outputs): # pylint: disable=R0913
        """NCX gets the output data of OM model running on board."""
        with SessionOnAscend(
                config['distribute_config']) as session_on_ascend:
            session_on_ascend.load_model(om_model)
            session_on_ascend.run_model(csv_file, config['loop_count'],
                                        inputs, outputs)
