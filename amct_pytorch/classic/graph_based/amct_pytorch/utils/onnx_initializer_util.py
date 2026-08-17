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
import numpy as np

from google.protobuf.internal import api_implementation
from onnx.onnx_pb import TensorProto  # pylint: disable=E0401

from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.common.utils import files as files_util

RAW_DATA = 'raw_data'


def pack_int4_to_int8(arr):
    """
    把每两个 INT4 值（值域 [-8,7]）打包成一个 int8 字节：低 4 位存第一个、高 4 位存第二个。
    奇数长度末尾补 0。返回 np.int8 数组。
    """
    flat = np.asarray(arr).reshape(-1).astype(np.int8)
    if flat.size % 2 == 1:
        LOGGER.logw(
            'INT4 weight has odd length {}, padding one zero for packing'.format(
                flat.size
            ),
            'pack_int4_to_int8',
        )
        flat = np.append(flat, np.int8(0))
    low = flat[0::2] & 0x0F
    high = flat[1::2] & 0x0F
    packed = (low | (high << 4)).astype(np.uint8)
    return packed.astype(np.int8)


def unpack_int8_to_int4(packed, count):
    """把打包的 int8 还原成 count 个 INT4 值（值域 [-8,7]，np.int8 数组）。"""
    raw = np.asarray(packed).reshape(-1).astype(np.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    out = np.empty(raw.size * 2, dtype=np.int8)
    out[0::2] = low
    out[1::2] = high
    # 4-bit 有符号还原：>7 的值减 16
    out = np.where(out > 7, out - 16, out).astype(np.int8)
    return out[:count]


def parse_external_data_meta(external_data):
    """把 external_data 条目解析成 (file_name, offset, length, quantized_data)。"""
    file_name, offset, length, quantized_data = None, 0, -1, None
    for data in external_data:
        if data.key == 'location':
            file_name = data.value
        elif data.key == 'offset':
            offset = int(data.value)
        elif data.key == 'length':
            length = int(data.value)
        elif data.key == 'amct_quantized_raw_data':
            quantized_data = bool(data.value)
    return file_name, offset, length, quantized_data


class TensorProtoHelper:
    """Help cope with onnx.TensorProto
    APIs: map_data_location, map_np_type, get_data, clear_data, set_data
    """

    data_type = TensorProto.DataType  # pylint: disable=E1101
    data_type_maps = {
        # DataType: proto_value, data_location, np.type
        'UNDEFINED': [data_type.UNDEFINED, RAW_DATA, None],
        'FLOAT': [data_type.FLOAT, 'float_data', 'float32'],
        'UINT8': [data_type.UINT8, 'int32_data', 'uint8'],
        'INT8': [data_type.INT8, RAW_DATA, 'int8'],
        'UINT16': [data_type.UINT16, 'int32_data', 'uint16'],
        'INT16': [data_type.INT16, 'int32_data', 'int16'],
        'INT32': [data_type.INT32, 'int32_data', 'int32'],
        'INT64': [data_type.INT64, 'int64_data', 'int64'],
        'STRING': [data_type.STRING, 'string_data', 'str'],
        'BOOL': [data_type.BOOL, 'int32_data', 'bool_'],
        'FLOAT16': [data_type.FLOAT16, 'int32_data', 'float16'],
        'DOUBLE': [data_type.DOUBLE, 'double_data', 'float64'],
        'UINT32': [data_type.UINT32, 'uint64_data', 'uint32'],
        'UINT64': [data_type.UINT64, 'uint64_data', 'uint64'],
        'COMPLEX64': [data_type.COMPLEX64, 'float_data', 'complex64'],
        'COMPLEX128': [data_type.COMPLEX128, 'double_data', 'complex128'],
        'BFLOAT16': [data_type.BFLOAT16, RAW_DATA, 'float32'],
    }
    # 旧版 onnx 无 INT4 枚举，动态注册避免 import 时报错
    if hasattr(data_type, 'INT4'):
        data_type_maps['INT4'] = [data_type.INT4, RAW_DATA, 'int4']
    proto_value_id = 0
    data_location_id = 1
    np_type_id = 2

    def __init__(self, tensor, model_path=''):
        '''init function'''
        super().__init__()
        self.tensor = tensor
        self.model_path = model_path
        self.externel_data = self.check_external_data()

    @classmethod
    def map_data_location(cls, proto_value):
        '''find data_location according to TensorProto.data_type'''
        for key in cls.data_type_maps:
            value = cls.data_type_maps[key]
            if proto_value == value[cls.proto_value_id]:
                return value[cls.data_location_id]

        raise ValueError(f'The data_type{proto_value} is UNEXPECTED')

    @classmethod
    def map_np_type(cls, proto_value):
        '''find np's dtype according to TensorProto.data_type'''
        for key in cls.data_type_maps:
            value = cls.data_type_maps[key]
            if proto_value == value[cls.proto_value_id]:
                return value[cls.np_type_id]

        raise ValueError(f'The data_type{proto_value} is UNEXPECTED')

    @classmethod
    def cast_ori_data(cls, value, tensor_np_type):
        '''cast ori-data to numpy type'''
        if tensor_np_type == 'float16':
            value = np.array(value).astype(np.uint16).tobytes()
            np_value = np.frombuffer(value, np.dtype(tensor_np_type))
            np_value = np.array(np_value)
        else:
            np_value = np.array(value, np.dtype(tensor_np_type))
        return np_value

    def check_external_data(self):
        return self.tensor.HasField('data_location') and self.tensor.data_location == 1

    def get_data(self):
        '''
        Function: get data from tensor. Return a numpy.array If
            tensor.data_type is not UNDEFINED, otherwise return a binary
        Parameters:
            tensor: a instance of TensorProto
        Return: value: byte_value or np_value
        '''
        if self.externel_data:
            return self.get_external_data()[0]

        tensor_data_type = self.tensor.data_type
        byte_value = self.tensor.raw_data

        if tensor_data_type == self.data_type_maps['UNDEFINED'][self.proto_value_id]:
            return byte_value

        np_type = self.map_np_type(tensor_data_type)
        if byte_value:
            if np_type == 'int4':
                packed = np.frombuffer(byte_value, np.uint8)
                count = int(np.prod(self.tensor.dims))
                np_value = unpack_int8_to_int4(packed, count)
            else:
                np_dtype = np.dtype(np_type)
                np_value = np.frombuffer(byte_value, np_dtype)
            # to modify np_value.flags['WRITEABLE'] as True
            np_value = np.array(np_value)
        else:
            value = getattr(self.tensor, self.map_data_location(tensor_data_type))
            np_value = self.cast_ori_data(value, np_type)
        np_value = np_value.reshape(self.tensor.dims)
        return np_value

    def clear_data(self):
        '''
        Function: clear data of tensor, the raw_data is cleared if it exists.
        Parameters:
            tensor: a instance of TensorProto
        Return: None
        '''
        byte_value = self.tensor.raw_data
        if byte_value:
            self.tensor.ClearField(RAW_DATA)
        else:
            self.tensor.ClearField(self.map_data_location(self.tensor.data_type))

    def set_data(self, data, type_string=None, dims=None):
        '''
        Function: set data from tensor. The raw data is unexcepted
        Parameters:
            tensor: a instance of TensorProto
            data: a list containing the data
            type_string: a string to indicating the data_type, is must be in
                data_type_maps's keys
            dims: the dim of data
        Return: None
        '''
        if type_string is not None:
            entry = self.data_type_maps.get(type_string)
            if entry is None:
                raise ValueError(f'unsupported data type {type_string} in current onnx')
            self.tensor.data_type = entry[self.proto_value_id]
        if dims is not None:
            self.tensor.ClearField('dims')
            self.tensor.dims.extend(dims)

        data_location = self.map_data_location(self.tensor.data_type)
        data_location_new = self._rematch_data_location(
            data_location, self.tensor.data_type
        )
        if self.externel_data:
            return self.set_external_data(data)
        int4_entry = self.data_type_maps.get('INT4')
        if data_location == RAW_DATA:
            # INT4: nibble-pack two INT4 values into each INT8 byte
            if (
                int4_entry is not None
                and self.tensor.data_type == int4_entry[self.proto_value_id]
            ):
                packed = pack_int4_to_int8(data)
                setattr(self.tensor, RAW_DATA, bytes(packed.flatten().astype('uint8')))
            else:
                setattr(self.tensor, data_location, bytes(data.flatten()))
        else:
            self.tensor.ClearField(data_location)
            if data_location_new == RAW_DATA:
                setattr(self.tensor, RAW_DATA, bytes(data.flatten()))
                return
            if api_implementation.Type() == 'python':
                getattr(self.tensor, data_location).extend(data)
            else:
                getattr(self.tensor, data_location).MergeFrom(data)

    def get_external_data(self):
        '''
        Function: get data from external file. Return a numpy.array If
            tensor.data_type is not UNDEFINED, otherwise return a binary
        Parameters:
            tensor: a instance of TensorProto
        Return: value: raw_value or np_value; length: data length
        '''
        tensor_np_type = self.map_np_type(self.tensor.data_type)
        file_name, offset, length, quantized_data = parse_external_data_meta(
            self.tensor.external_data
        )

        if quantized_data:
            return self._get_quantized_raw_data(tensor_np_type), length

        # file path relative to the filesystem directory where the ONNX protobuf
        # model was stored. Data in external files uses the same binary bytes
        # format as the raw_data field in current ONNX implementations.
        if file_name is None:
            raise ValueError('The external_data is UNEXCEPTED, unspecified file path')
        file_name = os.path.realpath(os.path.join(self.model_path, file_name))

        if tensor_np_type is None:
            with open(file_name, 'rb') as f:
                f.seek(offset, 0)
                raw_data = f.read(length)
            return raw_data, length

        return self._read_external_file(file_name, offset, length, tensor_np_type)

    def set_external_data(self, data):
        '''
        Function: append quantized data to raw_data.
        Parameters:
            data: np_value
        Return: None
        '''
        quantized_data = self.tensor.external_data.add()
        quantized_data.key = 'amct_quantized_raw_data'
        quantized_data.value = str(1)
        if self.map_np_type(self.tensor.data_type) == 'int4':
            # 与 raw_data 路径一致：INT4 nibble-pack 成 packed 字节后存储
            packed = pack_int4_to_int8(data)
            setattr(self.tensor, RAW_DATA, bytes(packed.flatten().astype('uint8')))
        else:
            setattr(self.tensor, RAW_DATA, bytes(data.flatten()))

    def save_external_data(self, external_file):
        """
        Function: write data to external file
        Inpusts:
            initializer: TensorProto
            external_file: filename to save external data
        Return: None
        """
        tensor_dtype = self.tensor.data_type
        np_type = self.map_np_type(tensor_dtype)
        data_location = self.map_data_location(tensor_dtype)
        # INT4 以 packed 字节存储，按 uint8 搬运字节流；其余类型按自身 dtype
        store_np_type = np.uint8 if np_type == 'int4' else np.dtype(np_type)
        byte_size = np.dtype(store_np_type).itemsize
        # the list is index of TensorProto.Datetype
        if (
            tensor_dtype not in TensorProtoHelper.data_type.values()
            or tensor_dtype == TensorProtoHelper.data_type.STRING
        ):
            LOGGER.logd(
                "this patch does not support dtype %s for now" % (tensor_dtype), 'Utils'
            )
            return
        # DEFAULT is 0, EXTERNAL is 1
        if self.tensor.data_location == 0:
            self.tensor.ClearField('data_location')
            setattr(self.tensor, 'data_location', 1)
            raw_data = self.tensor.raw_data
            if raw_data:
                data_length = len(self.tensor.raw_data)
                np_value = np.frombuffer(self.tensor.raw_data, store_np_type)
                self.tensor.ClearField(RAW_DATA)
            else:
                data_length = len(getattr(self.tensor, data_location)) * byte_size
                np_value = np.array(getattr(self.tensor, data_location), store_np_type)
                self.tensor.ClearField(data_location)
            self.export_external_data(np_value, external_file, data_length)
            return
        else:
            if not self.tensor.HasField(RAW_DATA):
                np_value, data_length = self.get_external_data()
                self.tensor.ClearField(data_location)
            else:
                data_length = len(self.tensor.raw_data)
                np_value = np.frombuffer(self.tensor.raw_data, store_np_type)
                self.tensor.ClearField(RAW_DATA)
            self.export_external_data(np_value, external_file, data_length)
            return

    def export_external_data(self, np_value, external_file, data_length):
        self.tensor.ClearField('external_data')
        location = self.tensor.external_data.add()
        location.key = 'location'
        location.value = os.path.basename(external_file)

        offset_data = self.tensor.external_data.add()
        offset_data.key = 'offset'
        offset_data.value = str(0)

        length = self.tensor.external_data.add()
        length.key = 'length'
        length.value = str(data_length)
        with open(external_file, 'wb') as f:
            f.write(np_value.flatten())
            LOGGER.logi("external data %s" % (external_file), 'Utils')
        os.chmod(external_file, files_util.FILE_MODE)

    def _rematch_data_location(self, data_location, tensor_data_type):
        '''rematch data location for save data'''
        if tensor_data_type == self.data_type.FLOAT16:
            data_location = RAW_DATA
        return data_location

    def _get_quantized_raw_data(self, tensor_np_type):
        '''read quantized data already stored in tensor.raw_data'''
        if tensor_np_type == 'int4':
            # raw_data 为 packed 字节，unpack 成 prod(dims) 个 INT4 再 reshape
            packed = np.frombuffer(self.tensor.raw_data, np.uint8)
            np_value = unpack_int8_to_int4(packed, int(np.prod(self.tensor.dims)))
        else:
            np_value = np.frombuffer(self.tensor.raw_data, tensor_np_type)
        return np_value.reshape(self.tensor.dims)

    def _read_external_file(self, file_name, offset, length, tensor_np_type):
        '''read data from the external file for a typed (non-raw) tensor.
        Returns (np_value, length).'''
        if tensor_np_type == 'int4':
            # 外部文件存 packed 字节，按 uint8 读入后 unpack 成 prod(dims) 个 INT4
            byte_len = length if length > 0 else -1
            packed = np.fromfile(file_name, np.uint8, byte_len, '', offset)
            if length == -1:
                length = len(packed)
            np_value = unpack_int8_to_int4(packed, int(np.prod(self.tensor.dims)))
            return np_value.reshape(self.tensor.dims), length

        item_size = np.dtype(tensor_np_type).itemsize
        data_length = int(length / item_size) if length > 0 else length
        np_value = np.fromfile(file_name, tensor_np_type, data_length, '', offset)
        if length == -1:
            length = len(np_value.flatten()) * item_size
        if int(np.prod(self.tensor.dims)) != len(np_value.flatten()):
            raise ValueError('The external_data is not consistant with the shape')
        return np_value.reshape(self.tensor.dims), length
