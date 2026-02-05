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

from collections.abc import Iterable
import os
import sys
import functools
from google.protobuf import text_format
from .util import proto_float_to_python_float
from .vars_util import INT16
from .vars_util import DATA_OFFSET_RANGE, DATA_OFFSET_RANGE_INT16
from .files import create_empty_file
from ..config.field import ACT_SUPPORT_NUM_BITS, WTS_SUPPORT_NUM_BITS

DBL_EPSILON = sys.float_info.epsilon


def dst_type_generator(num_bits, support_scope):
    """
    Function: convert num_bits to 'INT${num_bits}' format
    Parameters: num_bits: num_bits of quantization
                support_scope: support num bits scope
    Return: 'INT${num_bits}'
    """
    if num_bits in support_scope:
        return 'INT{}'.format(num_bits)
    else:
        raise ValueError('Support num_bits is {}, current is {}.'.format(support_scope, num_bits))


def record_weights_scale_offset(records, layer_name, scale, offset, num_bits=None, scale_r=None, offset_r=None):
    """
    Function: Write scale_w and offset_w to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of scale_w, offset_w
                scale: vector of scale_w
                offset: vector of offset_w
    Return: None
    """
    done_flag = False

    for record in records.record:
        if record.key == layer_name:
            record.value.scale_w[:] = scale
            record.value.offset_w[:] = offset
            if num_bits is not None:
                record.value.wts_type = dst_type_generator(num_bits, WTS_SUPPORT_NUM_BITS)
            if scale_r is not None:
                record.value.scale_r[:] = scale_r
            if offset_r is not None:
                record.value.offset_r[:] = offset_r
            done_flag = True
            break
    if not done_flag:
        record = records.record.add()
        record.key = layer_name
        record.value.scale_w[:] = scale
        record.value.offset_w[:] = offset
        if num_bits is not None:
            record.value.wts_type = dst_type_generator(num_bits, WTS_SUPPORT_NUM_BITS)
        if scale_r is not None:
            record.value.scale_r[:] = scale_r
        if offset_r is not None:
            record.value.offset_r[:] = offset_r


def record_recurrence_weights_scale_offset(records, layer_name, scale, offset):
    """
    Function: Write scale_r and offset_r to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of scale_r, offset_r
                scale: vector of scale_r
                offset: vector of offset_r
    Return: None
    """
    done_flag = False
    for record in records.record:
        if record.key == layer_name:
            record.value.scale_r[:] = scale
            record.value.offset_r[:] = offset
            done_flag = True
            break
    if not done_flag:
        record = records.record.add()
        record.key = layer_name
        record.value.scale_r[:] = scale
        record.value.offset_r[:] = offset


def record_skip_status(records, layer_name, is_skip_fusion):
    """
    Function: Write scale_w and offset_w to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of scale_w, offset_w
                scale: vector of scale_w
                offset: vector of offset_w
    Return: None
    """
    done_flag = False
    for record in records.record:
        if record.key == layer_name:
            record.value.skip_fusion = is_skip_fusion
            done_flag = True
            break
    if not done_flag:
        record = records.record.add()
        record.key = layer_name
        record.value.skip_fusion = is_skip_fusion


def read_weights_scale_offset(records, layer_name):
    """
    Function: Read scale_w and offset_w from record file
    Parameters: records: ScaleOffsetRecord() object to read
                layer_name: layer name of scale_w, offset_w
    Return: scale: vector of scale_w
            offset: vector of offset_w
    """
    done_flag = False
    scale = []
    offset = []
    for record in records.record:
        if record.key == layer_name:
            # Read scale_w from record file
            if not record.value.scale_w:
                raise RuntimeError("Cannot find scale_w of layer '{}' " \
                    "in record file".format(layer_name))
            scale.extend(record.value.scale_w)
            # Read offset_w from record file
            if not record.value.offset_w:
                raise RuntimeError("Cannot find offset_w of layer \'{}\' " \
                    "in record file".format(layer_name))
            offset.extend(record.value.offset_w)
            done_flag = True
            break
    if not done_flag:
        raise RuntimeError("Cannot find layer '{}' in record " \
            "file".format(layer_name))
    return scale, offset


def record_shift_bits(records, layer_name, shift_bit):
    """
    Function: Write shift_bit to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of scale_w, offset_w
                shift_bit: vector of shift_bit
    Return: None
    """
    done_flag = False
    for record in records.record:
        if record.key == layer_name:
            record.value.shift_bit[:] = shift_bit
            done_flag = True
            break
    if not done_flag:
        record = records.record.add()
        record.key = layer_name
        record.value.shift_bit[:] = shift_bit


def read_shift_bits(records, layer_name):
    """
    Function: Read the number of bit to shift from record file
    Parameters: records: ScaleOffsetRecord() object to read
                layer_name: layer name to read shift_bits
    Return: : shift_bits: the number of bit to shift
    """
    shift_bits = []
    for record in records.record:
        if record.key == layer_name:
            if record.value.shift_bit:
                shift_bits.extend(record.value.shift_bit)
            break
    else:
        raise RuntimeError("Cannot find layer '{}' in record "
                           "file".format(layer_name))

    return shift_bits


def record_activation_scale_offset(records, layer_name, scale, offset, num_bits=None, scale_h=None, offset_h=None):
    """
    Function: Write scale_w and offset_w to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of scale_w, offset_w
                scale: vector of scale_w
                offset: vector of offset_w
    Return: None
    """
    key_found = False
    for record in records.record:
        if record.key == layer_name:
            record.value.scale_d = scale
            record.value.offset_d = offset
            if num_bits is not None:
                record.value.act_type = dst_type_generator(num_bits, ACT_SUPPORT_NUM_BITS)
            if scale_h is not None:
                record.value.scale_h = scale_h
            if offset_h is not None:
                record.value.offset_h = offset_h
            key_found = True
            break
    if not key_found:
        record = records.record.add()
        record.key = layer_name
        record.value.scale_d = scale
        record.value.offset_d = offset
        if num_bits is not None:
            record.value.act_type = dst_type_generator(num_bits, ACT_SUPPORT_NUM_BITS)
        if scale_h is not None:
            record.value.scale_h = scale_h
        if offset_h is not None:
            record.value.offset_h = offset_h


def read_activation_scale_offset(records, layer_name):
    """
    Function: Read scale_d and offset_d from record file
    Parameters: records: ScaleOffsetRecord() object to read
                layer_name: layer name of scale_d, offset_d
    Return: scale: scalar of scale_d
            offset: scalar of offset_d
    """
    done_flag = False
    scale = 1
    offset = 0
    for record in records.record:
        if record.key == layer_name:
            # Read scale_d from record file
            if not record.value.HasField('scale_d'):
                raise RuntimeError("Cannot find scale_d of layer '{}' " \
                    "in record file".format(layer_name))
            scale = record.value.scale_d
            # Read offset_d from record file
            if not record.value.HasField('offset_d'):
                raise RuntimeError("Cannot find offset_d of layer '{}' " \
                    "in record file".format(layer_name))
            offset = record.value.offset_d
            done_flag = True
            break
    if not done_flag:
        raise RuntimeError("Cannot find layer '{}' in record " \
            "file".format(layer_name))
    return scale, offset


def record_dmq_balancer_factor(records, layer_name, factor):
    """
    Function: Write tensor_balance_factor to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of tensor_balance_factor
                factor: vector of tensor_balance_factor
    Return: None
    """
    done_flag = False
    for record in records.record:
        if record.key == layer_name:
            record.value.tensor_balance_factor[:] = factor
            done_flag = True
            break
    if not done_flag:
        record = records.record.add()
        record.key = layer_name
        record.value.tensor_balance_factor[:] = factor


def create_empty_record(records, layer_name):
    """
    Function: create empty record for layer with name of layer_name
    param:records, proto type, add a new record for records
    param:layer_name, string, key of new record
    """
    for record in records.record:
        if record.key == layer_name:
            raise RuntimeError("the {} has already in records".format(layer_name))
    record = records.record.add()
    record.key = layer_name
    record.value.Clear()


def record_kv_cache_scale_offset(records, layer_name, scale, offset):
    """
    Function: Write scale and offset to record file
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name of scale, offset
                scale: vector of scale
                offset: vector of offset
    Return: None
    """
    done_flag = False
    for record in records.record:
        if record.key == layer_name:
            record.kv_cache_value.scale[:] = scale
            record.kv_cache_value.offset[:] = offset
            done_flag = True
            break
    if not done_flag:
        record = records.record.add()
        record.key = layer_name
        record.kv_cache_value.scale[:] = scale
        record.kv_cache_value.offset[:] = offset


def record_quant_factors(records, layer_name, quant_factors):
    """
    Function: Write quant factors into records
    Parameters: records: ScaleOffsetRecord() object to write
                layer_name: layer name to be recorded
                quant_factors: a dict including all factors to be recorded
    Return: None
    """
    for record in records.record:
        if record.key == layer_name:
            do_record_factors(record, quant_factors)
            return
    record = records.record.add()
    record.key = layer_name
    do_record_factors(record, quant_factors)


def do_record_factors(record, quant_factors):
    """
    Function: add quant factors to record objects
    Parameters: record: layer's ScaleOffsetRecord() object to write
                quant_factors: a dict including all factors to be recorded
    Return: None
    """
    for name, value in quant_factors.items():
        field = getattr(record.value, name)
        if isinstance(field, Iterable) and not isinstance(field, str):
            field[:] = value
        else:
            setattr(record.value, name, value)


class ScaleOffsetRecordHelper():
    """
    Utility class for reading, recording, modifying, and saving.

    Args:
        scale_offset_record (proto): proto message for scale and offset.
    """
    def __init__(self, scale_offset_record):
        self._record_file = None
        self._records = scale_offset_record()

    @property
    def keys(self):
        """
        Enumerates all keys.

        Returns:
            list: a list of keys string.
        """
        keys = list()
        for record in self._records.record:
            keys.append(record.key)
        return keys

    @property
    def records(self):
        """Return the instance itself"""
        return self._records

    @staticmethod
    def check_record(key, record_value):
        """
        Check if the record is valid.

        Args:
            key (str): a string of layer name.
            record_value : a record object.

        Returns:
            bool: whether the record is valid.
        """
        # scale_w and offset_w can be a vector, but must have same length
        if len(record_value.scale_w) != len(record_value.offset_w):
            raise RuntimeError('{} scale_w and offset_w must be same length'.format(key))

        # current layer is quant layer, need check parameters' legality
        scales = [proto_float_to_python_float(record_value.scale_d)] if record_value.HasField('scale_d') else []
        scales.extend([proto_float_to_python_float(value) for value in record_value.scale_w])
        for scale in scales:
            # scale_d and scale_w must in range DBL_EPSILON, 1/DBL_EPSILON
            if scale < DBL_EPSILON or scale > 1 / DBL_EPSILON:
                raise ValueError('Exist illegal scale {} in "{}"'.format(scale, key))
        if record_value.act_type == INT16:
            data_range = DATA_OFFSET_RANGE_INT16
        else:
            data_range = DATA_OFFSET_RANGE
        # offset_d if int8 must in range -128, 127, if int16 must in range -32768, 32767
        if record_value.offset_d < data_range[0] or record_value.offset_d > data_range[1]:
            raise ValueError('Exist illegal offset_d {} in "{}"'.format(record_value.offset_d, key))
        # offset_w must be zero
        for offset in record_value.offset_w:
            if offset != 0:
                raise ValueError('Offset_w must be 0, {} in "{}"'.format(offset, key))
        return True

    def init_from_file(self, record_file):
        """
        Initialize from a file.

        Args:
            record_file (str): a string of record file path.
        """
        record_file = os.path.realpath(record_file)
        self._record_file = record_file
        with open(record_file, 'r') as fid:
            pbtxt_string = fid.read()
            try:
                text_format.Merge(pbtxt_string, self._records)
            except text_format.ParseError as e:
                raise RuntimeError(
                    "the record_file{%s} cannot be parsered, please ensure "\
                    "it matches with scale_offset_record.proto!"
                    % (record_file)) from e
        return self

    def init(self, records):
        """
        Initialize from an ScaleOffsetRecord object.

        Args:
            records (ScaleOffsetRecord): a instance of record.
        """
        self._records = records
        return self

    def merge(self, records):
        """
        Merge records into the current instance.

        Args:
            records (ScaleOffsetRecord): a instance of record.
        """
        for record in records.record:
            if self.has_key(record.key):
                raise RuntimeError('Merge failed, alrady has key "%s"' % record.key)
            self._records.record.append(record)

    def has_key(self, key):
        """
        Determines whether the instance has the key.

        Args:
            key (str): a string of layer name.

        Returns:
            bool: whether the key exists.
        """
        for record in self._records.record:
            if record.key == key:
                return True
        return False

    def delete_key(self, key):
        """
        Delete the key from the instance.

        Args:
            key (str): a string of layer name.

        Returns:
            bool: whether the key has been deleted.
        """
        for index, record in enumerate(self._records.record):
            if record.key == key:
                del self._records.record[index]
                return True
        return False

    def get_record(self, key):
        """
        Get the value of the key.

        Args:
            key (str): a string of layer name.

        Returns:
            dict: the value of the key.
        """
        for record in self._records.record:
            if record.key == key:
                return record.value
        return None

    def record_weights_scale_offset(self, key, scale, offset, dst_type=None):
        """
        Record scale and offset of the layer's weight.

        Args:
            key (str): the layer name.
            scale (list): a list of scale value.
            offset (list): a list of offset value.
            dst_type (str): the weight type
        """
        done_flag = False
        for record in self.records.record:
            if record.key == key:
                record.value.scale_w[:] = scale
                record.value.offset_w[:] = offset
                if dst_type is not None:
                    record.value.wts_type = dst_type
                done_flag = True
                break
        if not done_flag:
            record = self.records.record.add()
            record.key = key
            record.value.scale_w[:] = scale
            record.value.offset_w[:] = offset
            if dst_type is not None:
                record.value.wts_type = dst_type

    def record_recurrence_weights_scale_offset(self, key, scale, offset):
        """
        Record scale_r and offset_r of the layer's weight.

        Args:
            key (str): the layer name.
            scale (list): a list of scale value.
            offset (list): a list of offset value.
        """
        record_recurrence_weights_scale_offset(self.records, key, scale, offset)

    def record_skip_status(self, key, is_skip_fusion):
        """
        Record whether the layer skips fusion.

        Args:
            key (str): the layer name.
            is_skip_fusion (bool): whether skips fusion.
        """
        done_flag = False
        for record in self.records.record:
            if record.key == key:
                record.value.skip_fusion = is_skip_fusion
                done_flag = True
                break
        if not done_flag:
            record = self.records.record.add()
            record.key = key
            record.value.skip_fusion = is_skip_fusion

    def read_weights_scale_offset(self, key):
        """
        Read scale and offset of the layer's weight.

        Args:
            key (str): the layer name.

        Returns:
            list: a list of scale value.
            list: a list of offset value.
        """
        done_flag = False
        scale = []
        offset = []
        for record in self.records.record:
            if record.key == key:
                # Read scale_w from record file
                if not record.value.scale_w:
                    raise RuntimeError("Cannot find scale_w of layer '{}' " \
                        "in record file".format(key))
                scale.extend(record.value.scale_w)
                # Read offset_w from record file
                if not record.value.offset_w:
                    raise RuntimeError("Cannot find offset_w of layer \'{}\' " \
                        "in record file".format(key))
                offset.extend(record.value.offset_w)
                done_flag = True
                break
        if not done_flag:
            raise RuntimeError("Cannot find layer '{}' in record file".format(key))
        return scale, offset

    def record_shift_bits(self, key, shift_bit):
        """
        Record the shift bit of layer.

        Args:
            key (str): the layer name.
            shift_bit (list): a list of shift bit.
        """
        done_flag = False
        for record in self.records.record:
            if record.key == key:
                record.value.shift_bit[:] = shift_bit
                done_flag = True
                break
        if not done_flag:
            record = self.records.record.add()
            record.key = key
            record.value.shift_bit[:] = shift_bit

    def read_shift_bits(self, key):
        """
        Read the number of bit to shift from record file.

        Args:
            key (str): the layer name.

        Returns:
            int: shift bit value.
        """
        shift_bits = []
        for record in self.records.record:
            if record.key == key:
                if record.value.shift_bit:
                    shift_bits.extend(record.value.shift_bit)
                break
        else:
            raise RuntimeError("Cannot find layer '{}' in record file".format(key))

        return shift_bits

    def record_activation_scale_offset(self, key, scale, offset, dst_type=None):
        """
        Reccord scale and offset of layer's activation.

        Args:
            key (str): the layer name.
            scale (float): the scale value.
            offset (int): the offset value.
            dst_type (str): the activation type
        """
        key_found = False
        for record in self.records.record:
            if record.key == key:
                record.value.scale_d = scale
                record.value.offset_d = offset
                if dst_type is not None:
                    record.value.act_type = dst_type
                key_found = True
                break

        if not key_found:
            record = self.records.record.add()
            record.key = key
            record.value.scale_d = scale
            record.value.offset_d = offset
            if dst_type is not None:
                record.value.act_type = dst_type

    def record_activation_h_scale_offset(self, key, scale, offset):
        """
        Reccord scale_h and offset_h of layer's initial_h.

        Args:
            key (str): the layer name.
            scale (float): the scale value.
            offset (int): the offset value.
        """
        key_found = False
        for record in self.records.record:
            if record.key == key:
                record.value.scale_h = scale
                record.value.offset_h = offset
                key_found = True
                break

        if not key_found:
            record = self.records.record.add()
            record.key = key
            record.value.scale_h = scale
            record.value.offset_h = offset

    def record_tensor_quant_scale_offset(self, key, scale, offset, dst_type):
        """
        Reccord scale and offset of layer's activation.

        Args:
            key (str): the layer name.
            scale (float): the scale value.
            offset (int): the offset value.
            dst_type (str): the activation type
        """
        key_found = False
        for record in self.records.record:
            if record.key == key:
                record.value.is_tensor_quantize = True
                record.value.scale_d = scale
                record.value.offset_d = offset
                record.value.act_type = dst_type
                key_found = True
                break

        if not key_found:
            record = self.records.record.add()
            record.key = key
            record.value.scale_d = scale
            record.value.offset_d = offset
            record.value.is_tensor_quantize = True
            record.value.act_type = dst_type

    def read_activation_scale_offset(self, key):
        """
        Read scale and offset of the layer's activation.

        Args:
            key (str): the layer name.

        Returns:
            float: the scale value.
            int: the offset value.
        """
        done_flag = False
        scale = 1
        offset = 0
        for record in self.records.record:
            if record.key == key:
                # Read scale_d from record file
                if not record.value.HasField('scale_d'):
                    raise RuntimeError("Cannot find scale_d of layer '{}' " \
                        "in record file".format(key))
                scale = record.value.scale_d
                # Read offset_d from record file
                if not record.value.HasField('offset_d'):
                    raise RuntimeError("Cannot find offset_d of layer '{}' " \
                        "in record file".format(key))
                offset = record.value.offset_d
                done_flag = True
                break
        if not done_flag:
            raise RuntimeError("Cannot find layer '{}' in record " \
                "file".format(key))
        return scale, offset

    def update_record(self):
        """Update the record file with the latest records."""
        if self._record_file is None:
            raise RuntimeError('Cannot update record without record file')

        with open(self._record_file, "w") as record_write_file:
            record_write_file.write(text_format.MessageToString(
                self._records, as_utf8=True))

    def dump(self, record_file):
        """
        Write records to the record file.

        Args:
            record_file (str): a string of record file path.
        """
        with open(record_file, "w") as record_write_file:
            record_write_file.write(text_format.MessageToString(
                self._records, as_utf8=True))
