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
from google.protobuf import text_format

from amct_pytorch.amct_pytorch_inner.amct_pytorch.proto import scale_offset_record_pytorch_pb2

def get_producer(prune_record):
    producer_names = [producer.name for producer in prune_record.producer]
    consumer_names = [consumer.name for consumer in prune_record.consumer]
    return producer_names, consumer_names

def read_record_file(record_file):
    record = scale_offset_record_pytorch_pb2.ScaleOffsetRecord()
    with open(record_file, 'r') as fid:
        pbtxt_string = fid.read()
        try:
            text_format.Merge(pbtxt_string, record)
        except text_format.ParseError:
            raise RuntimeError(
                "the record_file{%s} cannot be parsered, please ensure "\
                "it matches with scale_offset_record.proto!"
                % (record_file))

    return record