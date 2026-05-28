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

import threading
from functools import wraps

from ...amct_pytorch.proto import scale_offset_record_pb2


def singleton(cls):
    """singleton decorator"""
    _instances = {}
    _instances_lock = threading.Lock()

    @wraps(cls)
    def _wrapper(*args, **kwargs):
        if cls not in _instances:
            with _instances_lock:
                if cls not in _instances:
                    _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return _wrapper


@singleton
class SingletonScaleOffsetRecord():
    """
    Function: singleton for ScaleOffsetRecord.
    APIs: __init__, forward
    """
    def __init__(self, *args, **kwargs):
        self.record = scale_offset_record_pb2.ScaleOffsetRecord()
        self.prune_record = {}
        self.record_file = None
        self.lock = threading.Lock()

    def reset_singleton(self, record_file):
        self.record = scale_offset_record_pb2.ScaleOffsetRecord()
        self.prune_record = {}
        self.record_file = record_file

    def reset_record(self):
        self.record = scale_offset_record_pb2.ScaleOffsetRecord()
