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

Top APIs of AMCT_TORCH prune tool for user

"""

import threading
from functools import wraps

from amct_pytorch.proto import scale_offset_record_pb2


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
