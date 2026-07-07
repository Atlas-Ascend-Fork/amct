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
__all__ = ['check_params']

from inspect import signature
from functools import wraps


def check_params(*type_args, **type_kwargs):
    '''decorator util for check function params type '''
    def decorate(func):
        ''' decorate function. '''
        func_sig = signature(func)
        need_param_types = func_sig.bind_partial(*type_args,
                                                 **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            ''' decorate wrapper. '''
            func_param_types = func_sig.bind(*args, **kwargs)
            for name, value in func_param_types.arguments.items():
                if name in need_param_types:
                    if not isinstance(value, need_param_types[name]):
                        raise TypeError('Func {} argument {} must be {}'. \
                            format(func.__name__, name,
                                   need_param_types[name]))
            return func(*args, **kwargs)

        return wrapper

    return decorate
