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

decorator util for check function params type

"""
from inspect import signature
from functools import wraps

__all__ = ['check_params']


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
