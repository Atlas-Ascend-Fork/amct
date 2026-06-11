# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
                    check_param_types(name, value, need_param_types, func)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def check_param_types(name, value, need_param_types, func):
    '''check the params based on whether they are instance'''
    if isinstance(value, type):
        if not issubclass(value, need_param_types[name]):
            raise TypeError('Func {} argument {} must be {}'. \
                format(func.__name__, name,
                    need_param_types[name]))
    else:
        if not isinstance(value, need_param_types[name]):
            raise TypeError('Func {} argument {} must be {}'. \
                format(func.__name__, name,
                    need_param_types[name]))


def check_parameters_in_schema(func, *param_names):
    '''
    检查指定参数是否存在于函数的第一个schema中
    
    args:
        func: 要检查的函数，必须包含非空的_schemas属性
        *param_names: 参数名称，可传入多个字符串或一个list/tuple
    '''
    if len(param_names) == 1 and isinstance(param_names[0], (list, tuple)):
        param_names = param_names[0]
    schemas_dict = func._schemas
    schema = next(iter(schemas_dict.values()))
    schema_param_names = [arg.name for arg in schema.arguments]

    for param_name in param_names:
        if param_name not in schema_param_names:
            return False
        
    return True