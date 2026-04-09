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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import os
import sys
import setuptools
import torch
from torch.utils import cpp_extension


class BuildExtensionAcc(cpp_extension.BuildExtension):
    '''set parallel of BuildExtension to accelerate '''
    def __init__(self, *args, **kwargs):
        kwargs['use_ninja'] = False
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        '''rewrite build_extensions func'''
        num_jobs = int(os.environ.get('AMCT_NUM_BUILD_JOBS', '8'))
        if num_jobs > 1:
            self.parallel = num_jobs
        super().build_extensions()


def setup():
    extra_compile_args = ['-fopenmp', '-O3']
    extra_link_args = ['-fopenmp']
    
    ext_modules = [
        cpp_extension.CppExtension(
            name='hifloat8_cast',
            sources=['hifloat8_cast.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[('_GLIBCXX_USE_CXX11_ABI=0', None)]
        )
    ]
    
    setuptools.setup(
        name='hifloat8_cast',
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtensionAcc}
    )

if __name__ == '__main__':
    setup()
