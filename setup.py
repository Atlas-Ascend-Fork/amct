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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import tarfile
import setuptools
tarfile.TarFile.format = tarfile.GNU_FORMAT

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
OP_TMP_DIR = 'amct_pytorch/graph_based_compression/custom_op/src/tmp'
AMCT_LIB_DIR = os.path.join(CUR_DIR, "./amct_pytorch/graph_based_compression/amct_pytorch/custom_op")
UTIL_OP_PATH = 'amct_pytorch/graph_based_compression/custom_op/src/util_op.cpp'
SRC_FILES = [
             ['amct_pytorch/graph_based_compression/custom_op/src/cast_op.cpp',
              UTIL_OP_PATH],
             ['amct_pytorch/graph_based_compression/custom_op/src/dump_op.cpp'], ]
os.environ['SOURCE_DATE_EPOCH'] = \
    str(int(os.path.getctime(os.path.realpath(__file__))))


class SetupTool(): # pylint: disable=R0903
    """ tool for setup"""
    def __init__(self):
        self.packages = setuptools.find_packages()
        self.packages.extend(['amct_pytorch/graph_based_compression/custom_op'])
        self.set_version()
        self.set_platform()
        self.setup_args = dict()
        self.add_ops_modules()

    @classmethod
    def get_modules_set(cls):
        '''get setting for ext_modules'''
        op_tmp_dir = OP_TMP_DIR
        print("CUR_DIR:{CUR_DIR}")

        compile_files = []
        if not os.path.exists(os.path.realpath(op_tmp_dir)):
            os.makedirs(os.path.realpath(op_tmp_dir))
        for task in SRC_FILES:
            compile_task = []
            op_path = os.path.split(task[0])[-1]
            op_name = os.path.splitext(op_path)[0]
            op_file_tmp = os.path.join(op_tmp_dir, op_name)
            if not os.path.exists(os.path.realpath(op_file_tmp)):
                os.mkdir(os.path.realpath(op_file_tmp))
            for src_file in task:
                file_name = os.path.split(src_file)[-1]
                dst_file = os.path.join(op_file_tmp, file_name)
                shutil.copy(src_file, op_file_tmp)
                compile_task.append(dst_file)
            compile_files.append(compile_task)

        prefix = 'amct_pytorch_op'
        
        module_name = ['{}_cast'.format(prefix),
                       '{}_dump'.format(prefix), ]

        modules_set = zip(module_name, compile_files)
        return modules_set

    def set_version(self):
        """ set version"""
        version_file = os.path.join(CUR_DIR, 'amct_pytorch', '.version')
        with open(version_file) as fid:
            version = fid.readlines()[0].strip()
            self.version = version

    def set_platform(self):
        """ set platform"""
        if 'sdist' in sys.argv:
            platform = os.getenv('AMCT_PYTORCH_PLATFORM').replace("\n", "")
            self.platform = platform

    def add_ops_modules(self):
        """ set ext_modules for compile amct_pytorch_ops"""
        if 'sdist' in sys.argv:
            return
        # add ext_modules if not setup in source
        import platform # pylint: disable=C0415
        from torch.utils import cpp_extension # pylint: disable=E0401, C0415

        class BuildExtensionAcc(cpp_extension.BuildExtension):
            '''set parallel of BuildExtension to accelerate '''
            def __init__(self, *args, **kwargs):
                """ rewrite init """
                kwargs['use_ninja'] = False
                super().__init__(*args, **kwargs)

            def build_extensions(self):
                '''rewrite build_extensions func'''
                num_jobs = int(os.environ.get('AMCT_NUM_BUILD_JOBS', '8'))
                if num_jobs > 1:
                    self.parallel = num_jobs # pylint: disable=W0201
                super().build_extensions()

        modules_set = self.get_modules_set()
        extra_compile_args = []
        extra_compile_args.append('-fopenmp')
        ext_modules = [
            cpp_extension.CppExtension(
                # name same with name of amct_ops.py
                name=name,
                sources=src_file,
                include_dirs=[os.path.join(CUR_DIR, 'amct_pytorch/graph_based_compression/custom_op/inc/')],
                extra_compile_args=extra_compile_args,
                libraries=['quant_lib'],
                library_dirs=[AMCT_LIB_DIR]) for name, src_file in modules_set
                ]
        self.setup_args['ext_modules'] = ext_modules
        self.setup_args['cmdclass'] = {
            'build_ext': BuildExtensionAcc
        }

setup_tools = SetupTool() # pylint: disable=C0103

setuptools.setup(
    name='amct_pytorch',
    version=setup_tools.version,
    description='Ascend Model Compression Toolkit for PyTorch',
    url='',
    packages=setup_tools.packages,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3'
    ],
    author='Huawei Technologies Co., Ltd.',
    license='Apache 2.0',
    extras_require={
        "pytorch": ["1.5"]
    },
    package_data={
        '': ['.version'],
        'amct_pytorch.graph_based_compression': ['amct_pytorch/proto/*.proto',
                         'amct_pytorch/common/proto/*.proto',
                         'amct_pytorch/capacity/*.csv',
                         'lib/*.so',
                         'amct_pytorch/custom_op/*.so'],
        'amct_pytorch/graph_based_compression/custom_op': ['inc/*.h', 'src/*.cpp']
    },
    zip_safe=False,
    **setup_tools.setup_args
)

if 'sdist' in sys.argv:
    shutil.move(
        os.path.join(
            CUR_DIR,
            'dist/amct_pytorch-{}.tar.gz'.format(setup_tools.version)),
        os.path.join(
            CUR_DIR,
            'dist/amct_pytorch-{}-py3-none-{}.tar.gz'.format(
                setup_tools.version, setup_tools.platform)))
else:
    shutil.rmtree(os.path.realpath(OP_TMP_DIR))
