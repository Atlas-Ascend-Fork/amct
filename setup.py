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
print(f'current directort: {CUR_DIR}')

class SetupTool():
    """ tool for setup """
    def __init__(self):
        self.packages = setuptools.find_packages(include=['amct_pytorch*'])
        print(f'find packages: {self.packages}')
        self.set_version()
        self.set_platform()
        self.setup_args = dict()

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
        'Programming Language :: Python :: 3'
    ],
    author='Huawei Technologies Co., Ltd.',
    license='Apache 2.0',
    extras_require={
        "pytorch": ["2.1"]
    },
    package_data={
        '': ['.version'],
    },
    include_package_data=True,
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
