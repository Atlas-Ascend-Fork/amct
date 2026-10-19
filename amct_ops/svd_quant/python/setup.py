# -*- coding: UTF-8 -*-
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

import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

source_files = glob.glob(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "svd_quant/csrc", "*.cpp"
    ),
    recursive=True,
)

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
exts = []
ext = NpuExtension(
    name="libsvd_quant",
    sources=source_files,
    extra_compile_args=[
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
)
exts.append(ext)

setup(
    name="amct",
    version='1.0',
    keywords='svd_quant',
    ext_modules=exts,
    package_data={
        'svd_quant': ['*.py', '*.so'],
    },
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)
