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

"""
统一 NPU 算子 wheel 构建配置

构建脚本（build_wheel.sh）在调用前，会先将各算子的 Python 包
和编译产物（.so）统一汇集到 staging/ 目录，因此这里使用标准
的 find_packages() 即可。

已包含算子：
  hifloat8_cast  →  python 包名: amct_ops.hifloat8_cast
"""

import os
from setuptools import setup, find_packages
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """强制生成平台相关 wheel（包含编译 .so）"""
    def has_ext_modules(self):
        return True


STAGING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'staging')

# 收集 staging/amct_ops/<sub>/ 下的 .so 文件作为 package_data
package_data = {}
amct_ops_dir = os.path.join(STAGING, 'amct_ops')
if os.path.isdir(amct_ops_dir):
    for sub in os.listdir(amct_ops_dir):
        sub_path = os.path.join(amct_ops_dir, sub)
        if not os.path.isdir(sub_path) or sub.startswith(('.', '_')):
            continue
        so_files = [f for f in os.listdir(sub_path) if f.endswith('.so')]
        if so_files:
            package_data[f'amct_ops.{sub}'] = so_files

setup(
    name="amct_ops",
    version="1.0.0",
    description="NPU custom operators for Ascend NPU (HiFloat8, ...)",
    packages=find_packages(where='staging'),
    package_dir={'': 'staging'},
    package_data=package_data,
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torch_npu",
    ],
    distclass=BinaryDistribution,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)
