#!/bin/bash
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

# Build script for hifloat8_cast.cpp

echo "Building hifloat8_cast extension..."

# Set build jobs
export AMCT_NUM_BUILD_JOBS=${AMCT_NUM_BUILD_JOBS:-8}

# Build the extension
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "The compiled module will be available as: hifloat8_cast"
else
    echo "Build failed!"
    exit 1
fi
