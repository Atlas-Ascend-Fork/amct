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
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME SvdQuant
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        op_host/svd_quant_def.cpp
)

target_sources(optiling PRIVATE
        op_host/svd_quant_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
target_sources(opmaster_ct PRIVATE
        op_host/svd_quant_tiling.cpp
)
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)
