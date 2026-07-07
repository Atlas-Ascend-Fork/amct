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

cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# Target NPU architecture — pass -DNPU_ARCH=<value> to select platform:
#   dav-2201  Ascend A2 (ascend910b,   910B1/B2/B3, UB 256 KB)  [default]
#             Ascend A3 (ascend910_93, 910_93,       UB 512 KB)  共用同一 ISA
#   dav-3510  Ascend A5 (ascend950,                 UB 512 KB+)
#
# A2/A3 编译产物相同；UB 差异由运行时 GetCoreMemSize() 决定 tileLength。
# ---------------------------------------------------------------------------
set(NPU_ARCH "dav-2201" CACHE STRING "Target NPU architecture")
set_property(CACHE NPU_ARCH PROPERTY STRINGS "dav-2201" "dav-3510")

if(NOT NPU_ARCH MATCHES "^dav-(2201|3510)$")
    message(FATAL_ERROR
        "Unsupported NPU_ARCH='${NPU_ARCH}'. "
        "Supported values: dav-2201 (A2/A3), dav-3510 (A5)")
endif()

find_package(ASC REQUIRED)

project(hifloat8_cast LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set. Please source CANN set_env.sh before building.")
endif()

set(ASCEND_ARCH_DIR "" CACHE STRING "CANN host architecture directory, e.g. x86_64-linux or aarch64-linux")
if(NOT ASCEND_ARCH_DIR)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" HOST_PROCESSOR)
    if(HOST_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(ASCEND_ARCH_DIR "x86_64-linux")
    elseif(HOST_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(ASCEND_ARCH_DIR "aarch64-linux")
    else()
        message(FATAL_ERROR
            "Unsupported host CPU architecture='${CMAKE_SYSTEM_PROCESSOR}'. "
            "Please pass -DASCEND_ARCH_DIR=x86_64-linux or -DASCEND_ARCH_DIR=aarch64-linux.")
    endif()
endif()

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
set(ACL_INCLUDE_DIR "${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/include")
set(ASC_INCLUDE_DIR "${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/asc/include")
set(ACL_LIB_DIR "${ASCEND_HOME_PATH}/lib64")
set(ASCEND_ARCH_LIB_DIR "${ASCEND_HOME_PATH}/${ASCEND_ARCH_DIR}/lib64")

foreach(REQUIRED_CANN_DIR
    "${ACL_INCLUDE_DIR}"
    "${ASC_INCLUDE_DIR}"
    "${ASCEND_ARCH_LIB_DIR}"
)
    if(NOT IS_DIRECTORY "${REQUIRED_CANN_DIR}")
        message(FATAL_ERROR
            "Required CANN directory does not exist: ${REQUIRED_CANN_DIR}. "
            "Please install the ${ASCEND_ARCH_DIR} CANN package or pass the correct -DASCEND_ARCH_DIR.")
    endif()
endforeach()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include ${TORCH_NPU_PATH}/csrc)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

set_source_files_properties(op_kernel/hifloat8_cast_kernel.cpp PROPERTIES LANGUAGE ASC)

add_library(hifloat8_cast_ops SHARED
    op_kernel/hifloat8_cast_kernel.cpp
    op_extension/hifloat8_cast_torch.cpp
    op_extension/register.cpp
)

target_include_directories(hifloat8_cast_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${ACL_INCLUDE_DIR}
    ${ASC_INCLUDE_DIR}
)

target_link_libraries(hifloat8_cast_ops PRIVATE
    torch_npu
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
    ascendcl
    ascendc_runtime
)

target_link_directories(hifloat8_cast_ops PRIVATE
    ${ACL_LIB_DIR}
    ${ASCEND_ARCH_LIB_DIR}
    ${TORCH_NPU_LIBRARIES}
)

target_compile_options(hifloat8_cast_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)

message(STATUS "Building libhifloat8_cast_ops.so for ASCEND_ARCH_DIR=${ASCEND_ARCH_DIR}, NPU_ARCH=${NPU_ARCH}")
