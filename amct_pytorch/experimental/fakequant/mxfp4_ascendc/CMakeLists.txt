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
#
# Experimental MXFP4 Ascend-C operator (kept under amct_pytorch/experimental).
# Layout mirrors amct_ops/hifloat8_cast: op_kernel / op_extension / python.
# Build still uses ascendc.cmake (legacy ACLRT_LAUNCH_KERNEL path) rather than
# the ASC language toolchain used by amct_ops.

cmake_minimum_required(VERSION 3.16)
project(mxfp4_ascendc)

set(SOC_VERSION "Ascend910B3" CACHE STRING "SoC version")
set(RUN_MODE "npu" CACHE STRING "npu / cpu / sim")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "CANN toolkit path")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# ==================================================================
# 1.  Device kernel  (ccec via ascendc.cmake -> libascendc_*_npu.so)
# ==================================================================
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR
        ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR
        ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under CANN package")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels_${RUN_MODE} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mxfp4_kernel.cpp)

# Device compile unit must see op_kernel/mxfp4_tiling.h
if(TARGET ascendc_kernels_${RUN_MODE})
    target_include_directories(ascendc_kernels_${RUN_MODE} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
endif()

# ==================================================================
# 2.  Torch extension  (libmxfp4_ops.so) — TORCH_LIBRARY + host launch
# ==================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'include'))"
    OUTPUT_VARIABLE TORCH_NPU_INC OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_INC_RET)

if(NOT TORCH_NPU_INC_RET EQUAL 0)
    message(FATAL_ERROR
        "Failed to probe torch_npu include path (python exit code "
        "${TORCH_NPU_INC_RET}). Ensure torch_npu is importable by "
        "${Python3_EXECUTABLE} before building.")
endif()
if(TORCH_NPU_INC STREQUAL "")
    message(FATAL_ERROR
        "torch_npu include path probe returned empty. Ensure torch_npu is "
        "installed and importable by ${Python3_EXECUTABLE}.")
endif()
if(NOT EXISTS "${TORCH_NPU_INC}")
    message(FATAL_ERROR
        "torch_npu include directory does not exist: '${TORCH_NPU_INC}'.")
endif()

# CANN ships arch-specific headers/libs under <arch>-linux.
if(NOT DEFINED ARCH OR ARCH STREQUAL "")
    set(ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()

set(ASCEND_INC ${ASCEND_CANN_PACKAGE_PATH}/include)
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/include)
    list(APPEND ASCEND_INC ${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/include)
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/lib64)
    set(ASCEND_LIB ${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/lib64)
else()
    set(ASCEND_LIB ${ASCEND_CANN_PACKAGE_PATH}/lib64)
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_ABI_RET)

if(NOT TORCH_ABI_RET EQUAL 0)
    message(FATAL_ERROR
        "Failed to probe torch._C._GLIBCXX_USE_CXX11_ABI (python exit code "
        "${TORCH_ABI_RET}). Ensure PyTorch is importable by "
        "${Python3_EXECUTABLE} before building.")
endif()
if(NOT (TORCH_CXX11_ABI STREQUAL "0" OR TORCH_CXX11_ABI STREQUAL "1"))
    message(FATAL_ERROR
        "Unexpected torch._C._GLIBCXX_USE_CXX11_ABI value '${TORCH_CXX11_ABI}' "
        "(expected 0 or 1). Refusing to build with an ambiguous ABI setting.")
endif()

add_library(mxfp4_ops SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension/mxfp4_torch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension/register.cpp)

set_target_properties(mxfp4_ops PROPERTIES
    OUTPUT_NAME "mxfp4_ops"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_include_directories(mxfp4_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${ASCEND_INC}
    ${TORCH_NPU_INC}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR})

target_compile_options(mxfp4_ops PRIVATE
    -O2 -std=c++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

target_link_directories(mxfp4_ops PRIVATE ${ASCEND_LIB})
target_link_libraries(mxfp4_ops PRIVATE
    ${TORCH_LIBRARIES}
    ascendc_kernels_${RUN_MODE}
    ascendcl)

message(STATUS "Building libmxfp4_ops.so for SOC_VERSION=${SOC_VERSION}, ARCH=${ARCH}")
