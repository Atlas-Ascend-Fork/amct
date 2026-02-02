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
project(amct VERSION 1.0.0)
include(CMakePrintHelpers)

message("CMAKE_SOURCE_DIR === ${CMAKE_SOURCE_DIR}")
message("PROJECT_SOURCE_DIR === ${PROJECT_SOURCE_DIR}")

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    find_program(SYSTEM_CCACHE_PROGRAM ccache)
    if (SYSTEM_CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER ${SYSTEM_CCACHE_PROGRAM} CACHE PATH "cache Compiler")
        set(CMAKE_CXX_COMPILER_LAUNCHER ${SYSTEM_CCACHE_PROGRAM} CACHE PATH "cache Compiler")
    endif()
    
    set(CMAKE_C_FLAGS_RELEASE "")
    set(CMAKE_CXX_FLAGS_RELEASE "")
endif()

include(cmake/config.cmake)

cmake_print_variables(VERSION)
if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/build/third_party)
endif()
if(NOT CANN_3RD_PKG_PATH)
  set(CANN_3RD_PKG_PATH ${PROJECT_SOURCE_DIR}/build/third_party/pkg)
endif()

set(OPEN_SOURCE_DIR "${CANN_3RD_LIB_PATH}")
message("OPEN_SOURCE_DIR === ${OPEN_SOURCE_DIR}")
message("CANN_3RD_LIB_PATH === ${CANN_3RD_LIB_PATH}")
message("CANN_3RD_PKG_PATH === ${CANN_3RD_PKG_PATH}")

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(STATUS "Build AMCT with dependency cann pkgs installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build AMCT with dependency cann pkgs installed" TRUE)
else()
    message(STATUS "Build AMCT with cann source exist")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build AMCT need cann source code" FALSE)
endif()

if(EXISTS ${OPEN_SOURCE_DIR})
    # 依赖opensource
    message(STATUS "Build amct with open source third party")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG_COMMUNITY "Build amct with open source third party" FALSE)
else()
    # 额外下载安装第三方库
    message(STATUS "Build amct with out open source third party")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG_COMMUNITY "Build amct with out open source third party" TRUE)
endif()

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    include(./cmake/third_party/protobuf.cmake)
endif()

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND ENABLE_TEST)
    add_subdirectory(tests)
else()
    add_subdirectory(amct_pytorch)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
        add_custom_target(default_target ALL
            COMMAND echo "This runs by default"
            DEPENDS amct_pytorch
    )
endif()

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
add_dependencies(default_target protobuf_host_build ascend_protobuf_static)
endif()
endif()