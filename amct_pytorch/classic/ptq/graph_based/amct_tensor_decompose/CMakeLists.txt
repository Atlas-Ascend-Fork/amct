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
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(AMCT_TENSOR_DECOMPOSE)

# 开启debug
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_SKIP_RPATH TRUE)

message("========== tensor_decompose")
message(STATUS ${TOP_DIR})
message(STATUS "${TOP_DIR}/../../abl/libc_sec")
if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  set(AMCT_HW_SECURE_LIB_DIR ${TOP_DIR}/../../abl/libc_sec)
else()
  set(AMCT_HW_SECURE_LIB_DIR ${TOP_DIR}/abl/libc_sec)
endif()

# 定义常量
set(AMCT_PYTORCH_LIB_DIR ${BASE_DIR}/graph_based_compression/lib)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fstack-protector-strong -O3 -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fstack-protector-strong -D_FORTIFY_SOURCE=2 -O2 -std=c99")
# 设置链接选项
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,relro,-z,now -Wl,-z,noexecstack")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${AMCT_HW_SECURE_LIB_DIR}/include
  ${ASCEND_HOME_PATH}/${PREX}/include/
)

add_library(tensor_decompose SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_decomposition.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_decomposition_api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vector.cpp
)

target_include_directories(tensor_decompose PRIVATE
  ${ASCEND_HOME_PATH}/${PREX}/include/
  ${AMCT_ACL_EXTERNAL_INCLUDE}
)

target_link_directories(tensor_decompose PRIVATE
  ${ASCEND_HOME_PATH}/${PREX}/lib64/
)

target_link_libraries(tensor_decompose PRIVATE c_sec)

add_custom_target(tensor_decompose_install_pytorch
  DEPENDS tensor_decompose
)

add_custom_command(TARGET tensor_decompose_install_pytorch
  COMMAND mkdir -p ${AMCT_PYTORCH_LIB_DIR}
  COMMAND cp ${PROJECT_BINARY_DIR}/libtensor_decompose.so ${AMCT_PYTORCH_LIB_DIR}
  COMMAND cd ${AMCT_PYTORCH_LIB_DIR} && strip libtensor_decompose.so
)
