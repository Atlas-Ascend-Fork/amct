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
set(AMCT_PYTORCH_TMP_DIR ${AMCT_TORCH_OP_DIR}/tmp)
get_filename_component(AMCT_PYTORCH_TMP_DIR ${AMCT_PYTORCH_TMP_DIR} ABSOLUTE)
add_custom_target(amct_pytorch_mkdir_tmp)
add_custom_command(TARGET amct_pytorch_mkdir_tmp
  COMMAND rm -rf ${AMCT_PYTORCH_TMP_DIR}
  COMMAND mkdir -p -m 750 ${AMCT_PYTORCH_TMP_DIR}
)

set(COMMON_ALGRITHEMN_CPU
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_cpp/quantize_lib/src/arq.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_cpp/quantize_lib/src/util.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_cpp/quantize_lib/src/cast_util.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_cpp/quantize_lib/src/dump.cpp
)

set_source_files_properties(
  ${COMMON_ALGRITHEMN_CPU}
  PROPERTIES COMPILE_FLAGS "-fopenmp -O3"
)

# compile torch_op.so
add_library(quant_lib SHARED
  ${COMMON_ALGRITHEMN_CPU}
)

add_dependencies(quant_lib amct_pytorch_mkdir_tmp)

target_include_directories(quant_lib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_cpp/quantize_lib/inc
)


target_compile_definitions(quant_lib PRIVATE
  _GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(quant_lib PRIVATE
  -Wl,-Bsymbolic -rdynamic -Wl,--no-undefined
)

target_link_options(quant_lib PRIVATE
  -Wl,-z,relro,-z,now
  -Wl,-z,noexecstack
  -pthread -fopenmp
  -fstack-protector-strong
)
# packages
add_custom_target(amct_pytorch_lib
  DEPENDS quant_lib
  # copy to amct_pytorch
  COMMAND mv libquant_lib.so ${AMCT_TORCH_OP_DIR}/libquant_lib.so
  # strip
  COMMAND cd ${AMCT_TORCH_OP_DIR} && strip libquant_lib.so
)
