# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------
set(AMCT_PYTORCH_TMP_DIR ${AMCT_TORCH_OP_DIR}/tmp)
get_filename_component(AMCT_PYTORCH_TMP_DIR ${AMCT_PYTORCH_TMP_DIR} ABSOLUTE)
add_custom_target(amct_pytorch_mkdir_tmp)
add_custom_command(TARGET amct_pytorch_mkdir_tmp
  COMMAND rm -rf ${AMCT_PYTORCH_TMP_DIR}
  COMMAND mkdir -p -m 750 ${AMCT_PYTORCH_TMP_DIR}
)

set(COMMON_ALGRITHEMN_CPU
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_cpp/quantize_lib/src/nuq.cpp
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
  COMMAND cp libquant_lib.so ${AMCT_TORCH_OP_DIR}/libquant_lib.so
  # strip
  COMMAND cd ${AMCT_TORCH_OP_DIR} && strip libquant_lib.so
)
