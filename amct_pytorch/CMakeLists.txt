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

# 开启debug
set(CMAKE_VERBOSE_MAKEFILE on)

# 定义常量
get_filename_component(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
get_filename_component(AMCT_DIR ${BASE_DIR}/.. ABSOLUTE)

set(AMCT_TORCH_PROTO_DIR ${BASE_DIR}/amct_pytorch_inner/amct_pytorch/proto)
set(AMCT_TORCH_CUSTOM_DIR ${BASE_DIR}/amct_pytorch_inner/amct_pytorch/custom_op)
set(AMCT_TORCH_BUILD_DIR ${BASE_DIR}/build)

add_subdirectory(amct_pytorch_inner)

message("-------pytorch 1-----")
set(COMMON_LIB_INC_FILES
  util.h dump.h cast_util.h error_codes.h
)
set(HI_PYTHON python3)
add_custom_target(amct_pytorch_package
  DEPENDS amct_pytorch_proto_compile amct_pytorch_prepare_package tensor_decompose_install_pytorch
  COMMAND cd ${BASE_DIR} && chmod 640 `find ./ -type f` && chmod 750 `find ./ -type d` && cd ${BASE_DIR}/.. &&
          export AMCT_PYTORCH_PLATFORM=linux_${CMAKE_HOST_SYSTEM_PROCESSOR} &&
          export BEP_CGTLIST="$BEP_CGTLIST ${HI_PYTHON}" &&
          ${HI_PYTHON} setup.py sdist --formats=gztar
  # delete common header files
  COMMAND cd ${BASE_DIR}/amct_pytorch_inner/custom_op/inc && rm ${COMMON_LIB_INC_FILES}
  COMMAND rm -rf ${AMCT_TORCH_CUSTOM_DIR}/*.so
  # delete tmp files
  COMMAND rm -rf ${AMCT_TORCH_BUILD_DIR}
  COMMAND rm -rf ${AMCT_TORCH_PROTO_DIR}/*_pb2.py
  COMMAND rm -rf ${BASE_DIR}/amct_pytorch_inner/lib/*.so
  COMMAND rm -rf ${BASE_DIR}/../amct_pytorch.egg-info
)

add_custom_target(amct_pytorch
  DEPENDS amct_pytorch_package
)

install(DIRECTORY ${BASE_DIR}/dist/ DESTINATION lib OPTIONAL)

# clear temp files
add_custom_target(amct_pytorch_clean
  COMMAND cd ${BASE_DIR} && rm -f *.tar.gz
  COMMAND cd ${BASE_DIR} && rm -rf amct_pytorch.egg-info && rm -rf dist && rm -rf build
  COMMAND cd ${BASE_DIR} && rm -f amct_pytorch/proto/*_pb2.py
  COMMAND cd ${BASE_DIR} && rm -r amct_pytorch/custom_op/dump/*.so
  COMMAND cd ${BASE_DIR} && rm -r amct_pytorch/custom_op/cast/*.so
)
