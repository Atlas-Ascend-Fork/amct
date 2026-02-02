# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
set(AMCT_PYTORCH_PATH ${AMCT_TOP_DIR}/amct_pytorch)
set(TEST_FILES_DIR ${AMCT_TOP_DIR}/tests/amct_pytorch)

# compile proto
set(PROTOC_PROGRAM ${AMCT_TOP_DIR}/build/protobuf_host/bin/protoc)
set(AMCT_TORCH_PROTO_DIR ${AMCT_TOP_DIR}/amct_pytorch/amct_pytorch_inner/amct_pytorch/proto)
add_custom_target(amct_pytorch_proto_compile
  COMMAND cd ${AMCT_TOP_DIR} && ${PROTOC_PROGRAM} -I=./ -I=./amct_pytorch/amct_pytorch_inner/amct_pytorch/proto/ --python_out=./ amct_pytorch/amct_pytorch_inner/amct_pytorch/proto/basic_info.proto
  COMMAND cd ${AMCT_TOP_DIR} && ${PROTOC_PROGRAM} -I=./ -I=./amct_pytorch/amct_pytorch_inner/amct_pytorch/proto/ --python_out=./ amct_pytorch/amct_pytorch_inner/amct_pytorch/proto/calibration_config_pytorch.proto
  COMMAND cd ${AMCT_TOP_DIR} && ${PROTOC_PROGRAM} -I=./ -I=./amct_pytorch/amct_pytorch_inner/amct_pytorch/proto/ --python_out=./ amct_pytorch/amct_pytorch_inner/amct_pytorch/proto/scale_offset_record_pytorch.proto
  COMMAND cd ${AMCT_TOP_DIR} && ${PROTOC_PROGRAM} -I=./ -I=./amct_pytorch/amct_pytorch_inner/amct_pytorch/proto/ --python_out=./ amct_pytorch/amct_pytorch_inner/amct_pytorch/proto/retrain_config_pytorch.proto
  COMMAND cd ${AMCT_TORCH_PROTO_DIR} && ${PROTOC_PROGRAM} --python_out=./ distill_config_pytorch.proto
  COMMAND cd ${AMCT_TORCH_PROTO_DIR} && ${PROTOC_PROGRAM} --python_out=./ quant_calibration_config_pytorch.proto
)
if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    add_dependencies(amct_pytorch_proto_compile protobuf_host_build ascend_protobuf_static)
endif()

add_custom_target(amct_pytorch_utest
    DEPENDS amct_pytorch_proto_compile
    COMMAND PYTHONPATH=${AMCT_PYTORCH_PATH}:${PYTHONPATH} coverage run -m unittest discover .
    WORKING_DIRECTORY ${TEST_FILES_DIR}
    COMMENT "run all pytorch python utest"
)