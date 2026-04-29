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

set(TEST_FILES_DIR ${AMCT_TOP_DIR}/tests/amct_pytorch)

# compile proto
set(CMAKE_VERBOSE_MAKEFILE on)
set(PROTOC_PROGRAM ${AMCT_TOP_DIR}/build/protobuf_host/bin/protoc)
set(AMCT_TORCH_PROTO_DIR ${AMCT_TOP_DIR}/amct_pytorch/graph_based_compression/amct_pytorch/proto)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch;print(torch.__path__[0])"  OUTPUT_VARIABLE TORCH_PATH)
# 删除path中的换行符
string(STRIP "${TORCH_PATH}" PYTORCH_PATH)

add_custom_target(amct_pytorch_proto_compile
  DEPENDS protobuf_host_build ascend_protobuf_static
  COMMAND cd ${AMCT_TOP_DIR} && ${PROTOC_PROGRAM} -I=./ -I=./amct_pytorch/graph_based_compression/amct_pytorch/proto/ --python_out=./ amct_pytorch/graph_based_compression/amct_pytorch/proto/basic_info.proto
  COMMAND cd ${AMCT_TOP_DIR} && ${PROTOC_PROGRAM} -I=./ -I=./amct_pytorch/graph_based_compression/amct_pytorch/proto/ --python_out=./ amct_pytorch/graph_based_compression/amct_pytorch/proto/calibration_config_pytorch.proto
  COMMAND cd ${AMCT_TOP_DIR} && ${PROTOC_PROGRAM} -I=./ -I=./amct_pytorch/graph_based_compression/amct_pytorch/proto/ --python_out=./ amct_pytorch/graph_based_compression/amct_pytorch/proto/scale_offset_record_pytorch.proto
  COMMAND cd ${AMCT_TOP_DIR} && ${PROTOC_PROGRAM} -I=./ -I=./amct_pytorch/graph_based_compression/amct_pytorch/proto/ --python_out=./ amct_pytorch/graph_based_compression/amct_pytorch/proto/retrain_config_pytorch.proto
  COMMAND cd ${AMCT_TORCH_PROTO_DIR} && ${PROTOC_PROGRAM} --python_out=./ distill_config_pytorch.proto
  COMMAND cd ${AMCT_TORCH_PROTO_DIR} && ${PROTOC_PROGRAM} --python_out=./ quant_calibration_config_pytorch.proto
)

message(STATUS "Python3_VERSION_MAJOR: ${Python3_VERSION_MAJOR}")
message(STATUS "Python3_VERSION_MINOR: ${Python3_VERSION_MINOR}")
if(Python3_VERSION_MAJOR EQUAL 3)
    if(Python3_VERSION_MINOR VERSION_LESS 5)
        # Python 3.0-3.4: C++11
        set(CMAKE_CXX_STANDARD 11)
        message(STATUS "Python ${Python3_VERSION}: Using C++11")
    elseif(Python3_VERSION_MINOR VERSION_LESS 8)
        # Python 3.5-3.7: C++14
        set(CMAKE_CXX_STANDARD 14)
        message(STATUS "Python ${Python3_VERSION}: Using C++14")
    else()
        # Python 3.8+: C++17
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "Python ${Python3_VERSION}: Using C++17")
    endif()
else()
    # Python 2.x: C++11
    set(CMAKE_CXX_STANDARD 11)
    message(STATUS "Python ${Python3_VERSION}: Using C++11")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_CXX_STANDARD EQUAL 17)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    elseif(CMAKE_CXX_STANDARD EQUAL 14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif()
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_CXX_STANDARD EQUAL 17)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    elseif(CMAKE_CXX_STANDARD EQUAL 14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif()
endif()

add_custom_target(amct_pytorch_llt_lib
  DEPENDS amct_pytorch_proto_compile
)

add_custom_target(amct_pytorch_python_utest
    DEPENDS amct_pytorch_llt_lib
    COMMAND PYTHONPATH=${AMCT_TOP_DIR}:${PYTHONPATH} ${Python3_EXECUTABLE} -m coverage run -m unittest discover .
    WORKING_DIRECTORY ${TEST_FILES_DIR}
    COMMENT "run all pytorch python utest"
)

add_custom_target(amct_pytorch_utest
    DEPENDS amct_pytorch_python_utest
    COMMAND rm -rf ${AMCT_TOP_DIR}/amct_pytorch/graph_based_compression/amct_pytorch/proto/*_pb2.py
    COMMENT "run all amct pytorch utest"
)
