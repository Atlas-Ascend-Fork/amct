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
set(graph_based_compression_PATH ${AMCT_TOP_DIR}/amct_pytorch/graph_based_compression)
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

# =============================compile quant_lib===============================
set(AMCT_TORCH_OP_DIR ${AMCT_TOP_DIR}/amct_pytorch/graph_based_compression/amct_pytorch/custom_op)
set(COMMON_ALGRITHEMN_CPU
    ${graph_based_compression_PATH}/common_cpp/quantize_lib/src/arq.cpp
    ${graph_based_compression_PATH}/common_cpp/quantize_lib/src/util.cpp
    ${graph_based_compression_PATH}/common_cpp/quantize_lib/src/dump.cpp
    ${graph_based_compression_PATH}/common_cpp/quantize_lib/src/cast_util.cpp
)

set_source_files_properties(
    ${COMMON_ALGRITHEMN_CPU}
    PROPERTIES COMPILE_FLAGS -fopenmp
)

# compile torch_op.so
add_library(quant_lib SHARED
    ${COMMON_ALGRITHEMN_CPU}
)

add_dependencies(quant_lib
    amct_pytorch_prepare_stub)

target_include_directories(quant_lib PRIVATE
    ${PYTORCH_OP_COMPILE_INCLUDE}
    ${graph_based_compression_PATH}/custom_op/inc
    ${graph_based_compression_PATH}/common_cpp/quantize_lib/inc
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
)

# cp to path
add_custom_target(amct_pytorch_lib_prepare
  DEPENDS quant_lib
  # copy to amct_pytorch
  COMMAND mv libquant_lib.so ${AMCT_TORCH_OP_DIR}/libquant_lib.so
  # strip
  COMMAND cd ${AMCT_TORCH_OP_DIR} && strip libquant_lib.so
)
# =============================compile ops=====================================
# ------------------------compile fake torch related .so-----------------------

message(STATUS "Python3_VERSION_MAJOR: ${Python3_VERSION_MAJOR}")
message(STATUS "Python3_VERSION_MINOR: ${Python3_VERSION_MINOR}")
set(OP_PY ${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR})
if(${OP_PY} STREQUAL 37)
    set(OP_PY 37m)
endif()
message(STATUS "Detected OP_PY: ${OP_PY}")

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

# loop to generate *so
set(PYTORCH_STUB_SOS c10 torch torch_cpu torch_python)
foreach(PYTORCH_STUB_SO ${PYTORCH_STUB_SOS})
  add_library(${PYTORCH_STUB_SO} SHARED
      ${graph_based_compression_PATH}/common_cpp/quantize_lib/src/util.cpp
  )
  target_include_directories(${PYTORCH_STUB_SO} PRIVATE
      ${graph_based_compression_PATH}/custom_op/inc/
      ${graph_based_compression_PATH}/common_cpp/quantize_lib/inc
  )
  target_compile_options(${PYTORCH_STUB_SO} PRIVATE
    -Wall -fstack-protector-strong
    -fno-common -fno-strict-aliasing -pipe
  )
  target_compile_definitions(${PYTORCH_STUB_SO} PRIVATE
      _GLIBCXX_USE_CXX11_ABI=0
  )
endforeach(${PYTORCH_STUB_SO})
# cp fakeso to a path
add_custom_target(amct_pytorch_prepare_stub
    DEPENDS c10 torch torch_cpu torch_python
    # install to python whl
    COMMAND cp libc10.so ${AMCT_TOP_DIR}/tmp/
    COMMAND cp libtorch.so ${AMCT_TOP_DIR}/tmp/
    COMMAND cp libtorch_cpu.so ${AMCT_TOP_DIR}/tmp/
    COMMAND cp libtorch_python.so ${AMCT_TOP_DIR}/tmp/
)

# -----------------------compile op's .so for torch1.5.0-----------------------
message(STATUS "Python 3 include directories: ${Python3_INCLUDE_DIRS}")
message(STATUS "PYTORCH_PATH include directories: ${PYTORCH_PATH}")
set(PYTORCH_OP_COMPILE_INCLUDE
    ${PYTORCH_PATH}/include
    ${PYTORCH_PATH}/include/torch/csrc/api/include
    ${PYTORCH_PATH}/include/TH
    ${PYTORCH_PATH}/include/THC
    ${Python3_INCLUDE_DIRS}
)

set(PYTORCH_OP_COMPILE_OPTIONS
    -pthread -Wno-unused-result -Wsign-compare -DNDEBUG
    -fwrapv -O3 -Wall -fPIC -fno-common -fno-strict-aliasing
    -pipe -fopenmp
)

set(PYTORCH_OP_COMPILE_DEFINITION
    TORCH_API_INCLUDE_EXTENSION_H
    _GLIBCXX_USE_CXX11_ABI=0
)

set(PYTORCH_OP_LINK_OPTIONS
    -Wl,-z,relro,-z,now
    -Wl,-z,noexecstack
    -pthread -L${AMCT_TORCH_OP_DIR} -L${TOP_DIR}/tmp/
)

set(PYTORCH_OP_LINK_LIBRARIES
    -Wl,--no-as-needed
    -lc10 -ltorch -ltorch_cpu -ltorch_python
    -Wl,--as-needed
)

# compile several op.so, please prefix is amct_pytorch_op
set(PYTORCH_OPS
    amct_pytorch_op_cast
    amct_pytorch_op_dump
)
message(INFO PYTORCH_OPS: ${PYTORCH_OPS})
foreach(PYTORCH_OP ${PYTORCH_OPS})
    # set src file for different op
    if ("${PYTORCH_OP}" STREQUAL "amct_pytorch_op_cast")
        set(OP_SRC_FILE 
          ${graph_based_compression_PATH}/custom_op/src/cast_op.cpp
          ${graph_based_compression_PATH}/custom_op/src/util_op.cpp)
    elseif ("${PYTORCH_OP}" STREQUAL "amct_pytorch_op_dump")
        set(OP_SRC_FILE ${graph_based_compression_PATH}/custom_op/src/dump_op.cpp)
    else()
        message(ERROR " please set src file for op")
    endif()

    add_library(${PYTORCH_OP} SHARED
        ${OP_SRC_FILE}
    )

    add_dependencies(${PYTORCH_OP}
        amct_pytorch_lib_prepare
        amct_pytorch_prepare_stub
    )

    target_include_directories(${PYTORCH_OP} PRIVATE
        ${PYTORCH_OP_COMPILE_INCLUDE}
        ${graph_based_compression_PATH}/custom_op/inc
        ${graph_based_compression_PATH}/common_cpp/quantize_lib/inc
    )

    target_compile_options(${PYTORCH_OP} PRIVATE
        ${PYTORCH_OP_COMPILE_OPTIONS}
    )

    target_compile_definitions(${PYTORCH_OP} PRIVATE
        ${PYTORCH_OP_COMPILE_DEFINITION}
        TORCH_EXTENSION_NAME=${PYTORCH_OP}
    )

    target_link_libraries(${PYTORCH_OP} PRIVATE
        ${PYTORCH_OP_LINK_LIBRARIES}
        -lquant_lib
    )

    target_link_options(${PYTORCH_OP} PRIVATE
        ${PYTORCH_OP_LINK_OPTIONS}
    )

endforeach(${PYTORCH_OP})

# packages
add_custom_target(amct_pytorch_ops
    DEPENDS amct_pytorch_op_dump amct_pytorch_op_cast
    # copy to amct_pytorch
    COMMAND mv libamct_pytorch_op_cast.so ${AMCT_TOP_DIR}/amct_pytorch_op_cast.cpython-${OP_PY}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu.so
    COMMAND mv libamct_pytorch_op_dump.so ${AMCT_TOP_DIR}/amct_pytorch_op_dump.cpython-${OP_PY}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu.so
)

add_custom_target(amct_pytorch_llt_lib
  DEPENDS amct_pytorch_ops amct_pytorch_proto_compile
  # strip
  COMMAND cd ${AMCT_TORCH_OP_DIR} && strip libquant_lib.so
  COMMAND cd ${AMCT_TOP_DIR} && strip amct_pytorch_op*.so
)

add_custom_target(amct_pytorch_python_utest
    DEPENDS amct_pytorch_llt_lib
    COMMAND PYTHONPATH=${AMCT_TOP_DIR}:${PYTHONPATH} coverage run -m unittest discover .
    WORKING_DIRECTORY ${TEST_FILES_DIR}
    COMMENT "run all pytorch python utest"
)

add_custom_target(amct_pytorch_utest
    DEPENDS amct_pytorch_python_utest
    COMMAND rm -rf ${AMCT_TOP_DIR}/amct_pytorch/graph_based_compression/amct_pytorch/proto/*_pb2.py
    COMMAND rm -rf ${AMCT_TOP_DIR}/amct_pytorch_op*.so
    COMMAND rm ${AMCT_TORCH_OP_DIR}/*.so
    COMMENT "run all amct pytorch utest"
)