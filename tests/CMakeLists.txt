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
project(amct_test)
cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 17)

set(AMCT_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(BASE_DIR ${AMCT_TOP_DIR}/amct_pytorch)
set(AMCT_DIR ${AMCT_TOP_DIR})

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")

add_subdirectory(${AMCT_TOP_DIR}/amct_pytorch/classic/graph_based
                 ${CMAKE_BINARY_DIR}/amct_pytorch_classic_ptq_graph_based)

if(ENABLE_COVERAGE STREQUAL "TRUE")
    set(PYTEST_COV_ARGS --cov --cov-config=${AMCT_TOP_DIR}/pyproject.toml --cov-report=term-missing --cov-report=xml:${AMCT_TOP_DIR}/build/coverage.xml)
else()
    set(PYTEST_COV_ARGS "")
endif()

add_custom_target(amct_pytorch_utest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PYTEST_COV_ARGS}
    WORKING_DIRECTORY ${AMCT_TOP_DIR}
    COMMENT "run amct_pytorch python unit tests"
)
add_dependencies(amct_pytorch_utest amct_pytorch_proto_compile)

add_custom_target(amct_utest
    DEPENDS amct_pytorch_utest
    COMMENT "run all amct utest"
)
