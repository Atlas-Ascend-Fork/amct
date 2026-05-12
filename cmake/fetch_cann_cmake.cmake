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
if(NOT PROJECT_SOURCE_DIR)
    if(CANN_3RD_LIB_PATH AND IS_DIRECTORY "${CANN_3RD_LIB_PATH}/cann-cmake")
        include("${CANN_3RD_LIB_PATH}/cann-cmake/function/prepare.cmake")
    else()
        include(FetchContent)

        set(CANN_CMAKE_TAG "master-002")
        if(CANN_3RD_LIB_PATH AND EXISTS "${CANN_3RD_LIB_PATH}/cmake-${CANN_CMAKE_TAG}.tar.gz")
            FetchContent_Declare(
                cann-cmake
                URL "${CANN_3RD_LIB_PATH}/cmake-${CANN_CMAKE_TAG}.tar.gz"
                URL_HASH SHA256=115474655951d9ba6c2d82200e0cee48186417cb3f512f7e3974a468bb420230
            )
        else()
            FetchContent_Declare(
                cann-cmake
                GIT_REPOSITORY https://gitcode.com/cann/cmake.git
                GIT_TAG        ${CANN_CMAKE_TAG}
                GIT_SHALLOW    TRUE
            )
        endif()
        FetchContent_GetProperties(cann-cmake)
        if(NOT cann-cmake_POPULATED)
            FetchContent_Populate(cann-cmake)
        endif()
        include("${cann-cmake_SOURCE_DIR}/function/prepare.cmake")
    endif()
endif()
