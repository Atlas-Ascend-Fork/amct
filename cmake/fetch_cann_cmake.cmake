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

        set(CANN_CMAKE_TAG "master-049")
        if(CANN_3RD_LIB_PATH AND EXISTS "${CANN_3RD_LIB_PATH}/cmake-${CANN_CMAKE_TAG}.tar.gz")
            FetchContent_Declare(
                cann-cmake
                URL "${CANN_3RD_LIB_PATH}/cmake-${CANN_CMAKE_TAG}.tar.gz"
                URL_HASH SHA256=19a3379a32007e279c4726d2c4b1e9a4eb8f7e187aa9f291e07c563ae3f5b321
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
        # overlay: master-049 protobuf.cmake 缓存复用缺少工具链兼容性检查,
        # 不兼容缓存(如 gcc-15 产物被 gcc-7.3 消费)会导致链接失败,
        # 用仓库内兼容改造版覆盖; 上游 cann-cmake 修复后可移除。
        if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/protobuf_compat.cmake")
            configure_file("${CMAKE_CURRENT_LIST_DIR}/protobuf_compat.cmake"
                "${cann-cmake_SOURCE_DIR}/third_party/protobuf.cmake" COPYONLY)
        endif()
        include("${cann-cmake_SOURCE_DIR}/function/prepare.cmake")
    endif()
endif()
