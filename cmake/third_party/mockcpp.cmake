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

set(open_source_target_name mockcpp)

if (CMAKE_HOST_SYSTEM_PROCESSOR  STREQUAL "aarch64")
    set(mockcpp_CXXFLAGS "-fPIC")
else()
    set(mockcpp_CXXFLAGS "-fPIC -std=c++11")
endif()
set(mockcpp_FLAGS "-fPIC")
set(mockcpp_LINKER_FLAGS "")

if ((NOT DEFINED ABI_ZERO) OR (ABI_ZERO STREQUAL ""))
    set(ABI_ZERO "true")
endif()


if (ABI_ZERO STREQUAL true)
    set(mockcpp_CXXFLAGS "${mockcpp_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(mockcpp_FLAGS "${mockcpp_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(BUILD_TYPE "DEBUG")

if (CMAKE_GENERATOR MATCHES "Unix Makefiles")
    set(IS_MAKE True)
    set(MAKE_CMD "$(MAKE)")
else()
    set(IS_MAKE False)
endif()

#依赖蓝区二进制仓mockcpp
set(MOCKCPP_SRC_DIR ${OPEN_SOURCE_DIR}/mockcpp_src)
set(MOCKCPP_DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/downloads/mockcpp-2.7)
set(URL_FILE ${MOCKCPP_DOWNLOAD_DIR}/mockcpp-2.7.tar.gz)
set(BOOST_INCLUDE_DIRS ${OPEN_SOURCE_DIR}/boost-1.87.0)

message(STATUS "mock cmake install prefix ${CMAKE_INSTALL_PREFIX}")
if (NOT EXISTS "${MOCKCPP_DOWNLOAD_DIR}/mockcpp-2.7.tar.gz")
    set(PATCH_FILE ${OPEN_SOURCE_DIR}/mockcpp-2.7/mockcpp-2.7_py3test.patch)
    message(STATUS, "mockcpp patch not use cache.")
    file(DOWNLOAD
        "https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h2/mockcpp-2.7_py3.patch"
        ${PATCH_FILE}
        TIMEOUT 60
    )
    include(ExternalProject)
    message(STATUS, "CMAKE_COMMAND is ${CMAKE_COMMAND}")
    set(URL_FILE "https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h2/mockcpp-2.7.tar.gz")
    message("mockcpp not use cache, new url file: ${URL_FILE}")
    ExternalProject_Add(mockcpp
        URL ${URL_FILE}
        DOWNLOAD_DIR ${MOCKCPP_DOWNLOAD_DIR}
        SOURCE_DIR ${MOCKCPP_SRC_DIR}
        TLS_VERIFY OFF
        PATCH_COMMAND git init && git apply ${PATCH_FILE}

        CONFIGURE_COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR}
            -DCMAKE_CXX_FLAGS=${mockcpp_CXXFLAGS}
            -DCMAKE_C_FLAGS=${mockcpp_FLAGS}
            -DBOOST_INCLUDE_DIRS=${BOOST_INCLUDE_DIRS}
            -DCMAKE_SHARED_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
            -DCMAKE_EXE_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
            -DBUILD_32_BIT_TARGET_BY_64_BIT_COMPILER=OFF
            -DCMAKE_INSTALL_PREFIX=${OPEN_SOURCE_DIR}/mockcpp
            <SOURCE_DIR>
            BUILD_COMMAND $(MAKE)
            INSTALL_COMMAND $(MAKE) install
            EXCLUDE_FROM_ALL TRUE
    )
    message(STATUS, "get mockcpp")
endif()