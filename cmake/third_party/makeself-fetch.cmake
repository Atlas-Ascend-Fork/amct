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

set(MAKESELF_NAME "makeself")
set(MAKESELF_PATH "${OPEN_SOURCE_DIR}/${MAKESELF_NAME}")

# 默认配置的makeself还是不存在则下载
if (NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh" OR NOT EXISTS "${MAKESELF_PATH}/makeself.sh")
    set(MAKESELF_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz")
    message(STATUS "Downloading ${MAKESELF_NAME} from ${MAKESELF_URL}")

    include(FetchContent)
    FetchContent_Declare(
        ${MAKESELF_NAME}
        URL ${MAKESELF_URL}
        URL_HASH SHA256=bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a
        SOURCE_DIR "${MAKESELF_PATH}"  # 直接解压到此目录
    )
    FetchContent_MakeAvailable(${MAKESELF_NAME})
    
else()
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/makeself")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy
        "${OPEN_SOURCE_DIR}/makeself/makeself-header.sh"
        "${OPEN_SOURCE_DIR}/makeself/makeself.sh"
        "${CMAKE_BINARY_DIR}/makeself"
        RESULT_VARIABLE copy_result
    )
    if (copy_result)
        message(FATAL_ERROR "FAILE TO COPY MAKE SELF FILES")
    endif()
endif()

execute_process(
    COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself.sh"
    COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself-header.sh"
    RESULT_VARIABLE CHMOD_RESULT
    ERROR_VARIABLE CHMOD_ERROR
)
