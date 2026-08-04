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

# Graph 闭源包拉取与解压：
# 优先使用本地预置包；未命中则按日期目录解析 OBS 上最新的 *_newest 包。

if(NOT DEFINED AMCT_OPEN_DIR OR AMCT_OPEN_DIR STREQUAL "")
    get_filename_component(AMCT_OPEN_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

set(AMCT_GRAPH_BASE_NAME "cann-amct-graph")
set(AMCT_GRAPH_BASE_URL_DEFAULT
    "https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN")
set(AMCT_GRAPH_LOOKBACK_DAYS_DEFAULT "120")
set(AMCT_GRAPH_DOWNLOAD_TIMEOUT_DEFAULT "10")

function(amct_graph_set_default _var _default)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
        set(${_var} "${_default}" PARENT_SCOPE)
    endif()
endfunction()

function(amct_graph_date_before _today _offset_days _out_var)
    execute_process(
        COMMAND date -d "${_today} -${_offset_days} day" "+%Y%m%d"
        RESULT_VARIABLE _date_result
        OUTPUT_VARIABLE _package_date
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _date_result EQUAL 0 OR _package_date STREQUAL "")
        message(FATAL_ERROR "failed to calculate package date from ${_today} -${_offset_days} day")
    endif()
    set(${_out_var} "${_package_date}" PARENT_SCOPE)
endfunction()

function(amct_graph_append_unique _list_var _value)
    if("${_value}" STREQUAL "")
        return()
    endif()
    list(FIND ${_list_var} "${_value}" _found_index)
    if(_found_index EQUAL -1)
        list(APPEND ${_list_var} "${_value}")
        set(${_list_var} "${${_list_var}}" PARENT_SCOPE)
    endif()
endfunction()

function(amct_graph_append_version_from_path _list_var _path)
    if("${_path}" MATCHES "cann-([0-9]+\\.[0-9]+\\.[0-9]+)")
        amct_graph_append_unique(${_list_var} "${CMAKE_MATCH_1}")
        set(${_list_var} "${${_list_var}}" PARENT_SCOPE)
    endif()
endfunction()

function(amct_graph_append_version_from_file _list_var _file_path)
    if(NOT EXISTS "${_file_path}")
        return()
    endif()

    file(STRINGS "${_file_path}" _version_lines REGEX "^[Vv]ersion=")
    foreach(_line ${_version_lines})
        if("${_line}" MATCHES "^[Vv]ersion=([0-9]+\\.[0-9]+\\.[0-9]+)")
            amct_graph_append_unique(${_list_var} "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    set(${_list_var} "${${_list_var}}" PARENT_SCOPE)
endfunction()

function(amct_graph_collect_versions)
    if(DEFINED AMCT_GRAPH_VERSION AND NOT "${AMCT_GRAPH_VERSION}" STREQUAL "")
        set(AMCT_GRAPH_VERSIONS "${AMCT_GRAPH_VERSION}" PARENT_SCOPE)
        return()
    endif()
    if(DEFINED AMCT_GRAPH_VERSIONS AND NOT "${AMCT_GRAPH_VERSIONS}" STREQUAL "")
        return()
    endif()

    set(_versions "")
    set(_ascend_home "")
    if(DEFINED ASCEND_HOME_PATH AND NOT "${ASCEND_HOME_PATH}" STREQUAL "")
        set(_ascend_home "${ASCEND_HOME_PATH}")
    elseif(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(_ascend_home "$ENV{ASCEND_HOME_PATH}")
    endif()

    if(NOT _ascend_home STREQUAL "")
        amct_graph_append_version_from_path(_versions "${_ascend_home}")
        amct_graph_append_version_from_file(_versions "${_ascend_home}/${AMCT_GRAPH_ARCH}/ascend_toolkit_install.info")
        amct_graph_append_version_from_file(_versions "${_ascend_home}/ascend_toolkit_install.info")
    endif()

    if(_versions STREQUAL "")
        message(FATAL_ERROR
            "failed to infer graph package version.\n"
            "Hint: set AMCT_GRAPH_VERSION, AMCT_GRAPH_VERSIONS, AMCT_GRAPH_URL, or ASCEND_HOME_PATH.")
    endif()
    set(AMCT_GRAPH_VERSIONS "${_versions}" PARENT_SCOPE)
endfunction()

function(amct_graph_make_source_url _package_date _version _out_var)
    set(_url
        "${AMCT_GRAPH_BASE_URL}/${_package_date}_newest/${AMCT_GRAPH_BASE_NAME}_${_version}_linux-${AMCT_GRAPH_ARCH}.tar.gz")
    set(${_out_var} "${_url}" PARENT_SCOPE)
endfunction()

function(amct_graph_download _url _tar_path _allow_fail _out_var)
    file(REMOVE "${_tar_path}")
    file(DOWNLOAD "${_url}" "${_tar_path}"
        TLS_VERIFY OFF
        TIMEOUT "${AMCT_GRAPH_DOWNLOAD_TIMEOUT}"
        STATUS _dl_status
        LOG _dl_log)
    list(GET _dl_status 0 _dl_code)
    if(_dl_code EQUAL 0)
        set(${_out_var} TRUE PARENT_SCOPE)
        return()
    endif()

    file(REMOVE "${_tar_path}")
    if(_allow_fail)
        set(${_out_var} FALSE PARENT_SCOPE)
        return()
    endif()

    list(GET _dl_status 1 _dl_msg)
    message(FATAL_ERROR "graph package download failed (${_dl_code}: ${_dl_msg})\nurl: ${_url}\n${_dl_log}")
endfunction()

function(amct_graph_resolve_and_download _tar_path)
    if(DEFINED AMCT_GRAPH_URL AND NOT "${AMCT_GRAPH_URL}" STREQUAL "")
        message(STATUS "graph download: ${AMCT_GRAPH_URL}")
        amct_graph_download("${AMCT_GRAPH_URL}" "${_tar_path}" FALSE _downloaded)
        return()
    endif()

    if(DEFINED AMCT_GRAPH_TODAY AND NOT "${AMCT_GRAPH_TODAY}" STREQUAL "")
        set(_today "${AMCT_GRAPH_TODAY}")
    else()
        string(TIMESTAMP _today "%Y%m%d")
    endif()

    foreach(_offset RANGE 0 ${AMCT_GRAPH_LOOKBACK_DAYS})
        amct_graph_date_before("${_today}" "${_offset}" _package_date)
        foreach(_version ${AMCT_GRAPH_VERSIONS})
            amct_graph_make_source_url("${_package_date}" "${_version}" _url)
            amct_graph_download("${_url}" "${_tar_path}" TRUE _downloaded)
            if(_downloaded)
                message(STATUS "graph download: ${_url}")
                return()
            endif()
        endforeach()
    endforeach()

    message(FATAL_ERROR
        "failed to find latest ${AMCT_GRAPH_BASE_NAME} package in the last ${AMCT_GRAPH_LOOKBACK_DAYS} days.\n"
        "Hint: set AMCT_GRAPH_URL to a known package URL, or adjust AMCT_GRAPH_LOOKBACK_DAYS.")
endfunction()

function(amct_install_graph)
    amct_graph_set_default(AMCT_GRAPH_BASE_URL "${AMCT_GRAPH_BASE_URL_DEFAULT}")
    amct_graph_set_default(AMCT_GRAPH_LOOKBACK_DAYS "${AMCT_GRAPH_LOOKBACK_DAYS_DEFAULT}")
    amct_graph_set_default(AMCT_GRAPH_DOWNLOAD_TIMEOUT "${AMCT_GRAPH_DOWNLOAD_TIMEOUT_DEFAULT}")

    if(NOT DEFINED AMCT_GRAPH_ARCH OR AMCT_GRAPH_ARCH STREQUAL "")
        set(AMCT_GRAPH_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    if(AMCT_GRAPH_ARCH STREQUAL "")
        execute_process(
            COMMAND uname -m
            RESULT_VARIABLE _uname_result
            OUTPUT_VARIABLE AMCT_GRAPH_ARCH
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT _uname_result EQUAL 0 OR AMCT_GRAPH_ARCH STREQUAL "")
            message(FATAL_ERROR "failed to detect arch for graph package download")
        endif()
    endif()
    if(NOT AMCT_GRAPH_ARCH MATCHES "^(x86_64|aarch64)$")
        message(FATAL_ERROR "unsupported arch for graph package download: ${AMCT_GRAPH_ARCH}")
    endif()
    if(NOT DEFINED AMCT_GRAPH_URL OR "${AMCT_GRAPH_URL}" STREQUAL "")
        amct_graph_collect_versions()
    endif()

    set(_graph_dir "${AMCT_OPEN_DIR}/amctgraph")
    if(EXISTS "${_graph_dir}" AND IS_DIRECTORY "${_graph_dir}")
        file(GLOB _graph_entries "${_graph_dir}/*")
        if(_graph_entries)
            message(STATUS "graph package already present at ${_graph_dir}, skip download")
            return()
        endif()
        message(STATUS "graph package dir empty, re-downloading")
    endif()
    file(MAKE_DIRECTORY "${_graph_dir}")

    set(_local_tar "")
    if(DEFINED AMCT_GRAPH_VERSIONS AND NOT "${AMCT_GRAPH_VERSIONS}" STREQUAL "")
        foreach(_version ${AMCT_GRAPH_VERSIONS})
            set(_tar_name "${AMCT_GRAPH_BASE_NAME}_${_version}_linux-${AMCT_GRAPH_ARCH}.tar.gz")
            foreach(_cand
                "${AMCT_OPEN_DIR}/build/${_tar_name}"
                "${CANN_3RD_LIB_PATH}/${_tar_name}")
                if(_local_tar STREQUAL "" AND EXISTS "${_cand}")
                    set(_local_tar "${_cand}")
                endif()
            endforeach()
        endforeach()
    endif()

    if(NOT _local_tar STREQUAL "")
        message(STATUS "graph package using local tarball: ${_local_tar}")
        file(COPY "${_local_tar}" DESTINATION "${_graph_dir}")
        get_filename_component(_tar_name "${_local_tar}" NAME)
        set(_tar_path "${_graph_dir}/${_tar_name}")
    else()
        set(_tar_path "${_graph_dir}/${AMCT_GRAPH_BASE_NAME}_linux-${AMCT_GRAPH_ARCH}.tar.gz")
        amct_graph_resolve_and_download("${_tar_path}")
    endif()

    file(ARCHIVE_EXTRACT
        INPUT "${_tar_path}"
        DESTINATION "${_graph_dir}")
    file(REMOVE "${_tar_path}")
    message(STATUS "graph package ready at ${_graph_dir}")
endfunction()

amct_install_graph()
