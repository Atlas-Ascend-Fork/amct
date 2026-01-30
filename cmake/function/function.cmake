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
function(protobuf_generate comp c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: protobuf_generate() called without any proto files")
        return()
    endif()
    set(${c_var})
    set(${h_var})
    set(_add_target FALSE)

    set(extra_option "")
    foreach(arg ${ARGN})
        if ("${arg}" MATCHES "--proto_path")
            set(extra_option ${arg})
        endif()
    endforeach()

    foreach(file ${ARGN})
        if("${file}" STREQUAL "TARGET")
            set(_add_target TRUE)
            continue()
        endif()

        if ("${file}" MATCHES "--proto_path")
            continue()
        endif()

        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        get_filename_component(parent_subdir ${file_dir} NAME)

        if("${parent_subdir}" STREQUAL "proto")
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto)
        else()
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto/${parent_subdir})
        endif()
        list(APPEND ${c_var} "${proto_output_path}/${file_name}.pb.cc")
        list(APPEND ${h_var} "${proto_output_path}/${file_name}.pb.h")

        add_custom_command(
                OUTPUT "${proto_output_path}/${file_name}.pb.cc" "${proto_output_path}/${file_name}.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_path}"
                COMMAND ${CMAKE_COMMAND} -E echo "generate proto cpp_out ${comp} by ${abs_file}"
                COMMAND host_protoc -I${file_dir} ${extra_option} --cpp_out=${proto_output_path} ${abs_file}
                DEPENDS ${abs_file} host_protoc
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM )
    endforeach()

    if(_add_target)
        add_custom_target(
                ${comp} DEPENDS ${${c_var}} ${${h_var}}
        )
    endif()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()

macro(install_package)
    set(file_count 1)
    set(directory_count 1)
    foreach(arg ${ARGN})
        if (arg STREQUAL "PACKAGE")
            set (key PKG_NAME)
        elseif (arg STREQUAL "TARGETS")
            set (key TARGET_LIST)
        elseif (arg STREQUAL "FILES")
            set (key ${arg}_${file_count})
            set (prekey "FILES")
        elseif (arg STREQUAL "DIRECTORY")
            set (key ${arg}_${directory_count})
            set (prekey "DIRECTORY")
        elseif (arg STREQUAL "DESTINATION")
            if (prekey STREQUAL "FILES")
                set (key ${prekey}_${arg}_${file_count})
                math(EXPR file_count "${file_count}+1")
            else ()
                set (key ${prekey}_${arg}_${directory_count})
                math(EXPR directory_count "${directory_count}+1")
            endif()
        else ()
            list(APPEND ${key} ${arg})
        endif()
    endforeach()
    math(EXPR file_count "${file_count}-1")
    math(EXPR directory_count "${directory_count}-1")
 
    install(TARGETS ${TARGET_LIST}
        EXPORT ${PKG_NAME}-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
        RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL COMPONENT opensdk
    )
 
    if (file_count GREATER 0)
        foreach(i RANGE 1 ${file_count})
            install(FILES ${FILES_${i}} DESTINATION ${FILES_DESTINATION_${i}} COMPONENT opensdk EXCLUDE_FROM_ALL)
        endforeach()
    endif()
 
    if (directory_count GREATER 0)
        foreach(i RANGE 1 ${directory_count})
            install(DIRECTORY ${DIRECTORY_${i}} DESTINATION ${DIRECTORY_DESTINATION_${i}}
                COMPONENT opensdk EXCLUDE_FROM_ALL
                FILES_MATCHING 
                PATTERN "*.h"
                PATTERN "*.cppm")
        endforeach()
    endif()
 
    if (PACKAGE STREQUAL "opensdk")
        install(EXPORT ${PKG_NAME}-targets DESTINATION ${INSTALL_CONFIG_DIR}
            FILE ${PKG_NAME}-targets.cmake COMPONENT opensdk EXCLUDE_FROM_ALL
        )
        configure_package_config_file(${TOP_DIR}/cmake/config/pkg_config_template.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}-config.cmake
            INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
            PATH_VARS INSTALL_BASE_DIR INSTALL_INCLUDE_DIR INSTALL_LIBRARY_DIR INSTALL_RUNTIME_DIR INSTALL_CONFIG_DIR
            INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
        )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}-config.cmake
            DESTINATION ${INSTALL_CONFIG_DIR} COMPONENT opensdk EXCLUDE_FROM_ALL
        )
    endif()
 
    unset(PKG_NAME)
    unset(TARGET_LIST)
    if (file_count GREATER 0)
        foreach(i RANGE 1 ${file_count})
            unset(FILES_${i})
            unset(FILES_DESTINATION_${i})
        endforeach()
    endif()
    if (directory_count GREATER 0)
        foreach(i RANGE 1 ${directory_count})
            unset(DIRECTORY_${i})
            unset(DIRECTORY_DESTINATION_${i})
        endforeach()
    endif()
endmacro(install_package)
 