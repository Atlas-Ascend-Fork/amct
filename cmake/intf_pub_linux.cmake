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
if(TARGET intf_pub)
  message(STATUS "intf_pub has been found, no need add library")
  return()
endif()

# intf_pub for c++11
add_library(intf_pub INTERFACE)
target_compile_options(intf_pub INTERFACE
  -Wall
  -fPIC
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
  -Wextra
  -Wfloat-equal
  -fno-common
  -fstack-protector-strong
  $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize=leak -fsanitize-recover=address,all -fno-stack-protector -fno-omit-frame-pointer -g>
  $<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread -fsanitize-recover=thread,all -g>
  $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined -fno-sanitize=alignment -g>
  $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
)
target_compile_definitions(intf_pub INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
  $<$<CONFIG:Debug>:CFG_BUILD_NDEBUG>
  WIN64=1
  LINUX=0
)
target_link_options(intf_pub INTERFACE
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
  $<$<CONFIG:Release>:-Wl,--build-id=none>
  $<$<CONFIG:Debug>:-g>
  $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize=leak -fsanitize-recover=address>
  $<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>
  $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>
  $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
)
target_link_directories(intf_pub INTERFACE)
target_link_libraries(intf_pub INTERFACE
  $<$<BOOL:${ENABLE_GCOV}>:-lgcov>
  -lpthread
)

# intf_pub_cxx14 for c++14
add_library(intf_pub_cxx14 INTERFACE)
target_compile_options(intf_pub_cxx14 INTERFACE
  -Wall
  -fPIC
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)
target_compile_definitions(intf_pub_cxx14 INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
  $<$<CONFIG:Debug>:CFG_BUILD_NDEBUG>
  WIN64=1
  LINUX=0
)
target_link_options(intf_pub_cxx14 INTERFACE
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
  $<$<CONFIG:Release>:-Wl,--build-id=none>
  $<$<CONFIG:Debug>:-g>
)
target_link_directories(intf_pub_cxx14 INTERFACE)
target_link_libraries(intf_pub_cxx14 INTERFACE
  -lpthread
)

# intf_pub_cxx14 for c++17
add_library(intf_pub_cxx17 INTERFACE)
target_compile_options(intf_pub_cxx17 INTERFACE
    -Wall
    -fPIC
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
target_compile_definitions(intf_pub_cxx17 INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
    $<$<CONFIG:Debug>:CFG_BUILD_NDEBUG>
    WIN64=1
    LINUX=0)
target_link_options(intf_pub_cxx17 INTERFACE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-Wl,--build-id=none>
    $<$<CONFIG:Debug>:-g>
)
target_link_directories(intf_pub_cxx17 INTERFACE)
target_link_libraries(intf_pub_cxx17 INTERFACE
  -lpthread)

#########intf_pub_aicpu#########
add_library(intf_pub_aicpu INTERFACE)
target_compile_options(intf_pub_aicpu INTERFACE
  -Wall
  -fPIC
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
)
target_compile_definitions(intf_pub_aicpu INTERFACE
  $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:_GLIBCXX_USE_CXX11_ABI=0>
  $<$<STREQUAL:${PRODUCT_SIDE},device>:_GLIBCXX_USE_CXX11_ABI=1>
  $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
  $<$<CONFIG:Debug>:CFG_BUILD_NDEBUG>
  WIN64=1
  LINUX=0
)
target_link_options(intf_pub_aicpu INTERFACE
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
  $<$<CONFIG:Release>:-Wl,--build-id=none>
  $<$<CONFIG:Debug>:-g>
)
target_link_directories(intf_pub_aicpu INTERFACE)
target_link_libraries(intf_pub_aicpu INTERFACE
  -lpthread
)