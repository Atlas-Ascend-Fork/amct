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
# 说明: 本文件是 cann-cmake(master-049) third_party/protobuf.cmake 的兼容性改造版,
#       经 cmake/fetch_cann_cmake.cmake 以 overlay 方式覆盖原始文件。
#       改造点: 复用 lib_cache 二进制缓存前, 先用当前工具链对缓存产物做试链接/试运行探测,
#       不兼容(如缓存由 gcc-15 构建、当前为 gcc-7.3 老工具链)时自动回退源码构建,
#       保证老/新工具链均能正常编译。上游 cann-cmake 修复后本文件可移除。
# ----------------------------------------------------------------------------
include_guard(GLOBAL)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(ExternalProject)
include(GNUInstallDirs)
include(CheckCXXSourceCompiles)
include(${CMAKE_CURRENT_LIST_DIR}/abseil-cpp.cmake)

# ==========================================================================================================
# 0. Cache Compatibility Probe
# ==========================================================================================================
# 动态库(.so): 用当前工具链试链接(复现 --no-undefined 行为)。
# 失败说明缓存由不兼容的工具链产出(如依赖更新版本的 GLIBCXX/GLIBC 符号), 需回退源码构建。
# 注: 需同时加 --no-as-needed, 否则默认 as-needed 语义下未被直接引用的 so 会被丢弃,
#     其内部不满足的版本化符号不会触发链接报错, 探测会漏判
macro(protobuf_cache_linkable out_var cache_lib)
    set(CMAKE_REQUIRED_QUIET ON)
    set(CMAKE_REQUIRED_LINK_OPTIONS -Wl,--no-undefined -Wl,--no-as-needed)
    set(CMAKE_REQUIRED_LIBRARIES "${cache_lib}")
    check_cxx_source_compiles(
        "__attribute__((used)) int protobuf_cache_probe_main() { return 0; }
        int main() { return protobuf_cache_probe_main(); }" ${out_var})
    unset(CMAKE_REQUIRED_QUIET)
    unset(CMAKE_REQUIRED_LINK_OPTIONS)
    unset(CMAKE_REQUIRED_LIBRARIES)
endmacro()

# 静态库(.a): relocatable 成员不带符号版本信息, 且 protobuf 归档内含多个同名成员,
# whole-archive 试链接会因归档内部变体间引用而必然失败, 不能作为兼容性判据。
# 改为只读检查成员的 ELF 机器架构与当前编译目标一致。
# 注: CMAKE_LIBRARY_ARCHITECTURE 形如 "aarch64-linux-gnu"/"x86_64-linux-gnu",
#     而 readelf 输出形如 "AArch64"/"Advanced Micro Devices X86-64", 取三元组首段做匹配
macro(protobuf_static_cache_compatible out_var cache_lib)
    set(${out_var} TRUE)
    if(NOT CMAKE_CXX_COMPILER_AR)
        set(CMAKE_CXX_COMPILER_AR ar)
    endif()
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER_AR} t "${cache_lib}"
        OUTPUT_VARIABLE _ar_members
        RESULT_VARIABLE _ar_rc
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _ar_rc EQUAL 0)
        message(WARNING "[Third-Party][protobuf] cannot list members of ${cache_lib}, treat as incompatible.")
        set(${out_var} FALSE)
    else()
        # cmake 3.20 的 execute_process 不会把 "\n" 自动转为 list 分隔符, 需手动转换
        string(REPLACE "\n" ";" _ar_members "${_ar_members}")
        list(GET _ar_members 0 _first_member)
        set(_probe_obj "${CMAKE_BINARY_DIR}/protobuf_cache_probe_member.o")
        file(REMOVE "${_probe_obj}")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER_AR} p "${cache_lib}" "${_first_member}"
            OUTPUT_FILE "${_probe_obj}"
            RESULT_VARIABLE _ex_rc)
        if(NOT _ex_rc EQUAL 0 OR NOT EXISTS "${_probe_obj}")
            message(WARNING "[Third-Party][protobuf] cannot extract member ${_first_member} from ${cache_lib}.")
            set(${out_var} FALSE)
        else()
            execute_process(
                COMMAND readelf -h "${_probe_obj}"
                OUTPUT_VARIABLE _elf_hdr
                RESULT_VARIABLE _re_rc
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            # 取三元组机器段: aarch64 -> AARCH64; x86_64 -> X86(readelf 输出为 x86-64)
            string(REGEX MATCH "^([^-]*)" _mach_seg "${CMAKE_LIBRARY_ARCHITECTURE}")
            string(TOUPPER "${CMAKE_MATCH_1}" _mach_upper)
            string(TOUPPER "${_elf_hdr}" _elf_hdr_upper)
            if("${CMAKE_MATCH_1}" STREQUAL "x86" )
                set(_mach_upper "X86")
            endif()
            if(NOT _re_rc EQUAL 0 OR NOT _elf_hdr_upper MATCHES "MACHINE:[ \t]+.*${_mach_upper}")
                message(WARNING "[Third-Party][protobuf] cache ${cache_lib} architecture mismatch "
                    "(toolchain: ${CMAKE_LIBRARY_ARCHITECTURE}), fallback to source build.")
                set(${out_var} FALSE)
            endif()
            file(REMOVE "${_probe_obj}")
        endif()
    endif()
    unset(_ar_members)
    unset(_ar_rc)
    unset(_first_member)
    unset(_probe_obj)
    unset(_elf_hdr)
    unset(_re_rc)
    unset(_ex_rc)
    unset(_mach_seg)
    unset(_mach_upper)
endmacro()

# 缓存的可执行文件(protoc)换到更老的节点可能无法运行, 探测失败则回退源码构建
macro(protobuf_protoc_runnable out_var protoc_bin)
    set(${out_var} FALSE)
    if(EXISTS "${protoc_bin}" AND NOT IS_DIRECTORY "${protoc_bin}")
        execute_process(COMMAND "${protoc_bin}" --version
            RESULT_VARIABLE _protoc_rc OUTPUT_QUIET ERROR_QUIET)
        if(_protoc_rc EQUAL 0)
            set(${out_var} TRUE)
        endif()
    endif()
    unset(_protoc_rc)
endmacro()

# ==========================================================================================================
# 1. Paths & Directories Setup
# ==========================================================================================================
set(PROTOBUF_STATIC_PKG_DIR ${CANN_3RD_LIB_PATH}/lib_cache/protobuf_static)
set(PROTOBUF_SHARED_PKG_DIR ${CANN_3RD_LIB_PATH}/lib_cache/protobuf_shared)
set(PROTOBUF_HOST_STATIC_PKG_DIR ${CANN_3RD_LIB_PATH}/lib_cache/protobuf_host_static)
set(PROTOBUF_HOST_PROTOC_DIR ${CANN_3RD_LIB_PATH}/lib_cache/protoc)

set(LIB_SUB_DIR "lib64")
set(PROTOBUF_STATIC_FILE_NAME "libhost_ascend_protobuf.a")
if(PRODUCT_SIDE STREQUAL "device")
    set(PROTOBUF_STATIC_FILE_NAME "libascend_protobuf.a")
    set(LIB_SUB_DIR "lib64/device/lib64")

    # in device mode, need another path to save lib.
    message("[Third-Party][protobuf] device mode set protobuf lib path")
    set(PROTOBUF_STATIC_PKG_DIR ${CANN_3RD_LIB_PATH}/lib_cache/device/protobuf_static)
    set(PROTOBUF_SHARED_PKG_DIR ${CANN_3RD_LIB_PATH}/lib_cache/device/protobuf_shared)
    set(PROTOBUF_HOST_STATIC_PKG_DIR ${CANN_3RD_LIB_PATH}/lib_cache/device/protobuf_host_static)
    set(PROTOBUF_HOST_PROTOC_DIR ${CANN_3RD_LIB_PATH}/lib_cache/device/protoc)
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(LD_LIB_PATHS "$ENV{ASCEND_HOME_PATH}/${LIB_SUB_DIR}")
    set(LD_BIN_PATHS "$ENV{ASCEND_HOME_PATH}/bin")
endif()

# ==========================================================================================================
# 2. Compile Flags & Optimization
# ==========================================================================================================
set(SECURITY_COMPILE_OPT "-Wl,-z,relro,-z,now,-z,noexecstack -s -Wl,-Bsymbolic")
set(DEV_PROTOBUF_SHARED_CXXFLAGS "${SECURITY_COMPILE_OPT} -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=1 -O2 -Dgoogle=ascend_private")
set(HOST_PROTOBUF_SHARED_CXXFLAGS "${SECURITY_COMPILE_OPT} -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -Dgoogle=ascend_private")
set(DEV_PROTOBUF_STATIC_CXXFLAGS "-fvisibility=hidden -fvisibility-inlines-hidden -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=1 -O2 -Dgoogle=ascend_private")
set(HOST_PROTOBUF_STATIC_CXXFLAGS "-fvisibility=hidden -fvisibility-inlines-hidden -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -Dgoogle=ascend_private")

message("[Third-Party][protobuf] PRODUCT_SIDE: ${PRODUCT_SIDE}")
message("[Third-Party][protobuf] CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
if(PRODUCT_SIDE STREQUAL "device")
    set(HOST_PROTOBUF_SHARED_CXXFLAGS ${DEV_PROTOBUF_SHARED_CXXFLAGS})
    set(HOST_PROTOBUF_STATIC_CXXFLAGS ${DEV_PROTOBUF_STATIC_CXXFLAGS})
    set(PROTOBUF_TOOLCHAIN_ARGS
        -DTOOLCHAIN_DIR=${TOOLCHAIN_DIR}
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    )
elseif(CMAKE_TOOLCHAIN_FILE)
    # used for ge
    set(PROTOBUF_TOOLCHAIN_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DLLVM_PATH=${LLVM_PATH}
    )
else()
    set(PROTOBUF_TOOLCHAIN_ARGS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/protobuf_sym_rename.cmake)
set(HOST_PROTOBUF_SHARED_CXXFLAGS "${HOST_PROTOBUF_SHARED_CXXFLAGS} ${PROTOBUF_SYM_RENAME}")
set(HOST_PROTOBUF_STATIC_CXXFLAGS "${HOST_PROTOBUF_STATIC_CXXFLAGS} ${PROTOBUF_SYM_RENAME}")

# ==========================================================================================================
# 3. Find Existing Libraries & Protoc
# ==========================================================================================================
find_program(PROTOC_PROGRAM
    NAMES protoc
    PATHS ${PROTOBUF_HOST_PROTOC_DIR} ${LD_BIN_PATHS}
    NO_DEFAULT_PATH
)

find_library(ASCEND_PROTOBUF_STATIC_LIB
    NAMES libascend_protobuf.a
    PATH_SUFFIXES lib lib64
    PATHS ${PROTOBUF_STATIC_PKG_DIR} ${LD_LIB_PATHS}
    NO_DEFAULT_PATH
)

find_path(ASCEND_PROTOBUF_STATIC_INCLUDE
    NAMES google/protobuf/any.h
    PATH_SUFFIXES include
    PATHS ${PROTOBUF_STATIC_PKG_DIR} ${LD_LIB_PATHS}
    NO_DEFAULT_PATH
)

find_library(HOST_PROTOBUF_STATIC_LIB
    NAMES ${PROTOBUF_STATIC_FILE_NAME}
    PATH_SUFFIXES lib lib64
    PATHS ${PROTOBUF_HOST_STATIC_PKG_DIR} ${LD_LIB_PATHS}
    NO_DEFAULT_PATH
)

find_path(HOST_PROTOBUF_STATIC_INCLUDE
    NAMES google/protobuf/any.h
    PATH_SUFFIXES include
    PATHS ${PROTOBUF_HOST_STATIC_PKG_DIR} ${LD_LIB_PATHS}
    NO_DEFAULT_PATH
)

find_library(ASCEND_PROTOBUF_SHARED_LIB
    NAMES libascend_protobuf.so
    PATH_SUFFIXES lib lib64
    PATHS ${PROTOBUF_SHARED_PKG_DIR} ${LD_LIB_PATHS}
    NO_DEFAULT_PATH
)

find_path(ASCEND_PROTOBUF_SHARED_INCLUDE
    NAMES google/protobuf/any.h
    PATH_SUFFIXES include
    PATHS ${PROTOBUF_SHARED_PKG_DIR} ${LD_LIB_PATHS}
    NO_DEFAULT_PATH
)

message("[Third-Party][protobuf] PROTOC_PROGRAM : ${PROTOC_PROGRAM}.")
message("[Third-Party][protobuf] ASCEND_PROTOBUF_STATIC_LIB : ${ASCEND_PROTOBUF_STATIC_LIB}.")
message("[Third-Party][protobuf] HOST_PROTOBUF_STATIC_LIB : ${HOST_PROTOBUF_STATIC_LIB}.")
message("[Third-Party][protobuf] ASCEND_PROTOBUF_SHARED_LIB : ${ASCEND_PROTOBUF_SHARED_LIB}.")

# 解析 protobuf 源码包路径
if (EXISTS ${CANN_3RD_LIB_PATH}/protobuf-all-25.1.tar.gz)
    set(REQ_URL ${CANN_3RD_LIB_PATH}/protobuf-all-25.1.tar.gz)
elseif (EXISTS ${CANN_3RD_LIB_PATH}/protobuf-25.1.tar.gz)
    set(REQ_URL ${CANN_3RD_LIB_PATH}/protobuf-25.1.tar.gz)
elseif (EXISTS ${CANN_3RD_LIB_PATH}/protobuf/protobuf-all-25.1.tar.gz)
    set(REQ_URL ${CANN_3RD_LIB_PATH}/protobuf/protobuf-all-25.1.tar.gz)
else()
    set(REQ_URL ${CANN_3RD_LIB_PATH}/protobuf/protobuf-25.1.tar.gz)
endif()

if(NOT EXISTS ${REQ_URL})
    message("[Third-Party][protobuf] ${REQ_URL} not found, need download.")
    set(REQ_URL "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/protobuf/protobuf-25.1.tar.gz")
endif()

# 避免host和device同时编译时下载路径冲突
if(PRODUCT_SIDE STREQUAL "device")
    set(PROTOBUF_DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}/pkg)
else()
    set(PROTOBUF_DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}/protobuf)
endif()

ExternalProject_Add(protobuf_src
    URL ${REQ_URL}
    URL_HASH SHA256=9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42
    TIMEOUT 300
    DOWNLOAD_DIR ${PROTOBUF_DOWNLOAD_DIR}
    PATCH_COMMAND patch --forward --batch -p1 < ${CMAKE_CURRENT_LIST_DIR}/patch/protobuf_25.1_change_version.patch
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    DEPENDS abseil_build
)

ExternalProject_Get_Property(protobuf_src SOURCE_DIR)
set(PROTOBUF_SRC_DIR ${SOURCE_DIR} CACHE INTERNAL "protobuf src dir")

# ---------------------------------------------------------
# Target: ascend_protobuf (Shared)
# ---------------------------------------------------------
add_library(ascend_protobuf SHARED IMPORTED GLOBAL)
add_library(ascend_protobuf_shared_headers INTERFACE IMPORTED GLOBAL)
set(PROTOBUF_SHARED_USE_CACHE FALSE)
if(ASCEND_PROTOBUF_SHARED_INCLUDE AND ASCEND_PROTOBUF_SHARED_LIB)
    protobuf_cache_linkable(PROTOBUF_SHARED_CACHE_OK "${ASCEND_PROTOBUF_SHARED_LIB}")
    if(PROTOBUF_SHARED_CACHE_OK)
        set(PROTOBUF_SHARED_USE_CACHE TRUE)
    else()
        message(WARNING "[Third-Party][protobuf] cache ${ASCEND_PROTOBUF_SHARED_LIB} is not linkable "
            "by current toolchain (${CMAKE_CXX_COMPILER}), fallback to source build.")
    endif()
endif()

if(PROTOBUF_SHARED_USE_CACHE)
    message("[Third-Party][protobuf] protobuf_shared use cache.")
    set_target_properties(ascend_protobuf PROPERTIES
        IMPORTED_LOCATION ${ASCEND_PROTOBUF_SHARED_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ASCEND_PROTOBUF_SHARED_INCLUDE}
    )
    set(_ASCEND_PROTOBUF_SHARED_INCLUDE ${ASCEND_PROTOBUF_SHARED_INCLUDE})
else()
    message("[Third-Party][protobuf] protobuf_shared build.")
    ExternalProject_Add(protobuf_shared_build
        DEPENDS protobuf_src
        SOURCE_DIR ${PROTOBUF_SRC_DIR}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -G ${CMAKE_GENERATOR}
            ${PROTOBUF_TOOLCHAIN_ARGS}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DCMAKE_INSTALL_LIBDIR=lib
            -DBUILD_SHARED_LIBS=ON
            -DCMAKE_CXX_STANDARD=14
            -Dprotobuf_WITH_ZLIB=OFF
            -DLIB_PREFIX=ascend_
            -DCMAKE_SKIP_RPATH=TRUE
            -Dprotobuf_BUILD_TESTS=OFF
            -DCMAKE_CXX_FLAGS=${HOST_PROTOBUF_SHARED_CXXFLAGS}
            -DCMAKE_INSTALL_PREFIX=${PROTOBUF_SHARED_PKG_DIR}
            -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
            -DABSL_ROOT_DIR=${ABS_INSTALL_DIR}
            <SOURCE_DIR>
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        EXCLUDE_FROM_ALL TRUE
    )
    set(_ASCEND_PROTOBUF_SHARED_INCLUDE ${PROTOBUF_SHARED_PKG_DIR}/include)
    set_target_properties(ascend_protobuf PROPERTIES
        IMPORTED_LOCATION ${PROTOBUF_SHARED_PKG_DIR}/lib/libascend_protobuf.so
        INTERFACE_INCLUDE_DIRECTORIES "${_ASCEND_PROTOBUF_SHARED_INCLUDE}"
    )
    add_dependencies(ascend_protobuf protobuf_shared_build)
    add_dependencies(ascend_protobuf_shared_headers protobuf_shared_build)
endif()
target_include_directories(ascend_protobuf_shared_headers INTERFACE ${PROTOBUF_SHARED_PKG_DIR}/include)
create_imported_interface_include_directories(ascend_protobuf)

# use for runtime: PROTOBUF_SHARED_LIB_DIR
set(PROTOBUF_SHARED_LIB_DIR "${PROTOBUF_SHARED_PKG_DIR}/lib")

# ---------------------------------------------------------
# Target: host_protoc
# ---------------------------------------------------------
add_executable(host_protoc IMPORTED GLOBAL)
if(PROTOC_PROGRAM)
    protobuf_protoc_runnable(PROTOC_RUNNABLE "${PROTOC_PROGRAM}")
    if(NOT PROTOC_RUNNABLE)
        message(WARNING "[Third-Party][protobuf] cached protoc ${PROTOC_PROGRAM} cannot run "
            "with current toolchain/runtime, fallback to build.")
        set(PROTOC_PROGRAM PROTOC_PROGRAM-NOTFOUND)
    endif()
endif()
if(PROTOC_PROGRAM)
    message("[Third-Party][protobuf] protoc use cache.")
    set_target_properties(host_protoc PROPERTIES IMPORTED_LOCATION ${PROTOC_PROGRAM})
else()
    message("[Third-Party][protobuf] protoc build.")
    ExternalProject_Add(protobuf_host_build
        DEPENDS protobuf_src
        SOURCE_DIR ${PROTOBUF_SRC_DIR}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -DCMAKE_CXX_STANDARD=14
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_INSTALL_PREFIX=${PROTOBUF_HOST_PROTOC_DIR}
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_WITH_ZLIB=OFF
            -DABSL_ROOT_DIR=${ABS_INSTALL_DIR}
            <SOURCE_DIR>
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${PROTOBUF_HOST_PROTOC_DIR}
            COMMAND cp <BINARY_DIR>/protoc ${PROTOBUF_HOST_PROTOC_DIR}/protoc
        EXCLUDE_FROM_ALL TRUE
    )
    set_target_properties(host_protoc PROPERTIES IMPORTED_LOCATION ${PROTOBUF_HOST_PROTOC_DIR}/protoc)
    add_dependencies(host_protoc protobuf_host_build)
endif()
# used for ge
add_custom_target(protoc DEPENDS host_protoc)

# ---------------------------------------------------------
# Target: ascend_protobuf_static
# ---------------------------------------------------------
add_library(ascend_protobuf_static STATIC IMPORTED GLOBAL)
add_library(ascend_protobuf_static_headers INTERFACE IMPORTED GLOBAL)
set(PROTOBUF_STATIC_USE_CACHE FALSE)
if(ASCEND_PROTOBUF_STATIC_INCLUDE AND ASCEND_PROTOBUF_STATIC_LIB)
    protobuf_static_cache_compatible(PROTOBUF_STATIC_CACHE_OK "${ASCEND_PROTOBUF_STATIC_LIB}")
    if(PROTOBUF_STATIC_CACHE_OK)
        set(PROTOBUF_STATIC_USE_CACHE TRUE)
    endif()
endif()

if(PROTOBUF_STATIC_USE_CACHE)
    message("[Third-Party][protobuf] protobuf_static use cache.")
    set_target_properties(ascend_protobuf_static PROPERTIES
        IMPORTED_LOCATION ${ASCEND_PROTOBUF_STATIC_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ASCEND_PROTOBUF_STATIC_INCLUDE}
    )
    set(_ASCEND_PROTOBUF_STATIC_INCLUDE ${ASCEND_PROTOBUF_STATIC_INCLUDE})
    set(PROTOBUF_STATIC_FINAL_PATH ${ASCEND_PROTOBUF_STATIC_LIB})
else()
    message("[Third-Party][protobuf] protobuf_static build.")
    ExternalProject_Add(protobuf_static_build
        DEPENDS protobuf_src
        SOURCE_DIR ${PROTOBUF_SRC_DIR}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -G ${CMAKE_GENERATOR}
            ${PROTOBUF_TOOLCHAIN_ARGS}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_INSTALL_LIBDIR=lib
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_CXX_STANDARD=14
            -Dprotobuf_WITH_ZLIB=OFF
            -DLIB_PREFIX=ascend_
            -DCMAKE_SKIP_RPATH=TRUE
            -Dprotobuf_BUILD_TESTS=OFF
            -DCMAKE_CXX_FLAGS=${HOST_PROTOBUF_STATIC_CXXFLAGS}
            -DCMAKE_INSTALL_PREFIX=${PROTOBUF_STATIC_PKG_DIR}
            -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
            -DABSL_COMPILE_OBJ=TRUE
            -DABSL_ROOT_DIR=${ABS_INSTALL_DIR}
            <SOURCE_DIR>
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        EXCLUDE_FROM_ALL TRUE
    )
    set(_ASCEND_PROTOBUF_STATIC_INCLUDE ${PROTOBUF_STATIC_PKG_DIR}/include)
    set_target_properties(ascend_protobuf_static PROPERTIES
        IMPORTED_LOCATION ${PROTOBUF_STATIC_PKG_DIR}/lib/libascend_protobuf.a
        INTERFACE_INCLUDE_DIRECTORIES "${_ASCEND_PROTOBUF_STATIC_INCLUDE}"
    )
    add_dependencies(ascend_protobuf_static protobuf_static_build)
    add_dependencies(ascend_protobuf_static_headers protobuf_static_build)
    set(PROTOBUF_STATIC_FINAL_PATH ${PROTOBUF_STATIC_PKG_DIR}/lib/libascend_protobuf.a)
endif()
target_include_directories(ascend_protobuf_static_headers INTERFACE ${_ASCEND_PROTOBUF_STATIC_INCLUDE})
create_imported_interface_include_directories(ascend_protobuf_static)

# ---------------------------------------------------------
# Target: protobuf_static (Host Static)
# ---------------------------------------------------------
add_library(protobuf_static STATIC IMPORTED GLOBAL)
set(PROTOBUF_HOST_STATIC_USE_CACHE FALSE)
if(HOST_PROTOBUF_STATIC_INCLUDE AND HOST_PROTOBUF_STATIC_LIB)
    protobuf_static_cache_compatible(PROTOBUF_HOST_STATIC_CACHE_OK "${HOST_PROTOBUF_STATIC_LIB}")
    if(PROTOBUF_HOST_STATIC_CACHE_OK)
        set(PROTOBUF_HOST_STATIC_USE_CACHE TRUE)
    endif()
endif()

if(PROTOBUF_HOST_STATIC_USE_CACHE)
    message("[Third-Party][protobuf] protobuf_host_static use cache.")
    set_target_properties(protobuf_static PROPERTIES
        IMPORTED_LOCATION ${HOST_PROTOBUF_STATIC_LIB}
        INTERFACE_INCLUDE_DIRECTORIES "${HOST_PROTOBUF_STATIC_INCLUDE}"
    )
    set(PROTOBUF_HOST_STATIC_FINAL_PATH ${HOST_PROTOBUF_STATIC_LIB})
else()
    message("[Third-Party][protobuf] protobuf_host_static build.")
    ExternalProject_Add(protobuf_host_static_build
        DEPENDS protobuf_src
        SOURCE_DIR ${PROTOBUF_SRC_DIR}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -G ${CMAKE_GENERATOR}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_INSTALL_LIBDIR=lib
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_CXX_STANDARD=14
            -Dprotobuf_WITH_ZLIB=OFF
            -DLIB_PREFIX=host_ascend_
            -DCMAKE_SKIP_RPATH=TRUE
            -Dprotobuf_BUILD_TESTS=OFF
            -DCMAKE_CXX_FLAGS=${HOST_PROTOBUF_STATIC_CXXFLAGS}
            -DCMAKE_INSTALL_PREFIX=${PROTOBUF_HOST_STATIC_PKG_DIR}
            -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
            -DABSL_COMPILE_OBJ=TRUE
            -DABSL_ROOT_DIR=${ABS_INSTALL_DIR}
            <SOURCE_DIR>
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        EXCLUDE_FROM_ALL TRUE
    )

    set_target_properties(protobuf_static PROPERTIES
        IMPORTED_LOCATION ${PROTOBUF_HOST_STATIC_PKG_DIR}/lib/libhost_ascend_protobuf.a
        INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_HOST_STATIC_PKG_DIR}/include"
    )
    add_dependencies(protobuf_static protobuf_host_static_build)
    set(PROTOBUF_HOST_STATIC_FINAL_PATH ${PROTOBUF_HOST_STATIC_PKG_DIR}/lib/libhost_ascend_protobuf.a)
endif()
create_imported_interface_include_directories(protobuf_static)
