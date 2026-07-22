cmake_minimum_required(VERSION 3.16)
project(ascendc_mxfp4)

set(SOC_VERSION "Ascend910B3" CACHE STRING "SoC version")
set(RUN_MODE "npu" CACHE STRING "npu / cpu / sim")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "CANN toolkit path")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# ==================================================================
# 1.  Device kernel  (ccec via ascendc.cmake -> libascendc_*_npu.so)
# ==================================================================
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR
        ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR
        ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under CANN package")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/mxfp4_kernel.cpp)
ascendc_library(ascendc_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})

# ==================================================================
# 2.  Pybind11 extension  (mxfp4_ascendc_ext.so)
# ==================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'include'))"
    OUTPUT_VARIABLE TORCH_NPU_INC OUTPUT_STRIP_TRAILING_WHITESPACE)

# CANN ships arch-specific headers/libs under <arch>-linux (e.g. aarch64-linux,
# x86_64-linux). Hardcoding aarch64 breaks x86_64 builds (dev hosts, simulator).
# Prefer the ARCH passed by build.sh (from `uname -m`); fall back to the CMake
# host processor. Only add the arch dir when it actually exists so we stay
# robust to CANN layouts that place headers directly under include/.
if(NOT DEFINED ARCH OR ARCH STREQUAL "")
    set(ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()

set(ASCEND_INC ${ASCEND_CANN_PACKAGE_PATH}/include)
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/include)
    list(APPEND ASCEND_INC ${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/include)
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/lib64)
    set(ASCEND_LIB ${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/lib64)
else()
    set(ASCEND_LIB ${ASCEND_CANN_PACKAGE_PATH}/lib64)
endif()

pybind11_add_module(mxfp4_ascendc_ext
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind_wrapper.cpp)

target_include_directories(mxfp4_ascendc_ext PRIVATE
    ${ASCEND_INC}
    ${TORCH_NPU_INC}
    ${CMAKE_CURRENT_BINARY_DIR})

execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI OUTPUT_STRIP_TRAILING_WHITESPACE)

target_compile_options(mxfp4_ascendc_ext PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

target_link_directories(mxfp4_ascendc_ext PRIVATE ${ASCEND_LIB})
target_link_libraries(mxfp4_ascendc_ext PRIVATE
    ${TORCH_LIBRARIES}
    ascendc_kernels_${RUN_MODE}
    ascendcl)
