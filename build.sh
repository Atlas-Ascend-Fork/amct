#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)
BUILD_PATH="${BASEPATH}/build/"
UT_TARGETS=("amct_utest")
TMP_PATH="${BASEPATH}/tmp/"

# print usage message
usage() {
  echo "Usage:"
  echo "  sh build.sh [-h | --help] [-v | --verbose] [-j<N>]"
  echo "              [--cann_3rd_lib_path=<PATH>] [--output_path=<PATH>]"
  echo ""
  echo "Options:"
  echo "    -h, --help        Print usage"
  echo "    -v, --verbose     Display build command"
  echo "    -j<N>             Set the number of threads used for building HIXL, default is 8"
  echo "    --build-type=<TYPE>"
  echo "                      Specify build type (TYPE options: Release/Debug), Default: Release"
  echo "    --pkg             Build run package"
  echo "    --cann_3rd_lib_path=<PATH>"
  echo "                      Set ascend third_party package install path, default ./third_party"
  echo "    --output_path=<PATH>"
  echo "                      Set output path, default ./build_out"
  echo "    -u, --utest       Build and run all unit tests"
  echo "    --cov             Enable coverage"
  echo ""
}

# parse and set options
checkopts() {
  VERBOSE=""
  THREAD_NUM=$(grep -c ^processor /proc/cpuinfo)
  BUILD_TYPE="Release"
  VERSION=""

  OUTPUT_PATH="${BASEPATH}/build_out/"

  # Process the options
  parsed_args=$(getopt -a -o j:hvt -l help,verbose,pkg,utest,cov,build-type:,cann_3rd_lib_path:,output_path:, -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      -j)
        THREAD_NUM="$2"
        shift 2
        ;;
      -v | --verbose)
        VERBOSE="VERBOSE=1"
        shift
        ;;
      --cann_3rd_lib_path)
        if [ -d "$2" ]; then
          CANN_3RD_LIB_PATH="$(realpath $2)"
        else
          echo "Warning: Third lib path '$2' does not exist or is not a directory"
        fi
        shift 2
        ;;
       --pkg)
        ENABLE_PACKAGE=TRUE
        shift
        ;;
      --output_path)
        if [ -d "$2" ]; then
          OUTPUT_PATH="$(realpath $2)"
        else
          echo "Warning: Output path '$2' does not exist or is not a directory"
        fi
        shift 2
        ;;
      --build-type)
        BUILD_TYPE=$2
        shift 2
        ;;
      -u | --utest)
        ENABLE_TEST=TRUE
        shift
        ;;
      --cov)
        ENABLE_COVERAGE=TRUE
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

build() {
  echo "create build directory and build amct";
  if [ -f "${BUILD_PATH}" ]
  then
    echo "${BUILD_PATH} exist, delete old path"
    rm -rf ${BUILD_PATH}
  fi
  echo "create path ${BUILD_PATH}"
  mkdir ${BUILD_PATH}

  cd "${BUILD_PATH}"
  echo "----------------BUILD_PATH  "${BUILD_PATH}"  ----------------"
  echo "----------------TOP_DIR  "${BASEPATH}"  ----------------"
  echo "----------------CMAKE_BUILD_TYPE  "${CMAKE_BUILD_TYPE}"  ----------------"
  echo "----------------ASCEND_HOME_PATH  "${ASCEND_HOME_PATH}"  ----------------"
  echo "----------------CANN_3RD_LIB_PATH  "${CANN_3RD_LIB_PATH}"  ----------------"

  cmake -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -D TOP_DIR=${BASEPATH} \
        -D CMAKE_VERBOSE_MAKEFILE=ON\
        -D ASCEND_HOME_PATH=${ASCEND_HOME_PATH}\
        -D PREX=${LOWER_PREX}\
        ${CANN_3RD_LIB_PATH:+-D CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}} \
        -D BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
        -D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
        ..

  make all ${VERBOSE} -j${THREAD_NUM}
  if [ $? -ne 0 ]
  then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM}."
    return 1
  fi
  echo "Build success!"

  if [ -d "${OUTPUT_PATH}" ];then
    echo "${OUTPUT_PATH} exist, delete old path"
    rm -rf ${OUTPUT_PATH}
  fi
  echo "create path ${OUTPUT_PATH}"
  mkdir ${OUTPUT_PATH}
  mkdir "${OUTPUT_PATH}/amct_package"
  mkdir "${OUTPUT_PATH}/amct_package/amct_pytorch"
  cp ${BASEPATH}/dist/*.tar.gz ${OUTPUT_PATH}/amct_package/amct_pytorch/
  cp -r ${BASEPATH}/amctgraph/* ${OUTPUT_PATH}/amct_package/

  VERSION=$(awk -F'=' '/^Version=/ {print $2}' ../version.info)
  echo "package ${VERSION}"
  SYS="$(uname -s)"
  PACKAGE_NAME="cann-amct_${VERSION}_${SYS,,}-$(uname -m).tar.gz"
  TAR_FILE="${OUTPUT_PATH}/${PACKAGE_NAME}"
  TARGET_DIR="${OUTPUT_PATH}/amct_package"
  tar -czf ${TAR_FILE} -C ${TARGET_DIR} .
  echo "tar -czf ${TAR_FILE} -C ${TARGET_DIR} ."

  if [ -f ${TAR_FILE} ];then
    echo "package amct run success"
  else
    echo "package amct run failed"
    return 1
  fi

  rm -rf ${BASEPATH}/dist
  rm -rf ${BASEPATH}/amct_pytorch/*.egg-info
  rm -rf ${TARGET_DIR}
  rm -rf ${BASEPATH}/amctgraph
}

assemble_cmake_args() {
  if [[ "$ENABLE_ASAN" == "TRUE" ]]; then
    set +e
    echo 'int main() {return 0;}' | gcc -x c -fsanitize=address - -o asan_test >/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "This environment does not have the ASAN library, no need enable ASAN"
      ENABLE_ASAN=FALSE
    else
      $(rm -f asan_test)
      CMAKE_ARGS="$CMAKE_ARGS -DENABLE_ASAN=TRUE"
    fi
    set -e
  fi
  echo "$(uname -m)-$(uname -s)"
  PREX="$(uname -m)-$(uname -s)"
  echo "${PREX}"
  LOWER_PREX=$(echo "${PREX}" | tr '[:upper:]' '[:lower:]')
  echo "$LOWER_PREX"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PACKAGE=${ENABLE_PACKAGE}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_TEST=${ENABLE_TEST}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_UT_EXEC=${ENABLE_UT_EXEC}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_COVERAGE=${ENABLE_COVERAGE}"
  CMAKE_ARGS="$CMAKE_ARGS -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  CMAKE_ARGS="$CMAKE_ARGS -DTOP_DIR=${BASEPATH}"
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_VERBOSE_MAKEFILE=ON"
  CMAKE_ARGS="$CMAKE_ARGS -DASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
  CMAKE_ARGS="$CMAKE_ARGS -DPREX=${LOWER_PREX}"
  CMAKE_ARGS="$CMAKE_ARGS -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON"
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    rm -rf ${BUILD_PATH}/*
  fi
  if [ -d "${TMP_PATH}" ]; then
    rm -rf ${TMP_PATH}/*
  fi
}

clean_build_out() {
  if [ -d "${OUTPUT_PATH}" ]; then
    rm -rf ${OUTPUT_PATH}/*
  fi
}

build_package() {
  echo "--------------- build package start ---------------"
  bash install_graph.sh --cann_3rd_lib_path=${CANN_3RD_LIB_PATH}
  build || { echo "Build failed."; exit 1; }
  echo "--------------- build package end ---------------"
}

build_ut() {
  echo $dotted_line
  echo "Start to build ut"
  # clean_build

  git submodule init && git submodule update
  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  if [ ! -d "${TMP_PATH}" ]; then
    mkdir -p "${TMP_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  cmake --build . --target ${UT_TARGETS[@]} -- ${VERBOSE} -j $THREAD_NUM
}

main() {
  cd "${BASEPATH}"
  echo "----------------BASEPATH  ${BASEPATH}  ----------------"
  checkopts "$@"
  g++ -v
  assemble_cmake_args
  echo "CMAKE_ARGS: ${CMAKE_ARGS}"
  echo "----------------OUTPUT_PATH  ${OUTPUT_PATH}  ----------------"
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    build_package
  fi
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    build_ut
  fi
  echo "---------------- Build finished ----------------"
}

main "$@"
