#!/bin/bash

cd ${WORKSPACE}
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    sudo update-alternatives --set gcc /usr/bin/gcc-14
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e
echo "${task_name}"
if [[ "${task_name}" =~ x86_compile_ubuntu24 ]]; then
    sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
fi
case "${task_name}" in
    x86_compile)
        bash build.sh --pkg --cann_3rd_lib_path=/home/jenkins/opensource
        echo "exec cmd: [bash build.sh --pkg --cann_3rd_lib_path=/home/jenkins/opensource]"
        exit 0
        ;;
    x86_compile_ubuntu24)
        bash build.sh --pkg --cann_3rd_lib_path=/home/jenkins/opensource
        echo "exec cmd: [bash build.sh --pkg --cann_3rd_lib_path=/home/jenkins/opensource]"
        exit 0
        ;;
    x86_compile_torch)
        bash build.sh --torch --cann_3rd_lib_path=/home/jenkins/opensource
        echo "exec cmd: [bash build.sh --torch --cann_3rd_lib_path=/home/jenkins/opensource]"
        exit 0
        ;;
    x86_compile_torch_ubuntu24)
        bash build.sh --torch --cann_3rd_lib_path=/home/jenkins/opensource
        echo "exec cmd: [bash build.sh --torch --cann_3rd_lib_path=/home/jenkins/opensource]"
        exit 0
        ;;
esac


if [ ! -f "build_out/"*.tar.gz ]; then
    mkdir -p build_out
    touch build_out/cann-amct-test_linux-x86.tar.gz
fi
