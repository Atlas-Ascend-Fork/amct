# Environment Deployment
<!-- md-trans-meta sourceCommit=65bf5a94c11d2bea49c717f3358af979012aba0b translatedAt=2026-10-16T07:58:59.624Z pushedAt=2026-11-05T01:12:03.401Z nl=2 -->

Before using the AMCT tool, refer to the following steps to complete the basic environment setup and source code download, and ensure that the NPU firmware, driver, and CANN software (**Ascend-cann-toolkit** and **Ascend-cann-ops**) are installed.

## Environment Installation

This project provides multiple methods for setting up an Ascend environment. Select one as needed.

> **Note**: The compilation mode and runtime mode mentioned in this document are defined as follows. Select the appropriate one based on your actual situation.
>
> - Compilation mode: For scenarios where you only compile this project without running it, you only need to install the CANN toolkit package.
> - Runtime mode: For scenarios where you run this project (compile and run, or run only), you need to install the driver and firmware, the CANN toolkit package, and the CANN ops package.

|  Installation Method  |  Usage  |  Usage Scenario  |
| ----- | ------ | ------ |
|  WebIDE  | A one-stop development platform that provides an online, directly runnable Ascend environment without manual installation.<br>Currently, single-node computing power is provided and **the latest commercial CANN package is installed by default**. | Suitable for developers without Ascend devices.|
|  Docker  | A Docker image is an efficient deployment method that comes pre-integrated with the CANN package and required dependencies.<br>Currently applicable only to Atlas A2 products, with Ubuntu as the only supported OS. **The latest commercial CANN package is installed by default**. |Suitable for developers who have Ascend devices and need to set up an environment quickly.|
|  Manual Installation  | Manually install the CANN package and basic dependencies, offering high flexibility. |Suitable for developers who have Ascend devices and want to experience manual installation of the CANN package or the latest master branch capabilities.|

### Method 1: WebIDE Environment

Developers without Ascend devices can directly use the WebIDE development platform, that is, the "**one-stop development platform**". This platform provides you with an online, ready-to-run Ascend environment in which the required drivers, firmware, software packages, and dependencies are already installed, eliminating the need for manual installation.

> **Note**: The environment installs the latest commercial CANN package by default. When downloading the source code, ensure that it is compatible with the software. For more information about the development platform, see [WebIDE Guide](https://gitcode.com/org/cann/discussions/54).

1. Enter the open-source project, click the cloud development button, and log in with an authenticated HUAWEI CLOUD account. If you have not registered or authenticated, register and authenticate as prompted on the page.

   <img src="./figures/cloudIDE.png" alt="Cloud Platform"  width="750px" height="90px">

2. Create an NPU environment and configure the specifications as prompted on the page. After the cloud development environment is started, choose "`Connect > WebIDE`" to enter the one-stop development platform.
  
   The resources of the current open-source project are stored in the `/mnt/workspace/gitCode/${gitCode_id}` directory by default, where **\$\{gitCode\_id\}** indicates the developer's personal gitCode account.

   <img src="./figures/webIDE.png" alt="Cloud Platform"  width="1000px" height="150px">

### Method 2: Docker Deployment

For developers with Ascend devices, if you want to quickly set up an Ascend environment, you can use a Docker image for deployment.

> **Note**:
>
> - The image file is relatively large, so downloading it takes some time. Please wait patiently. For an introduction to Docker command options, run `docker --help` to query them.
> - The environment installs the latest commercial CANN package by default. When downloading the source code, ensure that it is compatible with the software version.

1.**Install the driver and firmware (runtime dependency)**.

For downloading and installing the Ascend driver and firmware on the host, see the "Software Package Preparation" and "Installing the NPU Driver and Firmware" chapters in the [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

2.**Download the image**.

- Step 1: Log in to the host as the **root** user. Ensure that the Docker engine (version 1.11.2 or later) is installed on the host.
- Step 2: Pull the image that has the CANN software package and the dependencies required by `amct` pre-integrated from the [Ascend Image Repository](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884). The commands are as follows. Select one based on the actual architecture:

    ```bash
    # Example: Pull the CANN development image for the Arm architecture.
    docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    # Example: Pull the CANN development image for the x86 architecture.
    docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    ```

3.**Run Docker**.

After pulling the image, start the container with specific parameters so that the container can access the Ascend devices on the host.

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```

| Parameter | Description | Precautions |
| :--- | :--- | :--- |
| `--name cann_container` | Specifies a name for the container for easier management. | Customizable. |
| `--device /dev/davinci0` | Core: Maps the host's NPU device card into the container. Multiple NPU devices can be mapped. | Must be adjusted based on the actual configuration: `davinci0` corresponds to NPU 0 in the system. Run the `npu-smi info` command on the host first, and modify this number based on the device numbers shown in the output (such as `NPU 0` and `NPU 1`). |
| `--device /dev/davinci_manager` | Maps the NPU device management interface. | - |
| `--device /dev/devmm_svm` | Maps the device memory management interface. | - |
| `--device /dev/hisi_hdc` | Maps the communication API between the host and the device. | - |
| `-v /usr/local/dcmi:/usr/local/dcmi` | Mounts the tools and libraries related to the Device Container Management Interface (DCMI). | - |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | Mounts the `npu-smi` tool. | Allows the container to run this command directly to query NPU status and performance information. |
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | Key mount: Maps the host's NPU driver libraries into the container. | - |
| `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | Mounts the driver version information file. | - |
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | Mounts the CANN software installation information file. | - |
| `-it` | A combination of `-i` (interactive) and `-t` (allocate a pseudo-terminal). | - |
| `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops` | Specifies the Docker image to run. | Ensure that the image name and tag exactly match the image pulled via `docker pull`. |
| `bash` | The command to execute immediately after the container starts. | - |

### Method 3: Manual Installation

If you have Ascend devices and want to manually set up the Ascend environment, refer to the following steps.

#### Prerequisites

- bash >= 5.1.16
- GCC >= 7.3.x
- CMake >= 3.16.0 (3.20.0 is recommended)
- patch >= 2.7
- Python >= 3.9.x

  Ensure that this dependency is installed and meets the version requirement.

- PyTorch: 2.7.1, 2.1.0

- TorchNPU: For version compatibility, click [here](https://hiascend.com/document/redirect/pytorchuserguide) and see the "Release Notes > Related Product Version Mapping" chapter.

### PyTorch + TorchNPU Version Compatibility Installation

When using the NPU acceleration backend, PyTorch must be installed as the CPU version and match the `TorchNPU` version. Do not mix a non-CPU version of PyTorch with `TorchNPU` in the same Python environment; otherwise, accelerator conflicts may occur at runtime.

It is recommended to install through either of the following paths:

- Path A: Refer to the "Release Notes > Related Product Version Mapping" chapter in the Ascend official [PyTorch User Guide](https://hiascend.com/document/redirect/pytorchuserguide), and download and install the matching CPU-version PyTorch wheel and `TorchNPU` wheel.
- Path B: Use the PyTorch CPU wheel index to install the CPU version of PyTorch, and then install the matching `TorchNPU`.

  ```bash
  pip3 uninstall -y torch
  pip3 install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
  pip3 install torch_npu==2.7.1.post4
  ```

After installation, run the following command to confirm that the CPU version of PyTorch is installed in the current environment:

```bash
python3 -c "import torch; assert '+cpu' in torch.__version__, 'must use CPU torch for NPU path'"
```

The following example uses Python 3.11 and PyTorch 2.7.1 to install the Python third-party libraries required by the service runtime. The installation command is as follows. `requirements.txt` has pinned the CPU version of PyTorch through the PyTorch CPU wheel index. For the dependency installation command for PyTorch 2.1.0, see [Ascend Model Compression Toolkit (AMCT)](https://www.hiascend.com/document/redirect/CannCommunityToolAmctInstalDepdence):

```bash
pip3 install -r requirements.txt
```

#### Environment Installation

1. **Install the driver and firmware**.

   When performing quantization calibration, you must install the driver and firmware. For download and installation, see the "Software Package Preparation" and "Installing the NPU Driver and Firmware" chapters in [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

2. **Install the CANN package**.

   **Scenario 1: Experience master version capabilities or perform development based on the master version**.

     Click [here](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master) to obtain the latest version, and download the corresponding package based on the product model and environment architecture. The installation command is as follows. For more guidance, see [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

     1. Install the CANN Toolkit package.

        ```bash
        # Ensure that the installation package has executable permissions.
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # Installation command.
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

     2. (Optional) Install the CANN ops operator package.

        The ops operator package must be installed before running a quantized deployment model. If you only compile AMCT, you can skip installing this package.

        ```bash
        # Ensure that the installation package has the executable permission.
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # Installation command.
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

      - `${cann_version}`: Indicates the CANN package version number.
      - `${arch}`: Indicates the CPU architecture, such as `aarch64` and `x86_64`.
      - `${soc_name}`: Indicates the SoC code of the NPU, which must match the actual product series of the local machine (for example, `910b` for Atlas A2 products, `A3` for Atlas A3 products, and `950` for Ascend 950PR&950DT products).
      - `${install_path}`: Indicates the specified installation path. The ops package must be installed in the same path as the Toolkit package. For the **root** user, the default installation path is `/usr/local/Ascend`.

      **How to query the local soc_version** (that is, the actual configuration value of the chip model; use the `npu-smi` query result as the final reference):

      - For Atlas A2 products: Run the `npu-smi info` command on the server where the AI processor is installed to query and obtain the **Name** information. The actual configuration value is `Ascend${Name}`. For example, if the **Name** value is `xxxyy`, the actual configuration value is `Ascendxxxyy`.
      - For Atlas A3 products: Run the `npu-smi info -t board -i id -c chip_id` command on the server where the AI processor is installed to query and obtain the **Chip Name** and **NPU Name** information. The actual configuration value is `Chip Name_NPU Name`. For example, if the **Chip Name** value is `Ascendxxx` and the **NPU Name** value is `1234`, the actual configuration value is `Ascendxxx_1234`. Where:
        - **id**: Device ID. The NPU ID obtained through the `npu-smi info -l` command is the device ID.
        - **chip_id**: Chip ID. The **Chip ID** obtained through the `npu-smi info -m` command is the chip ID.
      - For Ascend 950PR&950DT products: Run the `npu-smi info -t board -i id` command on the server where the AI processor is installed to query and obtain the **Chip Name** and **NPU Name** information. The actual configuration value is `Chip Name_NPU Name`. For example, if the **Chip Name** value is `Ascendxxx` and the **NPU Name** value is `1234`, the actual configuration value is `Ascendxxx_1234`. Here, **id** indicates the device ID, which is the NPU ID obtained through the `npu-smi info -l` command.

      > **Note**: For Atlas A3 products, the **soc_version** value is `Ascend910_93xx`. For the specific value, refer to the `npu-smi` query result above. After confirming that the product is an Atlas A3 product, select `A3` for `${soc_name}` in the ops package (that is, `Ascend-cann-A3-ops_${cann_version}_linux-${arch}.run`).

   **Scenario 2: Experience the capabilities of a released version or perform development based on a released version**.

    If you want to experience the capabilities of the **CANN package officially released on the official website**, visit the [download center on the official CANN website](https://www.hiascend.com/cann/download) and select the corresponding CANN software package (only CANN 8.5.0 and later versions are supported) for installation.

#### Environment Verification

After installing the CANN package, verify that the environment and driver are working properly.

- **Check the NPU device**.

    ```bash
    # Run npu-smi. If the device information is displayed normally, the driver is working properly.
    npu-smi info
    ```

- **Check the CANN version**.

   ```bash
   # View the version information provided by the version field of CANN Toolkit (installed in the default path). <arch> indicates the CPU architecture (aarch64 or x86_64).
   cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
   # View the version information provided by the version field of CANN ops (installed in the default path). <opsname> indicates the name of the ops subpackage to be queried. Replace it with the actual installation path.
   cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
   ```

#### Environment Variable Configuration

Select an appropriate command as needed to make the environment variables take effect:

  ```bash
  # Default path installation, using the root user as an example (for non-root users, replace /usr/local with ${HOME}).
  source /usr/local/Ascend/cann/set_env.sh
  # Specified path installation.
  source ${install_path}/cann/set_env.sh
  ```
