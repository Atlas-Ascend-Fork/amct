# Tool Build

## Environment Preparation

Before using the AMCT tool, please complete the basic environment setup and source code download by following the steps below. Ensure that NPU firmware, drivers, and CANN software (Ascend-cann-toolkit and Ascend-cann-ops) are installed.

### Prerequisites

- Python: 3.11
    
  Please ensure this dependency is installed and meets the version requirements.

- PyTorch: 2.7.1, 2.1.0
- Ascend Extension for PyTorch: For version compatibility information, please click [Link](https://hiascend.com/document/redirect/pytorchuserguide) and refer to the "Version Description > Related Product Version Compatibility" section.

Taking PyTorch 2.7.1 as an example, install the Python third-party libraries required for runtime operations using the following command. For PyTorch 2.1.0 dependency installation commands, please refer to the [AMCT Model Compression Tool](https://www.hiascend.com/document/redirect/CannCommunityToolAmctInstalDepdence):

```bash
pip3 install -r requirements.txt
```

### Software Installation

1. **Install Drivers and Firmware**

   When performing quantization calibration operations, drivers and firmware must be installed. For download and installation instructions, please refer to the "Prepare Software Packages" and "Install NPU Drivers and Firmware" sections in the [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

2. **Install CANN Packages**    

   **Scenario 1: Experience master version capabilities or develop based on the master version**

     Please click [Download Link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master) to get the latest time version, and download the corresponding package based on the product model and environment architecture. The installation commands are as follows. For more guidance, please refer to the [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

     1. Install the CANN Toolkit development kit package.

        ```bash
        # Ensure the installation package has executable permissions
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # Installation command
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

     2. Install the CANN ops operator package (optional).

        The ops operator package must be installed before running the quantized deployment model. If only compiling AMCT, this package can be skipped.

        ```bash
        # Ensure the installation package has executable permissions
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # Installation command
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

       - `${cann_version}`: Represents the CANN package version number.
       - `${arch}`: Represents the CPU architecture, such as `aarch64`, `x86_64`.
       - `${soc_name}`: Represents the NPU model name.
       - `${install_path}`: Represents the specified installation path. ops must be installed in the same path as as the Toolkit package. For root users, it is installed in the `/usr/local/Ascend` directory by default.
    
   **Scenario 2: Experience released version capabilities or develop based on released versions**

   If you want to experience the capabilities of the **officially released CANN packages from the website**, please visit the [CANN Official Download Center](https://www.hiascend.com/cann/download) and select the corresponding version of the CANN software package (only CANN 8.5.0 and later versions are supported) for installation.   

## Environment Verification

After installing the CANN packages, you need to verify that the environment is normal.

```bash
# View the version information provided by the version field of CANN Toolkit (default path installation), <arch> represents CPU architecture (aarch64 or x86_64).
cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
# View the version information provided by the version field of CANN ops (default path installation), <opsname> represents the name of the ops subpackage to be queried, please replace according to the actual installation path.
cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
```

## Environment Variable Configuration

Select the appropriate command based on your actual scenario:

  ```bash
  # Default path installation, taking root user as an example (for non-root users, replace /usr/local with ${HOME})
  source /usr/local/Ascend/cann/set_env.sh
  # Specified path installation
  source ${install_path}/cann/set_env.sh
  ```

## Source Code Compilation

### Download Source Code

   ```bash
   # Download project source code, taking master branch as an example
   git clone https://gitcode.com/cann/amct.git
   ```

### Source Code Compilation

Navigate to the project root directory and execute the following compilation command:

```bash
bash build.sh --pkg
```

After successful compilation, `cann-amct_${version}_linux-${arch}.tar.gz` will be generated in the build_out directory under the project root directory.

- ${version} represents the version number.
- ${arch} represents the CPU architecture, such as aarch64, x86_64.

### Local Verification

Perform local verification using test cases in the tests directory:

- Install dependencies

  ```bash
  # Install test framework dependencies
  pip3 install coverage
  ```

- Execute test cases:

  ```bash
  bash build.sh -u
  ```

  For more execution options, use -h to view:

  ```bash
  bash build.sh -h
  ```

### Installation and Uninstallation

- Install the run package generated in the [Source Code Compilation](#source-code-compilation) step (if the installation user is root, please remove --user from the installation command).

  ```bash
  tar -zxvf cann-amct_${version}_linux-${arch}.tar.gz
  cd amct_pytorch && pip3 install amct_pytorch_${version}-linux-${arch}.tar.gz --user
  ```

  > [!NOTE]Note
  > When installing the AMCT tool, please ensure that the pip version is <= 25.2, otherwise the error message "ModuleNotFoundError:No module named 'torch'" may appear; if the user's pip version is > 25.2 and does not want to downgrade, please append `--no-build-isolation` to the above installation command.

- Uninstall

  ```bash
  pip3 uninstall amct_pytorch
  ```

 After installation is complete, you can refer to [Example Execution](../examples/README.md) to run the examples.
