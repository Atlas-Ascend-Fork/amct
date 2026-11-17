# Source Code Build

<!-- md-trans-meta sourceCommit=4a92849f6e2ef6c55f6296f00eb841798530bb79 translatedAt=2026-10-16T07:50:43.843Z pushedAt=2026-11-05T01:12:03.397Z nl=2 -->

## Environment Setup
 
Before compiling the source code, complete the basic environment setup first. For details, see [Quick Installation](quick_install.md).

## Source Code Compilation

### Downloading the Source Code

Developers can download the source code of this repository by running the following command:

   ```bash
   # Download the project source code, using the master branch as an example. 
   git clone https://gitcode.com/cann/amct.git 
   ``` 

### Compilation

Go to the project root directory and run the following compilation command to generate the **amct_pytorch** installation package:

```bash
bash build.sh --torch
```

After successful compilation, `amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz` is generated in the **build_out** directory under the project root directory.

- **${version}** indicates the version number.
- **${arch}** indicates the CPU architecture, such as **aarch64** and **x86_64**.

For more features, visit https://hiascend.com/document/redirect/CannCommunityToolAmct.

### Local Verification

Use the test cases in the **tests** directory for local verification:

- Install dependencies.

  ```bash
  # Install test dependencies.
  pip3 install pytest-cov
  ```

- Run the test cases:

  ```bash
  bash build.sh -u
  ```

  For more execution options, use **-h**:

  ```bash
  bash build.sh -h
  ```

### Installation and Uninstallation

- Install the installation package generated in the [Compilation](#compilation) phase. (If the installation user is **root**, remove **--user** from the installation command.)

  ```bash
  pip3 install build_out/amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz --user
  ```

  > [!NOTE] Note
  > If you perform installation with `--no-build-isolation`, pip does not automatically install the build dependencies. Run `pip install wheel` first, and then append `--no-build-isolation` to the installation command above. Otherwise, the error "error: invalid command 'bdist_wheel'" may occur.

- Uninstall the package.

  ```bash
  pip3 uninstall amct_pytorch
  ```

 After the installation is complete, you can run samples by referring to [Sample Running](../../examples/README.md).
