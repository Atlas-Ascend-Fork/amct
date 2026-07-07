# Security Statement

## User Running Recommendations

From a security perspective, it is not recommended to use root or other administrator-type accounts to execute any commands. Follow the principle of least privilege.

## File Permission Control

- Recommend users set the running system umask value to 0027 or higher on the host (including host machine) and in containers, ensuring that new folder default maximum permission is 750 and new file default maximum permission is 640.
- Recommend users implement permission control and other security measures for sensitive content such as personal privacy data, business assets, source files, and various files saved during development. For example, permission control for this project's installation directory, input public data file permission control. Recommended permissions should refer to [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario).
- During runtime, compilation files may be cached and stored in the `kernel_meta_*` folder in the running directory to speed up subsequent calls. Users can implement permission control on generated related files as needed.
- Users need to implement permission control during installation and use. Recommend referring to [A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario](#a-file-folder-permission-control-recommended-maximum-values-for-each-scenario) file permission reference for settings.

## Build Security Statement

When compiling and installing this project from source code, you need to compile yourself. The compilation process generates some intermediate files. Recommend you implement permission control on intermediate files after compilation to ensure file security.

## Runtime Security Statement

- Recommend users write corresponding calling scripts based on runtime environment resource status. If calling scripts do not match resource status, such as generating input data or benchmark calculation results using space exceeding memory capacity limit, scripts saving data locally exceeding disk space size, etc., may trigger errors and cause process unexpected exit.
- When AMCT runtime encounters exceptions, it will exit the process and print error information. Recommend locating specific error cause based on error prompts, including setting methods such as viewing log files.
- When AMCT calls through [PyTorch](https://gitcode.com/Ascend/pytorch), may encounter runtime errors due to version mismatch. Please refer to [PyTorch Security Statement](https://gitcode.com/Ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Public Network Address Statement
The public network addresses contained in this project code are declared as follows:

| Type | Open Source Code Address | File Name　　　　　　　　　　　　　 | Public Network IP Address/Public Network URL Address/Domain Name/Email Address/Compressed File Address　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | Usage Description　　　　　　　　　　　　　　　　　|
| :----:| :------------:| :---------------------------------| :--------------------------------------------------------------------------------------------------------------| :------------------------------------------|
| Dependency | Not involved　　　 | cmake/third_party/protobuf.cmake | https://gitcode.com/cann-src-third-party/protobuf/releases/download/v3.13.0/protobuf-3.13.0.tar.gz　　　　　　| Download protobuf source code from gitcode, serves as compilation dependency　 |
| Dependency | Not involved　　　 | cmake/third_party/protobuf.cmake | https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz | Download abseil source code from gitcode, serves as compilation dependency　　 |
| Dependency | Not involved　　　 | cmake/fetch_cann_cmake.cmake　　 | https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-017.tar.gz　　　　　　　　　　　　　　　 | Download cann cmake source code from gitcode, serves as compilation dependency |
---

## Vulnerability Mechanism Description
[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-File (Folder) Permission Control Recommended Maximum Values for Each Scenario

| Type | Linux Permission Reference Maximum Value |
| --- | --- |
| User Home Directory | 750 (rwxr-x---) |
| Program Files (including script files, library files, etc.) | 550 (r-xr-x---) |
| Program File Directory | 550 (r-xr-x---) |
| Configuration File | 640 (rw-r-----) |
| Configuration File Directory | 750 (rwxr-x---) |
| Log File (finished recording or already archived) | 440 (r--r-----) |
| Log File (currently recording) | 640 (rw-r-----) |
| Log File Directory | 750 (rwxr-x---) |
| Debug File | 640 (rw-r-----) |
| Debug File Directory | 750 (rwxr-x---) |
| Temporary File Directory | 750 (rwxr-x---) |
| Maintenance Upgrade File Directory | 770 (rwxrwx---) |
| Business Data File | 640 (rw-r-----) |
| Business Data File Directory | 750 (rwxr-x---) |
| Key Component, Private Key, Certificate, Ciphertext File Directory | 700 (rwx---) |
| Key Component, Private Key, Certificate, Encrypted Ciphertext | 600 (rw-------) |
| Encryption/Decryption Interface, Encryption/Decryption Script | 500 (r-x------) |