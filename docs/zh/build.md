# 源码构建

## 环境准备
 
在源码编译前，请先完成基础环境搭建，具体操作请参见[快速安装](quick_install.md)。

## 源码编译

### 下载源码

开发者可通过如下命令下载本仓源码：

   ```bash
   # 下载项目源码，以master分支为例 
   git clone https://gitcode.com/cann/amct.git 
   ``` 

### 编译

进入项目根目录，执行如下编译命令，生成 amct_pytorch 安装包：

```bash
bash build.sh --torch
```

编译成功后，会在项目根目录的build_out目录下生成`amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz`。

- ${version}表示版本号。
- ${arch}表示CPU架构，如aarch64、x86_64。

更多特性请参考：https://hiascend.com/document/redirect/CannCommunityToolAmct

### 离线编译

若您的编译环境无法访问网络，可以先在有网环境执行脚本一键下载所有三方件，再将整个目录拷贝到离线环境：

```bash
bash download_3rd_party.sh
# 或指定下载目录
bash download_3rd_party.sh --3rd_party third_party
```

说明：

- 未指定参数时，默认下载到仓库根目录的 `third_party` 目录；三方件压缩包直接存放于该目录，所有 patch 存放于其 `patch` 子目录。
- 每个文件均通过 SHA256 校验；已存在的缓存会先校验，一致则跳过下载，不一致则重新下载；下载或校验失败时脚本返回非零退出码。
- 下载列表与 [cann/cmake master-059](https://gitcode.com/cann/cmake/tree/master-059/third_party) 的三方件定义保持一致，下载地址和 SHA256 固定在 `download_3rd_party.sh` 中。若从其他地址下载，请确保版本号和 SHA256 一致。

下载完成后，编译时将 `--cann_3rd_lib_path` 指向相同目录即可：

```bash
bash build.sh --torch --cann_3rd_lib_path=third_party
```

### 本地验证

利用tests路径下的测试用例进行本地验证：

- 安装依赖

  ```bash
  # 安装测试依赖
  pip3 install pytest-cov
  ```

- 执行测试用例：

  ```bash
  bash build.sh -u
  ```

  更多执行选项可以用 -h 查看：

  ```bash
  bash build.sh -h
  ```

### 依赖的开源第三方软件

amct 在编译时，依赖的第三方开源软件列表如下：

| 开源软件 | 版本 | 下载地址 |
|---|---|---|
| protobuf | 25.1 | [protobuf-25.1.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/protobuf/protobuf-25.1.tar.gz) |
| abseil-cpp | 20230802.1 | [abseil-cpp-20230802.1.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/abseil-cpp/abseil-cpp-20230802.1.tar.gz) |
| abseil-cpp 补丁 | 20230802.1-h0 | [patch/backport-CVE-2025-0838.patch](https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1-h0/backport-CVE-2025-0838.patch) |
| cann-cmake | master-059 | [cmake-master-059.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-059.tar.gz) |

> [!NOTE]说明
> protobuf 编译所需的 `protobuf_25.1_change_version.patch`、`protobuf-hide_absl_symbols.patch`、`fix-gcc15-gcc16-container-memory-cstdint.patch` 等其余补丁已收录于 cann-cmake 的 `third_party/patch` 目录，随 `cmake-master-059.tar.gz` 一并获取，无须单独下载。

### 安装与卸载

- 安装[编译](#编译)环节生成的安装包（如果安装用户为root，请将安装命令中的--user删除）。

  ```bash
  pip3 install build_out/amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz --user
  ```

  > [!NOTE]说明
  > 若使用`--no-build-isolation`安装，pip不会自动安装构建依赖。请先执行`pip install wheel`，再在上述安装命令后追加`--no-build-isolation`，否则可能出现“error: invalid command 'bdist_wheel'”错误信息。

- 卸载

  ```bash
  pip3 uninstall amct_pytorch
  ```

 安装完成后，可以参考[样例运行](../../examples/README.md)运行样例。
