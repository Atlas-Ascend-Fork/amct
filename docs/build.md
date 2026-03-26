# 工具构建

## 环境准备

使用AMCT工具之前，请先参考下面步骤完成基础环境搭建和源码下载，确保已经安装NPU固件、驱动和CANN软件（Ascend-cann-toolkit和Ascend-cann-ops）。

### 前置依赖

- Python：3.11
   
  请确保该依赖已安装，注意满足版本要求。

- PyTorch：2.7.1、2.1.0
- Ascend Extension for PyTorch：版本配套关系请单击[Link](https://hiascend.com/document/redirect/pytorchuserguide)，查看“版本说明 >相关产品版本配套说明”章节。

此处以PyTorch2.7.1版本为例，安装业务运行时依赖的Python第三方库，安装命令如下，PyTorch2.1.0版本安装依赖命令请参见《[AMCT模型压缩工具](https://www.hiascend.com/document/redirect/CannCommunityToolAmctInstalDepdence)》：

```bash
pip3 install -r requirements.txt
```

### 软件安装

1. **安装驱动与固件**

   执行量化校准操作时，必须安装驱动和固件，下载和安装操作请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中“准备软件包”和“安装NPU驱动和固件”章节。

2. **安装CANN包**    

   **场景1：体验master版本能力或基于master版本进行开发**

     请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)获取最新时间版本，并根据产品型号和环境架构下载对应包。安装命令如下，更多指导请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)。

     1. 安装CANN Toolkit开发套件包。

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # 安装命令
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

     2. 安装CANN ops算子包（可选）。

        运行量化部署模型前必须安装ops算子包，若仅编译AMCT，可不安装此包。

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
        # 安装命令
        ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

      - `${cann_version}`：表示CANN包版本号。
      - `${arch}`：表示CPU架构，如`aarch64`、`x86_64`。
      - `${soc_name}`表示NPU型号名称。
      - `${install_path}`：表示指定安装路径，ops需要与Toolkit包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录。
   
   **场景2：体验已发布版本能力或基于已发布版本进行开发**

    如果您想体验**官网正式发布的CANN包**能力，请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，选择对应版本CANN软件包（仅支持CANN 8.5.0及后续版本）进行安装。   

## 环境验证

安装完CANN包后，需验证环境是否正常。

```bash
# 查看CANN Toolkit的version字段提供的版本信息（默认路径安装），<arch>表示CPU架构（aarch64或x86_64）。
cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
# 查看CANN ops的version字段提供的版本信息（默认路径安装），<opsname>表示待查询的ops子包的名称，请用户根据实际安装路径替换。
cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
```

## 环境变量配置

根据实际场景，选择合适的命令：

  ```bash
  # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}） 
  source /usr/local/Ascend/cann/set_env.sh
  # 指定路径安装
  source ${install_path}/cann/set_env.sh
  ```

## 源码编译

### 下载源码

   ```bash
   # 下载项目源码，以master分支为例 
   git clone https://gitcode.com/cann/amct.git 
   ``` 

### 源码编译

进入项目根目录，执行如下编译命令：

```bash
bash build.sh --pkg
```

编译成功后，会在项目根目录的build_out目录下生成`cann-amct_${version}_linux-${arch}.tar.gz`。

- ${version}表示版本号。
- ${arch}表示表示CPU架构，如aarch64、x86_64。

### 本地验证

利用tests路径下的测试用例进行本地验证：

- 安装依赖

  ```bash
  # 安装测试框架依赖 
  pip3 install coverage
  ```

- 执行测试用例：

  ```bash
  bash build.sh -u
  ```

  更多执行选项可以用 -h 查看：

  ```bash
  bash build.sh -h
  ```

### 安装与卸载

- 安装[源码编译](#源码编译)环节生成的run包（如果安装用户为root，请将安装命令中的--user删除）。

  ```bash
  tar -zxvf cann-amct_${version}_linux-${arch}.tar.gz
  cd amct_pytorch && pip3 install amct_pytorch_${version}-linux-${arch}.tar.gz --user
  ```

  > [!NOTE]说明
  > 安装AMCT工具时，请确保pip版本<=25.2，否则可能出现“ModuleNotFoundError:No module named 'torch' ”错误信息；如果用户pip版本>25.2，且不想降低版本，则请在上述安装命令后追加`--no-build-isolation` 。

- 卸载

  ```bash
  pip3 uninstall amct_pytorch
  ```

 安装完成后，可以参考[样例运行](../examples/README.md)运行样例。
