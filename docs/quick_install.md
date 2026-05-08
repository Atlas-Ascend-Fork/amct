# 环境部署

使用AMCT工具之前，请先参考下面步骤完成基础环境搭建和源码下载，确保已经安装NPU固件、驱动和CANN软件（Ascend-cann-toolkit和Ascend-cann-ops）。

## 前置依赖

- bash >= 5.1.16
- GCC >= 7.3.x
- CMake >= 3.16.0（建议使用3.20.0版本）
- Python >= 3.9.x
   
  请确保该依赖已安装，注意满足版本要求。

- PyTorch：2.7.1、2.1.0
- Ascend Extension for PyTorch：版本配套关系请单击[Link](https://hiascend.com/document/redirect/pytorchuserguide)，查看“版本说明 >相关产品版本配套说明”章节。

此处以Python3.11、PyTorch2.7.1版本为例，安装业务运行时依赖的Python第三方库，安装命令如下，PyTorch2.1.0版本安装依赖命令请参见《[AMCT模型压缩工具](https://www.hiascend.com/document/redirect/CannCommunityToolAmctInstalDepdence)》：

```bash
pip3 install -r requirements.txt
```

## 环境安装

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

安装完CANN包后，需验证环境和驱动是否正常。

- **检查NPU设备** 

    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```

- **检查CANN版本**

   ```bash
   # 查看CANN Toolkit的version字段提供的版本信息（默认路径安装），<arch>表示CPU架构（aarch64或x86_64）。
   cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
   # 查看CANN ops的version字段提供的版本信息（默认路径安装），<opsname>表示待查询的ops子包的名称，请用户根据实际安装路径替换。
   cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
   ```

## 环境变量配置

按需选择合适的命令使环境变量生效：

  ```bash
  # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}） 
  source /usr/local/Ascend/cann/set_env.sh
  # 指定路径安装
  source ${install_path}/cann/set_env.sh
  ```
