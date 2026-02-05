# 工具构建

## 环境准备

使用AMCT工具之前，请根据如下步骤完成相关环境的搭建。

1. **安装依赖**

   该工具源码编译用到的依赖如下，请注意版本要求：

   - PyTorch：2.7.1、2.1.0

   - Python：3.11

   - Ascend Extension for PyTorch：版本配套关系请单击[Link](https://hiascend.com/document/redirect/pytorchuserguide)，查看“版本说明 >相关产品版本配套说明”章节。

2. **安装固件和驱动**

   执行量化校准操作时，必须安装驱动和固件，安装指导详见[《CANN软件安装指南》](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)。

3. **安装社区尝鲜版CANN Toolkit包**

   根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，下载链接为[toolkit x86_64包](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0/x86_64/Ascend-cann-toolkit_8.5.0_linux-x86_64.run)、[toolkit aarch64包](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0/aarch64/Ascend-cann-toolkit_8.5.0_linux-aarch64.run)。

   ```bash
   # 确保安装包具有可执行权限
   chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run 
   # 安装命令(其中--install-path为可选)
   ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --force --install-path=${install_path}
   ```

   -   $\{cann\_version\}：表示CANN包版本号。
   -   $\{arch\}：表示CPU架构，如aarch64、x86\_64。
   -   $\{install\_path\}：表示指定安装路径，可选，默认安装在/usr/local/Ascend目录，指定路径安装时，指定的路径权限需设置为755。

4. **安装社区尝鲜版CANN ops包**
   运行量化部署模型前必须安装本包，若仅编译AMCT，可跳过本操作。


5. **配置环境变量**

   根据实际场景，选择合适的命令。

   ```bash
   # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
   source /usr/local/Ascend/cann/set_env.sh 
   # 指定路径安装
   source ${install_path}/cann/set_env.sh
   ```

6. **安装后配置**
   CANN包安装完成后，需安装业务运行时依赖的Python第三方库（如果使用root用户安装，请将命令中的--user删除）。
   ```bash
   pip3 install attrs cython 'numpy>=1.19.2,<=1.24.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py --user
   ```

7. **源码下载**

   ```bash
   # 下载项目源码，以master分支为例 
   git clone https://gitcode.com/cann/amct.git 
   ```

## 编译执行

进入项目根目录，执行如下编译命令：

```bash
bash build.sh --pkg
```

编译成功后，会在项目根目录的build_out目录下生成`cann-amct_${version}_linux-${arch}.tar.gz`。

- ${version}表示版本号。
- ${arch}表示表示CPU架构，如aarch64、x86_64。

## 本地验证

利用tests路径下的测试用例进行本地验证：

- 安装依赖

  ```bash
  # 安装测试目录requirements.txt依赖 
  cd tests && pip3 install -r requirements.txt
  ```

- 执行测试用例：

  ```bash
  bash build.sh -u
  ```

  更多执行选项可以用 -h 查看：

  ```bash
  bash build.sh -h
  ```

## 安装与卸载

- 安装[编译执行](#编译执行)环节生成的run包（如果安装用户为root，请将安装命令中的--user删除）。

  ```bash
  tar -zxvf cann-amct_${version}_linux-${arch}.tar.gz
  cd amct_pytorch && pip3 install amct_pytorch_${version}-linux-${arch}.tar.gz --user
  ```

- 卸载

  ```bash
  pip3 uninstall amct_pytorch
  ```

 安装完成后，可以参考[样例运行](../examples/README.md)运行样例。

