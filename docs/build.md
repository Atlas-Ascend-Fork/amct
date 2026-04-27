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

- 安装[编译](#编译)环节生成的run包（如果安装用户为root，请将安装命令中的--user删除）。

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
