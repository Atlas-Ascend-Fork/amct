# MXFP4 Ascend-C 加速算子

在 Ascend 910B3 NPU 上实现 MXFP4（Microscaling FP4 E2M1）伪量化算子的 Ascend-C 自定义 kernel，对比 torch_npu 实现 **3.3x 加速**（大矩阵），**18x 加速**（小矩阵）。正确性与 PyTorch 参考实现 bit-exact 一致。

## 运行环境

| 组件 | 版本 |
|------|------|
| 硬件 | Ascend 910B3 |
| CANN | 8.2.RC1 |
| Python | 3.10 (aarch64) |
| PyTorch | 2.6.0 |
| torch_npu | 2.6.0.post4 |

> 开源仓不附带预编译 `.so`，需在本地按本机 SoC / CANN / Python ABI 自行编译。

## 从源码编译

```bash
# 前提：CANN 已安装，conda 环境已配置
cd /path/to/mxfp4_ascendc

# 编译（约 5 分钟）
bash build.sh

# 将产物拷贝到运行目录（文件名中的 Python 版本/架构以本机构建结果为准）
cp build/mxfp4_ascendc_ext.cpython-*-linux-gnu.so mxfp4_npu/_lib/
cp build/lib/libascendc_kernels_npu.so mxfp4_npu/_lib/

# 测试正确性 + 性能
python test_mxfp4.py
```

## 快速使用

编译并拷贝产物后即可 import：

```python
import sys
sys.path.insert(0, "/path/to/mxfp4_ascendc")  # 包含 mxfp4_npu/ 的目录

from mxfp4_npu import quant_dequant_mxfp4

# 替换原来的 PyTorch 实现
x_npu = x.npu()  # tensor 移到 NPU
result = quant_dequant_mxfp4(x_npu)
```

### API

```python
quant_dequant_mxfp4(
    x: torch.Tensor,       # 任意 shape，float32，在 NPU 设备上
    block_size: int = 32,   # 量化 block 宽度（必须为 32）
    block_dim: int = 20,    # AI Core 数量（910B3 默认 20）
) -> torch.Tensor           # 同 shape / dtype / device
```

### 性能

| Shape | torch_npu | Ascend-C | 加速比 |
|-------|-----------|----------|--------|
| (64, 4096) | 0.69 ms | 0.038 ms | **18.1x** |
| (256, 4096) | 0.72 ms | 0.059 ms | **12.3x** |
| (1024, 4096) | 0.72 ms | 0.219 ms | **3.28x** |

---

## 项目结构

```
mxfp4_ascendc/
├── mxfp4_npu/                  # Python 包装（import 入口）
│   ├── __init__.py             #    导出 quant_dequant_mxfp4()
│   └── _lib/                   #    本地编译产物目录（开源仓不含 .so）
│       └── __init__.py         #    占位；编译后拷入 .so
│
├── csrc/                       # Ascend-C 源码
│   ├── mxfp4_kernel.cpp        #    设备核（唯一需要优化的文件）
│   ├── pybind_wrapper.cpp      #    Pybind11 封装
│   └── CMakeLists.txt          #    构建配置
│
├── build.sh                    # 一键编译脚本
├── build/                      # 编译中间产物（本地生成，勿提交）
├── test_mxfp4.py               # 正确性 + benchmark 测试
├── test_inv_scale.py           # inv_scale 相关正确性测试
├── mxfp4_npu.py                # 开发用 Python 加载器
└── README.md                   # 本文件
```
