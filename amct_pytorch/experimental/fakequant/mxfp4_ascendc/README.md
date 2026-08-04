# MXFP4 Ascend-C 加速算子（experimental）

在 Ascend NPU 上实现 MXFP4（Microscaling FP4 E2M1）伪量化算子的 Ascend-C 自定义 kernel。目录布局对齐 `amct_ops/hifloat8_cast`（`op_kernel` / `op_extension` / `python`），实现仍保留在 `amct_pytorch/experimental/fakequant/`（试验特性，不进入 `amct_ops`）。

对比 torch_npu 软件路径：**3.3x 加速**（大矩阵），**18x 加速**（小矩阵）。正确性与 PyTorch 参考实现 bit-exact 一致。

## 运行环境

| 组件 | 版本 |
|------|------|
| 硬件 | Ascend 910B3 / 兼容 SoC |
| CANN | 8.2.RC1+ |
| Python | 3.10 (aarch64) |
| PyTorch | 2.6.0 |
| torch_npu | 2.6.0.post4 |

> 开源仓不附带预编译 `.so`，需在本地按本机 SoC / CANN / Python ABI 自行编译。

## 目录结构

```
mxfp4_ascendc/
├── op_kernel/
│   ├── mxfp4_kernel.cpp      # Ascend-C device kernel
│   └── mxfp4_tiling.h        # Host/device 共用 tiling 常量与结构体
├── op_extension/
│   ├── mxfp4_torch.cpp       # PyTorch host：tiling + ACLRT_LAUNCH_KERNEL
│   ├── ops.h                 # C++ host 接口声明（namespace AscendKernel）
│   └── register.cpp          # TORCH_LIBRARY_FRAGMENT(amct, ...) + Meta
├── python/
│   └── mxfp4/
│       ├── __init__.py       # 加载 .so、自检、re-export
│       └── ops.py            # 薄 Python 包装（pad / dtype）
├── reference/
│   └── mxfp4_ref.py          # 纯 PyTorch 参考实现
├── CMakeLists.txt            # 构建入口
├── build.sh                  # 一键编译并 stage .so 到 python/mxfp4/
├── tests/                    # 正确性 / inv_scale / benchmark
└── README.md
```

## 从源码编译

```bash
cd /path/to/mxfp4_ascendc

# 编译（约数分钟）；成功后自动将 .so 拷到 python/mxfp4/
bash build.sh

# 测试正确性 + 性能
python tests/test_mxfp4.py
# python tests/test_inv_scale.py   # inv_scale 参数正确性
# python tests/bench_qdq.py       # 额外性能对比
```

指定 SoC：

```bash
SOC_VERSION=Ascend910_9392 bash build.sh
```

## 快速使用

```python
import sys
sys.path.insert(0, "/path/to/mxfp4_ascendc/python")

from mxfp4 import quant_dequant_mxfp4

x_npu = x.npu()
result = quant_dequant_mxfp4(x_npu)

# 等价底层调用（输入需已是 float32 flat，numel 为 32 的倍数）
# result = torch.ops.amct.quant_dequant_mxfp4(x_flat, 1.0)
```

### API

```python
quant_dequant_mxfp4(
    x: torch.Tensor,                 # 任意 shape，建议 float32，在 NPU 上
    block_size: int = 32,            # 量化 block 宽度（必须为 32）
    inv_scale_factor_scale: float = 1.0,
) -> torch.Tensor                    # 同 shape / dtype / device
```

AIV 核数由 host 侧 `PlatformAscendC::GetCoreNumAiv()` 运行时查询，无需手动指定。

### 性能

| Shape | torch_npu | Ascend-C | 加速比 |
|-------|-----------|----------|--------|
| (64, 4096) | 0.69 ms | 0.038 ms | **18.1x** |
| (256, 4096) | 0.72 ms | 0.059 ms | **12.3x** |
| (1024, 4096) | 0.72 ms | 0.219 ms | **3.28x** |
