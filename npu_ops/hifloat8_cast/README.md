# HiFloat8 Cast Kernel

HiFloat8 数据类型转换算子，支持 FP32/FP16/BF16 ↔ HiFloat8 双向转换。

## 功能

支持 6 种转换：

| 功能 | 说明 |
|------|------|
| FP32 → HiFloat8 | 32-bit 浮点压缩为 8-bit |
| FP16 → HiFloat8 | 16-bit 浮点压缩为 8-bit |
| BF16 → HiFloat8 | 16-bit 浮点压缩为 8-bit |
| HiFloat8 → FP32 | 8-bit 还原为 32-bit 浮点 |
| HiFloat8 → FP16 | 8-bit 还原为 16-bit 浮点 |
| HiFloat8 → BF16 | 8-bit 还原为 16-bit 浮点 |

## 接口

### encode_to_hifloat8(x: Tensor) -> Tensor

将 FP32/FP16/BF16 张量编码为 HiFloat8 格式。

**参数**：
- `x`: 输入张量，支持 `torch.float32` / `torch.float16` / `torch.bfloat16`

**返回**：
- `torch.uint8` 张量

**示例**：
```python
y = encode_to_hifloat8(x_fp32)  # 自动识别类型
```

---

### decode_from_hifloat8(x: Tensor, dtype: dtype = torch.float32) -> Tensor

将 HiFloat8 格式解码为 FP32/FP16/BF16 张量。

**参数**：
- `x`: HiFloat8 编码张量（`torch.uint8`）
- `dtype`: 输出类型，默认 `torch.float32`

**返回**：
- 解码后的张量

**示例**：
```python
z = decode_from_hifloat8(y)                  # 默认 FP32
z = decode_from_hifloat8(y, torch.float16)   # 输出 FP16
```

## 文件结构

```
├── op_kernel/
│   ├── hifloat8_cast_tiling.h    Tiling 常量 + 结构体
│   ├── hifloat8_cast_kernel.asc  Kernel 实现
│   └── hifloat8_cast_wrapper.asc Kernel + Torch binding
├── op_extension/
│   ├── register.cpp              TORCH_LIBRARY 注册
│   └── ops.h                     函数声明
├── python/
│   └── hifloat8/
│       └── __init__.py           Python 接口
├── scripts/
│   └── test_python_api.py        Python 接口测试
├── docs/perf/
│   └ summary.md                  性能报告
├── CMakeLists.txt                构建
└── README.md                     本文档
```

## 编译

```bash
cd npu_ops/hifloat8_cast
source $ASCEND_HOME_PATH/set_env.sh
mkdir -p build && cd build && cmake .. && make -j
```

## 使用

```python
import sys
sys.path.insert(0, 'python')

from hifloat8 import encode_to_hifloat8, decode_from_hifloat8
import torch

# 编码
x = torch.randn(256, dtype=torch.float32, device='npu')
y = encode_to_hifloat8(x)

# 解码
z = decode_from_hifloat8(y, torch.float32)
```

## 性能

**测试平台**: Ascend 910B (A2)

| 数据大小 | Encode (ms) | Decode (ms) | 吞吐 (MB/s) |
|---------|------------|------------|------------|
| 4K      | 0.24       | 0.24       | ~70        |
| 16K     | 0.78       | 0.74       | ~85        |
| 256K    | 11.0       | 10.4       | ~95        |

## 测试

```bash
python3 scripts/test_python_api.py
```

测试内容：
- 自动类型推断
- 类型验证
- 编码-解码精度 100%