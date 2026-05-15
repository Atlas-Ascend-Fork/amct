# HiFloat8 Cast 测试说明

## 测试文件

| 文件 | 用途 | 依赖 |
|------|------|------|
| `run_tests.py` | 统一入口脚本 | 无 |
| `test_accuracy.py` | 精度测试（仅 NPU） | amct_ops |
| `test_accuracy_cpu_npu.py` | CPU vs NPU 精度对比 | amct_ops + CPU 参考 |
| `test_performance.py` | 性能测试（仅 NPU） | amct_ops |
| `test_performance_cpu_npu.py` | CPU vs NPU 性能对比 | amct_ops + CPU 参考 |


## 快速使用

运行脚本前需先安装 `amct_ops` wheel，或在源码树中设置 `PYTHONPATH` 指向 `amct_ops/staging`：

```bash
export PYTHONPATH=/path/to/amct/amct_ops/staging
```

### 运行全部测试

```bash
python scripts/run_tests.py
```

### 运行指定测试

```bash
python scripts/run_tests.py --test accuracy   # 精度测试
python scripts/run_tests.py --test perf       # 性能测试
python scripts/run_tests.py --test cpu-npu    # CPU vs NPU 对比
```

### 直接运行单个脚本

```bash
python scripts/test_accuracy.py
python scripts/test_performance.py
```

## 测试覆盖

### test_accuracy.py

- 类型推断（FP16/BF16 自动识别，非法类型报错）
- Decode 输出类型（默认 BF16，可指定 FP16/BF16）
- 编解码往返（基础值、随机数据）
- 边界值测试
- 特殊值（±0、±Inf、NaN）
- 非规格化数（Denormal）
- 全 256 个 HiFloat8 值 decode 完备性

### test_performance.py

- Encode 性能（FP16/BF16 → HiFloat8）
- Decode 性能（HiFloat8 → FP16/BF16）
- 编解码往返性能
- 数据规模：1K ~ 16M

### CPU vs NPU 对比测试

需要额外构建 CPU 参考实现：

```bash
cd amct_pytorch/experimental/hifloat8
bash build.sh
```

运行时需设置 LD_PRELOAD：

```bash
LIBGOMP_ARCH="$(uname -m)"
LD_PRELOAD=/usr/lib/${LIBGOMP_ARCH}-linux-gnu/libgomp.so.1 python scripts/test_accuracy_cpu_npu.py
LD_PRELOAD=/usr/lib/${LIBGOMP_ARCH}-linux-gnu/libgomp.so.1 python scripts/test_performance_cpu_npu.py
```
