# amct_ops 测试说明

`tests/amct_ops` 中的测试依赖已编译好的 `amct_ops` Python 包和算子 `.so`。执行测试前需要先构建 `amct_ops`。

## 方式一：使用 staging 运行（推荐开发使用）

```bash
bash amct_ops/ops_build.sh hifloat8_cast

PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_hifloat8_cast
```

## 方式二：安装 wheel 后运行

```bash
bash amct_ops/ops_build.sh hifloat8_cast
pip install amct_ops/dist/amct_ops-*.whl

python3 -m unittest tests.amct_ops.test_hifloat8_cast
```

## 环境要求

- 已 source CANN 环境变量，例如 `$ASCEND_HOME_PATH/set_env.sh`
- 当前环境可用 `torch`、`torch_npu`
- 当前机器可访问 NPU，测试会调用 `torch.npu.set_device(0)`

## HiFloat8 dtype 冒烟验证

`run_hifloat8_dtype_smoke.py` 使用固定的 FP16/BF16 输入验证 dtype 层的原生
HiFloat8 cast、`amct_ops` fallback 和自动路由。脚本会覆盖零值、正负数、幂次值、
边界值及固定随机数据，并输出包含 shape/dtype/device 契约和误差指标的 JSON。

```bash
# 仅验证 torch_npu 原生 HiFloat8 cast
python3 tests/amct_ops/run_hifloat8_dtype_smoke.py --backend native

# 仅验证 amct_ops hifloat8_cast
PYTHONPATH=amct_ops/staging \
python3 tests/amct_ops/run_hifloat8_dtype_smoke.py --backend amct_ops

# 原生优先，失败时自动尝试 amct_ops
PYTHONPATH=amct_ops/staging \
python3 tests/amct_ops/run_hifloat8_dtype_smoke.py --backend auto
```

可通过 `--device npu:1` 指定设备。成功时退出码为 `0`，并设置
`hardware_path_verified: true`；shape、dtype 或 device 契约失败时退出码为 `1`；
缺少 NPU、`torch_npu`、`amct_ops` 或后端执行失败时输出错误 JSON 并以非零状态退出。

仅通过 `py_compile`、`--help` 或 CPU 单元测试不能证明硬件路径可用。未在真实 NPU
环境执行上述命令时，原生和 `amct_ops` 路径均应记录为“硬件路径未验证”。
