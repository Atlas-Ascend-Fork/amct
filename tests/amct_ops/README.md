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
