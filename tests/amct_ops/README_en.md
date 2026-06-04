# amct_ops Test Instructions

Tests in `tests/amct_ops` depend on the compiled `amct_ops` Python package and operator `.so`. Before running tests, you need to first build `amct_ops`.

## Method 1: Use Staging to Run (Recommended for Development)

```bash
bash amct_ops/ops_build.sh hifloat8_cast

PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_hifloat8_cast
```

## Method 2: Install wheel then Run

```bash
bash amct_ops/ops_build.sh hifloat8_cast
pip install amct_ops/dist/amct_ops-*.whl

python3 -m unittest tests.amct_ops.test_hifloat8_cast
```

## Environment Requirements

- Have sourced CANN environment variables, such as `$ASCEND_HOME_PATH/set_env.sh`
- Current environment has available `torch`, `torch_npu`
- Current machine can access NPU, tests will call `torch.npu.set_device(0)`