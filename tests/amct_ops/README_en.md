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

## HiFloat8 dtype smoke validation

`run_hifloat8_dtype_smoke.py` validates the native HiFloat8 cast, the
`amct_ops` fallback, and automatic routing with deterministic FP16/BF16 inputs.
It covers zeros, signed values, powers of two, boundary values, and seeded
random data, then emits JSON with shape/dtype/device contracts and error metrics.

```bash
# Validate only the native torch_npu HiFloat8 cast
python3 tests/amct_ops/run_hifloat8_dtype_smoke.py --backend native

# Validate only amct_ops hifloat8_cast
PYTHONPATH=amct_ops/staging \
python3 tests/amct_ops/run_hifloat8_dtype_smoke.py --backend amct_ops

# Prefer native and fall back to amct_ops after a native failure
PYTHONPATH=amct_ops/staging \
python3 tests/amct_ops/run_hifloat8_dtype_smoke.py --backend auto
```

Use `--device npu:1` to select another device. A successful run exits with `0`
and sets `hardware_path_verified: true`; a shape, dtype, or device contract
failure exits with `1`; a missing NPU, `torch_npu`, `amct_ops`, or backend
execution failure emits an error JSON object and exits non-zero.

Passing `py_compile`, `--help`, or CPU unit tests does not validate a hardware
path. Until the commands above run on a real NPU, both the native and
`amct_ops` paths must remain recorded as "hardware path not validated."
