#!/usr/bin/env python3
"""Conformance check: a calibration algorithm's params must actually take effect.

Runs ptq (a few blocks) then eval with vs without the trained params, and asserts
the quantized PPL differs from direct quant — i.e. the calibration params reached
the forward path. This guards against the class of "params exported/loaded but
stranded on the wrong instance" bugs (e.g. MoE expert_modules vs transient PTQ
instances; attention wrappers lacking export_ptq_params).

Integration test: requires an NPU, a real model, and calibration data; intended to
be run per (model, target, algo) on hardware. Skips when env is not configured.

Env:
  CONF_MODEL      model path (a MoE model exercises the hardest path)
  CONF_MODEL_NAME registered adapter, e.g. qwen3_moe
  CONF_TARGET     quant target, e.g. moe / attn-linear
  CONF_BITCFG     bit_config yaml (e.g. W4A16-g64)
  CONF_DATA_DIR   extracted ptq data dir
  CONF_BLOCKS     blocks to gptq (default 4)
"""
import os
import subprocess
import sys

WRAP = os.environ.get("CONF_WRAPPER", "cli_wrapper.py")  # amct CLI launcher


def _need(name):
    v = os.environ.get(name)
    if not v:
        print(f"SKIP: env {name} not set")
        sys.exit(0)
    return v


def _eval(model, name, target, bitcfg, algos=None, param_dir=None):
    cmd = [sys.executable, WRAP, "eval", "--model", model, "--model_name", name,
           "--device", "npu:0", "--granularity", "block", "--eval_mode", "quant",
           "--quant_target", target, "--quant_dtype", "int", "--bit_config", bitcfg,
           "--seq_len", "4096"]
    if algos:
        cmd += ["--algos", algos]
    if param_dir:
        cmd += ["--moe_mlp_param_dir", param_dir]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "PPL:" in line:
            return float(line.split("PPL:")[1].strip())
    raise RuntimeError("no PPL in eval output")


def test_calibration_params_take_effect():
    model = _need("CONF_MODEL"); name = _need("CONF_MODEL_NAME")
    target = _need("CONF_TARGET"); bitcfg = _need("CONF_BITCFG")
    data = _need("CONF_DATA_DIR")
    blocks = int(os.environ.get("CONF_BLOCKS", "4"))
    param = os.path.join(os.environ.get("CONF_OUT", "/tmp/conf_out"), "param")
    os.makedirs(param, exist_ok=True)
    subprocess.run([sys.executable, WRAP, "ptq", "--model", model, "--model_name", name,
                    "--device", "npu:0", "--granularity", "block", "--quant_target", target,
                    "--quant_dtype", "int", "--bit_config", bitcfg, "--algos", "gptq",
                    "--data_dir", data, "--start_block_idx", "0", "--end_block_idx", str(blocks),
                    "--moe_mlp_param_dir", param], check=True)
    direct = _eval(model, name, target, bitcfg)
    gptq = _eval(model, name, target, bitcfg, algos="gptq", param_dir=param)
    print(f"direct={direct} gptq={gptq}")
    # calibration params must reach the forward path → result must differ from direct
    assert abs(gptq - direct) > 1e-4, (
        f"calibration params had no effect (gptq {gptq} == direct {direct}); "
        f"likely stranded on a non-forward instance"
    )


if __name__ == "__main__":
    test_calibration_params_take_effect()
    print("CONFORMANCE PASS")
