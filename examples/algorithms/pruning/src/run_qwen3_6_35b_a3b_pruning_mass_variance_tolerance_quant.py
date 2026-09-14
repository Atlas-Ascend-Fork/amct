# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Issue #183 task 18: mass_variance pruning followed by Qwen3.6 quant eval.

The pruning search is delegated to the maintained task-2 sample so both
examples use exactly the same calibration/evaluation semantics.  Quantization
is then run through the public Qwen3.6 blockwise workflow, which is the
supported path for the model's fused MoE implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PRUNE_SCRIPT = Path(__file__).with_name(
    "run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py"
)
PPL_RE = re.compile(r"PPL evaluation completed:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.6-35B-A3B mass_variance+tolerance+quant sample"
    )
    parser.add_argument("--model", required=True, help="Local Qwen3.6 checkpoint")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        choices=[0.1],
        help="Allowed absolute PPL increase; task 18 fixes this at 0.1",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/qwen3_6_35b_a3b_mass_variance_quant",
        help="Relative output directory",
    )
    parser.add_argument(
        "--device-map", choices=["auto", "single", "cpu"], default="auto"
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--baseline-ppl", type=float, default=None)
    parser.add_argument("--calib-nsamples", type=int, default=8)
    parser.add_argument("--calib-seq-len", type=int, default=512)
    parser.add_argument("--ratio-grid", default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--bit-config", default="amct_pytorch/configs/w8a8.yaml")
    parser.add_argument(
        "--quant-target",
        nargs="+",
        default=["attn-linear", "moe"],
        choices=["mlp", "moe", "attn-linear"],
    )
    parser.add_argument("--quant-dtype", default="int", choices=["int", "mxfp", "hifp"])
    parser.add_argument("--quant-algos", nargs="*", default=[])
    parser.add_argument("--attn-linear-param-dir", default=None)
    parser.add_argument("--skip-quant-eval", action="store_true")
    return parser.parse_args()


def _run_pruning(args: argparse.Namespace, prune_output: Path) -> tuple[Path, str]:
    command = [
        sys.executable,
        str(PRUNE_SCRIPT.relative_to(ROOT)),
        "--model",
        args.model,
        "--tolerance",
        str(args.tolerance),
        "--output-dir",
        str(prune_output),
        "--device-map",
        args.device_map,
        "--device",
        args.device,
        "--calib-nsamples",
        str(args.calib_nsamples),
        "--calib-seq-len",
        str(args.calib_seq_len),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.baseline_ppl is not None:
        command += ["--skip-baseline-eval", "--baseline-ppl", str(args.baseline_ppl)]
    if args.ratio_grid:
        command += ["--ratio-grid", args.ratio_grid]
    if args.eval_batches is not None:
        command += ["--eval-batches", str(args.eval_batches)]
    print("[prune-command]", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    process.wait()
    output = "".join(lines)
    if process.returncode:
        raise RuntimeError(
            f"pruning command failed with exit code {process.returncode}"
        )
    candidate = prune_output / f"tolerance_{args.tolerance}" / "pruned_model"
    # Tight tolerances can legitimately leave the original model untouched.
    model_for_quant = (
        candidate.resolve() if candidate.is_dir() else Path(args.model).resolve()
    )
    return model_for_quant, output


def _run_quant_eval(
    args: argparse.Namespace, model_dir: Path, output_dir: Path
) -> tuple[float | None, str, int]:
    command = [
        sys.executable,
        "-m",
        "amct_pytorch.eval",
        "--model",
        str(model_dir),
        "--model_name",
        "qwen3_6_moe",
        "--seq_len",
        "4096",
        "--granularity",
        "block",
        "--device",
        args.device,
        "--eval_mode",
        "quant",
        "--quant_target",
        *args.quant_target,
        "--quant_dtype",
        args.quant_dtype,
        "--bit_config",
        args.bit_config,
        "--output_dir",
        str(output_dir / "quant_eval"),
    ]
    if args.trust_remote_code:
        command.append("--trust_remote_code")
    if args.quant_algos:
        command += ["--algos", *args.quant_algos]
    if args.attn_linear_param_dir:
        command += ["--attn_linear_param_dir", args.attn_linear_param_dir]
    print("[quant-command]", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    process.wait()
    output = "".join(lines)
    match = PPL_RE.findall(output)
    return (float(match[-1]) if match else None), " ".join(command), process.returncode


def main() -> None:
    args = parse_args()
    print(
        "[run-config]",
        json.dumps(vars(args), ensure_ascii=True, sort_keys=True),
        flush=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir, prune_log = _run_pruning(args, output_dir / "pruning")
    nested_result_path = (
        output_dir / "pruning" / f"tolerance_{args.tolerance}" / "result.json"
    )
    nested_result = {}
    if nested_result_path.is_file():
        nested_result = json.loads(nested_result_path.read_text(encoding="utf-8"))
    quant_ppl = quant_command = None
    quant_returncode = None
    if not args.skip_quant_eval:
        quant_ppl, quant_command, quant_returncode = _run_quant_eval(
            args, model_dir, output_dir
        )
    result = {
        "model": "Qwen3.6-35B-A3B",
        "method": "mass_variance",
        "mode": "tolerance",
        "tolerance": args.tolerance,
        "ratio_grid": args.ratio_grid,
        "post_process": "quant",
        "model_for_quant": str(model_dir),
        "quant_seq_len": 4096,
        "quant_granularity": "block",
        "quant_target": args.quant_target,
        "quant_dtype": args.quant_dtype,
        "quant_ppl": quant_ppl,
        "quant_command": quant_command,
        "quant_returncode": quant_returncode,
        "pruning_log_tail": prune_log[-4000:],
        "baseline_ppl": nested_result.get("baseline_ppl"),
        "pruned_ppl": nested_result.get("pruned_ppl"),
        "post_process_ppl": quant_ppl,
        "params_before": nested_result.get("params_before"),
        "params_after": nested_result.get("params_after"),
        "param_cut_rate": nested_result.get("param_cut_rate"),
        "prune_minutes": nested_result.get("prune_minutes"),
        "pruning_result": str(nested_result_path),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print("Saved:", result_path, flush=True)
    if quant_returncode:
        raise RuntimeError(
            f"quantization evaluation failed with exit code {quant_returncode}"
        )


if __name__ == "__main__":
    main()
