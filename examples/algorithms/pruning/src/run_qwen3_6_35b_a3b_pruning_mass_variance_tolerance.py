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
"""Community task #183 / task 2 sample.

Model: Qwen3.6-35B-A3B
Domain: MoE experts
Method: mass_variance
Mode: tolerance (run independently for 0.1 / 0.2 / 0.5)

Pruning runs on CPU (full BF16 model does not fit one NPU card).
Final WikiText2 PPL uses AMCT blockwise eval on NPU (seq_len=4096).
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import amct_pytorch as amct
from amct_pytorch.common.datasets.preproc import get_pileval
from amct_pytorch.pruning import MOE_MASSVAR_PRUNE_CFG, PruneReport, prune_diagnose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.6-35B-A3B MoE mass_variance tolerance pruning sample"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data/models/Qwen3.6-35B-A3B",
        help="Local path to Qwen3.6-35B-A3B weights (placeholder path)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        required=True,
        choices=[0.1, 0.2, 0.5],
        help="Allowed PPL increase vs baseline; run each value as a separate job",
    )
    parser.add_argument(
        "--calib-nsamples",
        type=int,
        default=8,
        help="Number of Pileval calibration sequences",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=512,
        help="Calibration sequence length (CPU forward; keep moderate)",
    )
    parser.add_argument(
        "--eval-seq-len",
        type=int,
        default=4096,
        help="WikiText2 evaluation sequence length (must stay 4096 for the task)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="npu:0",
        help="Device for final blockwise PPL evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/qwen3_6_35b_a3b_mass_variance_tolerance",
        help="Directory for pruned weights and JSON reports (relative/placeholder)",
    )
    parser.add_argument(
        "--skip-baseline-eval",
        action="store_true",
        help="Reuse --baseline-ppl instead of re-running BF16 eval",
    )
    parser.add_argument(
        "--baseline-ppl",
        type=float,
        default=None,
        help="Optional cached baseline WikiText2 PPL",
    )
    parser.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="Skip post-prune WikiText2 PPL (debug only)",
    )
    return parser.parse_args()


def print_run_config(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("Full run parameters:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}={value!r}")
    print("=" * 72)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_amct_ppl(model_dir: str, *, seq_len: int, device: str) -> float:
    """Run official AMCT blockwise BF16 eval and parse WikiText2 PPL."""
    cmd = [
        sys.executable,
        "-m",
        "amct_pytorch.eval",
        "--trust_remote_code",
        "--model",
        model_dir,
        "--model_name",
        "qwen3_6_moe",
        "--seq_len",
        str(seq_len),
        "--granularity",
        "block",
        "--device",
        device,
        "--eval_mode",
        "bf16",
        "--bit_config",
        "amct_pytorch/configs/bf16.yaml",
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(combined)
    marker = "Wikitext2-ppl="
    for line in combined.splitlines():
        if marker in line:
            return float(line.split(marker, 1)[1].strip().split()[0])
    raise RuntimeError("Failed to parse Wikitext2-ppl from amct_pytorch.eval output")


class NegPplEvaluator:
    """Search-mode evaluator: larger is better, so return -PPL.

    Uses a short CPU forward proxy on a few WikiText2 chunks for search speed.
    Final acceptance numbers must still come from AMCT blockwise eval (seq_len=4096).
    """

    def __init__(self, batches: list[torch.Tensor], max_batches: int = 2):
        self.batches = batches[:max_batches]

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module) -> float:
        loss_fn = torch.nn.CrossEntropyLoss()
        nll_sum = 0.0
        tokens = 0
        was_training = model.training
        model.eval()
        for batch in self.batches:
            input_ids = batch if batch.dim() == 2 else batch.unsqueeze(0)
            logits = model(input_ids=input_ids).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].to(shift_logits.device)
            loss = loss_fn(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            nll_sum += float(loss) * shift_labels.numel()
            tokens += shift_labels.numel()
            del logits, shift_logits, shift_labels, loss
        if was_training:
            model.train()
        ppl = torch.exp(torch.tensor(nll_sum / max(tokens, 1))).item()
        return -ppl


def build_config() -> dict:
    cfg = copy.deepcopy(MOE_MASSVAR_PRUNE_CFG)
    # Default mass_variance kwargs (boundary / variance_score) kept as in task appendix.
    # prune_ratio is overridden by tolerance search; keep a seed value for diagnose.
    cfg["methods"]["moe"]["kwargs"]["prune_ratio"] = 0.1
    return cfg


def main() -> None:
    args = parse_args()
    print_run_config(args)

    out_root = Path(args.output_dir) / f"tolerance_{args.tolerance}"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.eval_seq_len != 4096:
        raise ValueError("Community task requires eval_seq_len=4096")

    if args.skip_baseline_eval:
        if args.baseline_ppl is None:
            raise ValueError("--skip-baseline-eval requires --baseline-ppl")
        baseline_ppl = args.baseline_ppl
    else:
        print("[1/5] Baseline WikiText2 PPL (original model, blockwise on NPU)...")
        baseline_ppl = run_amct_ppl(
            args.model, seq_len=args.eval_seq_len, device=args.device
        )
    print(f"baseline_ppl={baseline_ppl}")

    print("[2/5] Load model on CPU for pruning...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()

    print("[3/5] Prepare Pileval calibration + diagnose...")
    calib = get_pileval(tokenizer, args.calib_nsamples, seq_len=args.calib_seq_len)
    cfg = build_config()
    diagnose = prune_diagnose(model, data=calib, config=cfg)
    print("prune_diagnose():")
    print(diagnose)

    # Lightweight proxy batches for search-time evaluator (not the final metric).
    from amct_pytorch.common.datasets.preproc import get_wiki_inputs

    proxy_batches = get_wiki_inputs(tokenizer, seq_len=min(512, args.eval_seq_len))[:2]
    evaluator = NegPplEvaluator(proxy_batches)

    print(f"[4/5] Tolerance search prune: tolerance={args.tolerance} ...")
    t0 = time.time()
    report = PruneReport()
    amct.prune(
        model,
        cfg,
        data=calib,
        tolerance=args.tolerance,
        evaluator=evaluator,
        report=report,
    )
    prune_minutes = (time.time() - t0) / 60.0
    report_dict = report.as_dict()
    print("PruneReport:")
    print(json.dumps(report_dict, indent=2, default=str))

    params_before = report.params_before or count_params(model)
    params_after = report.params_after or count_params(model)
    cut_rate = 1.0 - (params_after / params_before) if params_before else 0.0

    pruned_dir = out_root / "pruned_model"
    pruned_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(pruned_dir))
    tokenizer.save_pretrained(str(pruned_dir))
    print(
        "NOTE: If loading with qwen3_6_moe AMCT path, restore VL text_config nesting "
        "and set text_config.num_experts to the pruned expert count. "
        "See examples/models/qwen3.6/Qwen3.6-Moe-Pruning.md"
    )

    if args.skip_post_eval:
        pruned_ppl = None
    else:
        print("[5/5] Post-prune WikiText2 PPL (blockwise on NPU)...")
        pruned_ppl = run_amct_ppl(
            str(pruned_dir), seq_len=args.eval_seq_len, device=args.device
        )

    result = {
        "model": "Qwen3.6-35B-A3B",
        "method": "mass_variance",
        "mode": "tolerance",
        "tolerance": args.tolerance,
        "baseline_ppl": baseline_ppl,
        "pruned_ppl": pruned_ppl,
        "post_process_ppl": "N/A",
        "params_before": params_before,
        "params_after": params_after,
        "param_cut_rate": cut_rate,
        "prune_minutes": prune_minutes,
        "calib_nsamples": args.calib_nsamples,
        "calib_seq_len": args.calib_seq_len,
        "eval_seq_len": args.eval_seq_len,
        "diagnose": str(diagnose),
        "report": report_dict,
    }
    result_path = out_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("Saved:", result_path)
    print(
        f"RESULT_ROW\tQwen3.6-35B-A3B\tmass_variance\ttolerance={args.tolerance}\t"
        f"{baseline_ppl}\t{pruned_ppl}\tN/A\t"
        f"{params_before} -> {params_after}\t{cut_rate:.4f}\t{prune_minutes:.2f}"
    )


if __name__ == "__main__":
    main()
