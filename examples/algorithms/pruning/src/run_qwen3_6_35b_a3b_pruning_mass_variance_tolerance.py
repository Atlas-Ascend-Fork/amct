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

Note: official ~10% expert prune raises WikiText2 PPL by ~0.33. With the
default ratio_grid starting at 0.1, tolerance=0.1/0.2 may leave the model
unchanged; that is a valid outcome to report.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import amct_pytorch as amct
from amct_pytorch.common.datasets.preproc import get_pileval, get_wiki_inputs
from amct_pytorch.pruning import MOE_MASSVAR_PRUNE_CFG, PruneReport, prune_diagnose
from amct_pytorch.pruning.accuracy_based_auto_prune import DEFAULT_RATIO_GRID


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
    parser.add_argument(
        "--proxy-batches",
        type=int,
        default=4,
        help="WikiText2 chunks for search-time -PPL proxy evaluator",
    )
    parser.add_argument(
        "--proxy-seq-len",
        type=int,
        default=512,
        help="Sequence length for search-time proxy evaluator",
    )
    parser.add_argument(
        "--search-evaluator",
        type=str,
        default="fidelity",
        choices=["fidelity", "proxy_ppl"],
        help=(
            "Tolerance-search metric. 'fidelity' uses AMCT built-in score "
            "(recommended; short CPU proxy_ppl over-penalizes and may reject "
            "all DEFAULT_RATIO_GRID points even when WikiText2 PPL would pass). "
            "Final reported PPL always comes from AMCT blockwise WikiText2 eval."
        ),
    )
    return parser.parse_args()


def print_run_config(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("Full run parameters:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}={value!r}")
    print(f"  ratio_grid={DEFAULT_RATIO_GRID!r}")
    print("=" * 72)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_ppl_from_text(text: str) -> float:
    """Parse PPL from amct_pytorch.eval / loguru output."""
    patterns = [
        r"PPL evaluation completed:\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bPPL:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Wikitext2-ppl=\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise RuntimeError(
        "Failed to parse PPL from amct_pytorch.eval output. "
        "Expected 'PPL evaluation completed:' or 'PPL:'."
    )


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
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(combined)
    if proc.returncode != 0:
        raise RuntimeError(
            f"amct_pytorch.eval failed with exit={proc.returncode}. "
            "See stdout/stderr above."
        )
    return parse_ppl_from_text(combined)


def _read_num_experts(model: torch.nn.Module) -> int | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    text_cfg = getattr(cfg, "text_config", None)
    for obj in (text_cfg, cfg):
        if obj is None:
            continue
        for key in ("num_experts", "num_local_experts", "n_routed_experts"):
            val = getattr(obj, key, None)
            if isinstance(val, int) and val > 0:
                return val
    return None


def restore_vl_config(
    src_model_dir: str, pruned_dir: Path, model: torch.nn.Module
) -> None:
    """Rewrite saved config.json back to VL nesting for amct qwen3_6_moe eval.

    AutoModelForCausalLM.save_pretrained emits a flat language-only config, while
    AMCT qwen3_6_moe expects the original VL layout with text_config.num_experts.
    """
    src = Path(src_model_dir) / "config.json"
    dst = pruned_dir / "config.json"
    with src.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    num_experts = _read_num_experts(model)
    if "text_config" in cfg and isinstance(cfg["text_config"], dict) and num_experts:
        cfg["text_config"]["num_experts"] = num_experts
        print(f"Restored VL config.json with text_config.num_experts={num_experts}")
    else:
        print(
            "WARNING: could not update text_config.num_experts; "
            f"num_experts={num_experts}, keys={list(cfg.keys())}"
        )

    with dst.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


class NegPplEvaluator:
    """Search-mode evaluator: larger is better, so return -PPL.

    Uses a short CPU forward proxy on WikiText2 chunks for search speed.
    Final acceptance numbers must still come from AMCT blockwise eval (seq_len=4096).
    """

    def __init__(self, batches: list[torch.Tensor]):
        self.batches = list(batches)
        self._call = 0

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module) -> float:
        self._call += 1
        loss_fn = torch.nn.CrossEntropyLoss()
        nll_sum = 0.0
        tokens = 0
        was_training = model.training
        model.eval()
        t0 = time.time()
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
        print(
            f"[evaluator] call={self._call} proxy_ppl={ppl:.6f} "
            f"score={-ppl:.6f} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
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
    if out_root.exists():
        shutil.rmtree(out_root)
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
    print(f"baseline_ppl={baseline_ppl}", flush=True)

    print("[2/5] Load model on CPU for pruning...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    params_loaded = count_params(model)
    print(f"loaded_params={params_loaded}", flush=True)

    print("[3/5] Prepare Pileval calibration + diagnose...", flush=True)
    calib = get_pileval(tokenizer, args.calib_nsamples, seq_len=args.calib_seq_len)
    cfg = build_config()
    diagnose = prune_diagnose(model, data=calib, config=cfg)
    print("prune_diagnose():", flush=True)
    print(diagnose, flush=True)

    if args.search_evaluator == "proxy_ppl":
        proxy_batches = get_wiki_inputs(tokenizer, seq_len=args.proxy_seq_len)[
            : args.proxy_batches
        ]
        evaluator = NegPplEvaluator(proxy_batches)
    else:
        evaluator = None
        print(
            "Using AMCT built-in fidelity evaluator for tolerance search. "
            "Final WikiText2 PPL is still measured by blockwise eval.",
            flush=True,
        )

    print(
        f"[4/5] Tolerance search prune: tolerance={args.tolerance} "
        f"search_evaluator={args.search_evaluator} "
        f"ratio_grid={DEFAULT_RATIO_GRID} ...",
        flush=True,
    )
    t0 = time.time()
    report = PruneReport()
    amct.prune(
        model,
        cfg,
        data=calib,
        tolerance=args.tolerance,
        evaluator=evaluator,
        ratio_grid=DEFAULT_RATIO_GRID,
        report=report,
    )
    prune_minutes = (time.time() - t0) / 60.0
    report_dict = report.as_dict()
    print("PruneReport:", flush=True)
    print(json.dumps(report_dict, indent=2, default=str), flush=True)

    params_before = report.params_before or params_loaded
    params_after = report.params_after or count_params(model)
    cut_rate = 1.0 - (params_after / params_before) if params_before else 0.0
    unchanged = cut_rate < 1e-6
    if unchanged:
        print(
            "NOTE: no ratio met this tolerance on DEFAULT_RATIO_GRID; "
            "model left unchanged. This can be expected for tight tolerances "
            "(official ~10% prune already raises PPL by ~0.33).",
            flush=True,
        )

    pruned_dir = out_root / "pruned_model"
    if unchanged:
        print(
            "Skip save_pretrained (model unchanged) to avoid duplicating ~67G weights.",
            flush=True,
        )
        pruned_dir_str = args.model
    else:
        pruned_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving pruned model to {pruned_dir} ...", flush=True)
        model.save_pretrained(str(pruned_dir))
        tokenizer.save_pretrained(str(pruned_dir))
        restore_vl_config(args.model, pruned_dir, model)
        pruned_dir_str = str(pruned_dir)

    if args.skip_post_eval:
        pruned_ppl = None
    elif unchanged:
        print(
            "[5/5] Model unchanged -> reuse baseline WikiText2 PPL "
            "(skip redundant blockwise eval).",
            flush=True,
        )
        pruned_ppl = baseline_ppl
    else:
        print("[5/5] Post-prune WikiText2 PPL (blockwise on NPU)...", flush=True)
        pruned_ppl = run_amct_ppl(
            pruned_dir_str, seq_len=args.eval_seq_len, device=args.device
        )

    result = {
        "model": "Qwen3.6-35B-A3B",
        "method": "mass_variance",
        "mode": "tolerance",
        "tolerance": args.tolerance,
        "search_evaluator": args.search_evaluator,
        "baseline_ppl": baseline_ppl,
        "pruned_ppl": pruned_ppl,
        "post_process_ppl": "N/A",
        "params_before": params_before,
        "params_after": params_after,
        "param_cut_rate": cut_rate,
        "model_unchanged": unchanged,
        "prune_minutes": prune_minutes,
        "calib_nsamples": args.calib_nsamples,
        "calib_seq_len": args.calib_seq_len,
        "eval_seq_len": args.eval_seq_len,
        "ratio_grid": list(DEFAULT_RATIO_GRID),
        "diagnose": str(diagnose),
        "report": report_dict,
    }
    result_path = out_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("Saved:", result_path, flush=True)
    print(
        f"RESULT_ROW\tQwen3.6-35B-A3B\tmass_variance\ttolerance={args.tolerance}\t"
        f"{baseline_ppl}\t{pruned_ppl}\tN/A\t"
        f"{params_before} -> {params_after}\t{cut_rate:.4f}\t{prune_minutes:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
