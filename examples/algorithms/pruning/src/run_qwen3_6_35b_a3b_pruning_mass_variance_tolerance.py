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

Default placement matches the activation_count task-1 sample: shard the model
across visible NPUs with device_map=auto and run WikiText2 PPL search on NPU.
Pass --device-map cpu only when NPU memory is insufficient.

Tolerance search and acceptance both use WikiText2 PPL (seq_len=4096):
search via NegativePplEvaluator (-PPL); final numbers via the same in-process
WikiText2 PPL path (or AMCT blockwise eval when --device-map cpu).

Note: default ratio_grid starts at 0.1. Absolute PPL deltas at that ratio
often exceed 0.1 / 0.2 / 0.5, so the model may stay unchanged — a valid
outcome. Pass a finer --ratio-grid if pruning must take effect.
"""

from __future__ import annotations

import argparse
import copy
import gc
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    NegativePplEvaluator,
    build_wikitext2_batches,
    eval_wikitext2_ppl,
)

try:
    import torch_npu
except ImportError:  # pragma: no cover
    torch_npu = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.6-35B-A3B MoE mass_variance tolerance pruning sample"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data/models/Qwen3.6-35B-A3B",
        help=(
            "Local directory of Qwen3.6-35B-A3B weights (placeholder path). "
            "Must exist on disk; remote Hub IDs are rejected."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow transformers to execute custom Python shipped with the model. "
            "Off by default. Only enable for a local checkpoint you trust; "
            "this sample's Qwen3.6 weights require the flag."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        required=True,
        choices=[0.1, 0.2, 0.5],
        help=(
            "Allowed absolute WikiText2 PPL increase for tolerance search: "
            "pruned_ppl - baseline_ppl <= tolerance. Default search evaluator is "
            "wikitext2_ppl (same metric family as the final result table). "
            "Run each of 0.1 / 0.2 / 0.5 as a separate job."
        ),
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
        help=(
            "Single-device target when --device-map single, and device for "
            "optional AMCT blockwise eval fallback"
        ),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "single", "cpu"],
        help=(
            "Model placement. 'auto' shards across visible NPUs (recommended, "
            "same as task 1); 'single' uses --device; 'cpu' loads on host RAM."
        ),
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
        "--search-evaluator",
        type=str,
        default="wikitext2_ppl",
        choices=["wikitext2_ppl", "fidelity", "proxy_ppl"],
        help=(
            "Tolerance-search metric. Default 'wikitext2_ppl' uses full-seq "
            "WikiText2 -PPL (aligned with final acceptance PPL). 'fidelity' = "
            "AMCT top-1 agreement; 'proxy_ppl' = short-seq CPU proxy. Do not mix "
            "result-table rows across modes."
        ),
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help=(
            "Optional cap on WikiText2 batches for wikitext2_ppl search; "
            "omit to use the full test split (recommended)."
        ),
    )
    parser.add_argument(
        "--search-device",
        type=str,
        default=None,
        help=(
            "Device for wikitext2_ppl / proxy_ppl search forwards. "
            "Default: follow the model runtime device (NPU when --device-map auto)."
        ),
    )
    parser.add_argument(
        "--proxy-batches",
        type=int,
        default=4,
        help="WikiText2 chunks for search-time proxy_ppl evaluator only",
    )
    parser.add_argument(
        "--proxy-seq-len",
        type=int,
        default=512,
        help="Sequence length for search-time proxy_ppl evaluator only",
    )
    parser.add_argument(
        "--ratio-grid",
        type=str,
        default=None,
        help=(
            "Comma-separated prune ratios for tolerance search; "
            "default uses library DEFAULT_RATIO_GRID"
        ),
    )
    return parser.parse_args()


def resolve_local_model_dir(model: str) -> Path:
    """Require a local model directory; reject remote Hub IDs."""
    path = Path(model).expanduser()
    if not path.exists() or not path.is_dir():
        raise ValueError(
            f"--model must be an existing local directory, got {model!r}. "
            "Remote Hub repo IDs are rejected so untrusted remote code cannot run."
        )
    if not (path / "config.json").is_file():
        raise ValueError(
            f"--model directory {path} has no config.json; "
            "point to a complete local checkpoint."
        )
    return path.resolve()


def model_load_kwargs(trust_remote_code: bool) -> dict:
    if trust_remote_code:
        print(
            "WARNING: --trust-remote-code is enabled. Custom Python from the "
            "model directory may execute under the current user. Use only with "
            "checkpoints you trust.",
            flush=True,
        )
        return {"trust_remote_code": True}
    return {"trust_remote_code": False}


def print_run_config(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("Full run parameters:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}={value!r}")
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


def run_amct_ppl(
    model_dir: str, *, seq_len: int, device: str, trust_remote_code: bool
) -> float:
    """Run official AMCT blockwise BF16 eval and parse WikiText2 PPL."""
    cmd = [
        sys.executable,
        "-m",
        "amct_pytorch.eval",
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
    if trust_remote_code:
        cmd.append("--trust_remote_code")
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


def _config_to_dict(obj) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "to_dict"):
        data = obj.to_dict()
        return data if isinstance(data, dict) else {}
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def _build_text_config(
    src_cfg: dict, saved_cfg: dict | None, model: torch.nn.Module
) -> dict:
    """Build a language text_config dict from VL source, flat save, or model."""
    if isinstance(src_cfg.get("text_config"), dict):
        return dict(src_cfg["text_config"])

    model_cfg = getattr(model, "config", None)
    nested = getattr(model_cfg, "text_config", None) if model_cfg is not None else None
    text = _config_to_dict(nested)
    if text:
        return text

    flat = _config_to_dict(model_cfg)
    if flat:
        return flat

    if isinstance(saved_cfg, dict) and isinstance(saved_cfg.get("text_config"), dict):
        return dict(saved_cfg["text_config"])
    if isinstance(saved_cfg, dict) and saved_cfg:
        return dict(saved_cfg)
    return dict(src_cfg)


def restore_vl_config(
    src_model_dir: str, pruned_dir: Path, model: torch.nn.Module
) -> None:
    """Rewrite saved config.json back to VL nesting for amct qwen3_6_moe eval.

    AutoModelForCausalLM.save_pretrained may emit a flat language-only config.
    AMCT qwen3_6_moe expects VL layout with text_config.num_experts. Prefer the
    original source layout when present; otherwise nest a text_config created from
    the source / saved / model config.
    """
    src = Path(src_model_dir) / "config.json"
    dst = pruned_dir / "config.json"
    with src.open("r", encoding="utf-8") as f:
        src_cfg = json.load(f)

    saved_cfg = None
    if dst.is_file():
        with dst.open("r", encoding="utf-8") as f:
            saved_cfg = json.load(f)

    num_experts = _read_num_experts(model)
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise RuntimeError(
            "Cannot determine pruned num_experts from model.config; "
            "refusing to write config.json without text_config.num_experts."
        )

    text_config = _build_text_config(src_cfg, saved_cfg, model)
    text_config["num_experts"] = num_experts

    if isinstance(src_cfg.get("text_config"), dict):
        cfg = dict(src_cfg)
        cfg["text_config"] = text_config
    else:
        # Flat language-only source: keep top-level keys and add nested text_config
        # so qwen3_6_moe._update_config can read text_config.num_experts.
        cfg = dict(src_cfg)
        cfg["text_config"] = text_config

    with dst.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with dst.open("r", encoding="utf-8") as f:
        written = json.load(f)
    written_text = written.get("text_config")
    written_experts = (
        written_text.get("num_experts") if isinstance(written_text, dict) else None
    )
    if written_experts != num_experts:
        raise RuntimeError(
            "Post-save config assertion failed: expected "
            f"text_config.num_experts={num_experts}, got {written_experts!r} "
            f"in {dst}"
        )
    print(f"Restored VL config.json with text_config.num_experts={num_experts}")


class ProxyNegPplEvaluator:
    """Short-seq CPU WikiText2 proxy (-PPL). Debug / compare only; not default."""

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


def parse_ratio_grid(raw: str | None) -> list[float]:
    if raw is None or not str(raw).strip():
        return list(DEFAULT_RATIO_GRID)
    ratios = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not ratios:
        raise ValueError("--ratio-grid must contain at least one float")
    return ratios


def _npu_empty_cache() -> None:
    gc.collect()
    if torch_npu is not None and torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()


def load_model_for_prune(args: argparse.Namespace, load_kwargs: dict):
    """Load tokenizer/model; default device_map=auto on NPU like task 1."""
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    common = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )
    if args.device_map == "cpu":
        print("[2/5] Load model on CPU for pruning...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, **common).eval()
        runtime_device = "cpu"
    elif args.device_map == "auto":
        print("[2/5] Load model with device_map=auto on NPU...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", **common
        ).eval()
        runtime_device = str(next(model.parameters()).device)
    else:
        print(f"[2/5] Load model on single device {args.device}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, **common).eval()
        model = model.to(args.device)
        runtime_device = args.device
    print(f"runtime_device={runtime_device} device_map={args.device_map}", flush=True)
    return tokenizer, model, runtime_device


def main() -> None:
    args = parse_args()
    print_run_config(args)

    model_dir = resolve_local_model_dir(args.model)
    args.model = str(model_dir)
    load_kwargs = model_load_kwargs(args.trust_remote_code)

    out_root = Path(args.output_dir) / f"tolerance_{args.tolerance}"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.eval_seq_len != 4096:
        raise ValueError("Community task requires eval_seq_len=4096")

    tokenizer, model, runtime_device = load_model_for_prune(args, load_kwargs)
    search_device = args.search_device or runtime_device
    params_loaded = count_params(model)
    print(f"loaded_params={params_loaded}", flush=True)

    print("[3/5] Prepare Pileval calibration + diagnose...", flush=True)
    calib = get_pileval(tokenizer, args.calib_nsamples, seq_len=args.calib_seq_len)
    eval_batches = build_wikitext2_batches(tokenizer, seq_len=args.eval_seq_len)
    if args.eval_batches is not None:
        eval_batches = eval_batches[: args.eval_batches]

    if args.skip_baseline_eval:
        if args.baseline_ppl is None:
            raise ValueError("--skip-baseline-eval requires --baseline-ppl")
        baseline_ppl = args.baseline_ppl
        print(f"[1/5] Reuse baseline_ppl={baseline_ppl}", flush=True)
    else:
        print(
            f"[1/5] Baseline WikiText2 PPL on {runtime_device} "
            f"(batches={len(eval_batches)})...",
            flush=True,
        )
        baseline_ppl = eval_wikitext2_ppl(
            model,
            eval_batches,
            seq_len=args.eval_seq_len,
            device=runtime_device,
        )
    print(f"baseline_ppl={baseline_ppl}", flush=True)

    cfg = build_config()
    ratio_grid = parse_ratio_grid(args.ratio_grid)
    diagnose = prune_diagnose(
        model,
        data=calib,
        config=cfg,
        tolerance=args.tolerance,
    )
    print("prune_diagnose():", flush=True)
    print(diagnose, flush=True)
    _npu_empty_cache()

    if args.search_evaluator == "wikitext2_ppl":
        evaluator = NegativePplEvaluator(
            eval_batches, seq_len=args.eval_seq_len, device=search_device
        )
        print(
            f"Using WikiText2 PPL search evaluator "
            f"(batches={len(eval_batches)}, seq_len={args.eval_seq_len}, "
            f"device={search_device}).",
            flush=True,
        )
    elif args.search_evaluator == "proxy_ppl":
        proxy_batches = get_wiki_inputs(tokenizer, seq_len=args.proxy_seq_len)[
            : args.proxy_batches
        ]
        evaluator = ProxyNegPplEvaluator(proxy_batches)
        print(
            "Using short-seq proxy_ppl search evaluator (not for README table).",
            flush=True,
        )
    else:
        evaluator = None
        print(
            "Using AMCT built-in fidelity evaluator for tolerance search "
            "(not aligned with final WikiText2 PPL; not for README table).",
            flush=True,
        )

    print(
        f"[4/5] Tolerance search prune: tolerance={args.tolerance} "
        f"search_evaluator={args.search_evaluator} "
        f"device_map={args.device_map} search_device={search_device} "
        f"ratio_grid={ratio_grid} ...",
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
        ratio_grid=ratio_grid,
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
            "NOTE: no ratio met this tolerance on the configured ratio_grid; "
            "model left unchanged. This can be expected for tight tolerances.",
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

    result = {
        "model": "Qwen3.6-35B-A3B",
        "method": "mass_variance",
        "mode": "tolerance",
        "tolerance": args.tolerance,
        "search_evaluator": args.search_evaluator,
        "device_map": args.device_map,
        "runtime_device": runtime_device,
        "search_device": search_device,
        "baseline_ppl": baseline_ppl,
        "pruned_ppl": None,
        "post_process_ppl": "N/A",
        "params_before": params_before,
        "params_after": params_after,
        "param_cut_rate": cut_rate,
        "model_unchanged": unchanged,
        "prune_minutes": prune_minutes,
        "calib_nsamples": args.calib_nsamples,
        "calib_seq_len": args.calib_seq_len,
        "eval_seq_len": args.eval_seq_len,
        "ratio_grid": list(ratio_grid),
        "diagnose": str(diagnose),
        "report": report_dict,
    }
    result_path = out_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("Saved prune metrics:", result_path, flush=True)

    if args.skip_post_eval:
        pruned_ppl = None
    elif unchanged:
        print(
            "[5/5] Model unchanged -> reuse baseline WikiText2 PPL "
            "(skip redundant eval).",
            flush=True,
        )
        pruned_ppl = baseline_ppl
    elif args.device_map == "cpu":
        print("[5/5] Post-prune WikiText2 PPL (AMCT blockwise on NPU)...", flush=True)
        pruned_ppl = run_amct_ppl(
            pruned_dir_str,
            seq_len=args.eval_seq_len,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        print(
            f"[5/5] Post-prune WikiText2 PPL on {runtime_device} "
            f"(in-process, same path as search)...",
            flush=True,
        )
        pruned_ppl = eval_wikitext2_ppl(
            model,
            eval_batches,
            seq_len=args.eval_seq_len,
            device=runtime_device,
        )

    result["pruned_ppl"] = pruned_ppl
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
