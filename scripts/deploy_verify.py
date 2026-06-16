#!/usr/bin/env python3
"""Verify GLM5.2 deploy correctness by comparing dequantized weights against original BF16.

Usage:
    # 1. First deploy partial layers:
    python -m amct_pytorch.cli.deploy \
        --model_name glm_5_2 --model /path/to/GLM-5.2 \
        --quant_target "moe attn-linear" \
        --bit_config amct_pytorch/configs/w4a8.yaml \
        --quant_dtype int --granularity block \
        --layer_range 0,3 --output_dir /tmp/glm5.2-deploy-l0-3

    # 2. Then run this script:
    python scripts/deploy_verify.py \
        --deploy_dir /tmp/glm5.2-deploy-l0-3 \
        --model_path /path/to/GLM-5.2 \
        --layers 0,1,2,3
"""

import argparse
import json
import torch
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict

from amct_pytorch.quantization.dtypes.int_impl import weight_dequant_int


def load_deploy_tensors(deploy_dir: Path) -> dict:
    """Load all layer_*.safetensors from deploy directory, keyed by weight name."""
    tensors = {}
    for f in sorted(deploy_dir.glob("layer_*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key)
    return tensors


def load_original_weight(model_path: Path, weight_name: str) -> torch.Tensor:
    """Load a single original BF16 weight from the model directory."""
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    file_name = weight_map[weight_name]
    with safe_open(str(model_path / file_name), framework="pt", device="cpu") as sf:
        return sf.get_tensor(weight_name)


def verify_weight(name: str, deploy_tensors: dict, model_path: Path,
                  bits_map: dict[str, int]) -> tuple[float, float, str]:
    """Dequantize and compare one weight. Returns (cosine_sim, max_abs_err, status)."""
    qkey = name + ".weight"
    skey = name + ".weight_scale"
    bkey = name + ".weight_bias"

    if qkey not in deploy_tensors:
        return 1.0, 0.0, "skip (not in deploy)"

    qweight = deploy_tensors[qkey]
    scale = deploy_tensors.get(skey)
    bias = deploy_tensors.get(bkey)
    bits = bits_map.get(name, 8)

    if scale is None:
        return 0.0, 0.0, "skip (no scale)"

    try:
        dequant = weight_dequant_int(qweight, scale, bias, bits=bits)
    except Exception as e:
        return 0.0, 0.0, f"error: {e}"

    original = load_original_weight(model_path, qkey)
    if dequant.shape != original.shape:
        return 0.0, 0.0, f"shape mismatch: {list(dequant.shape)} vs {list(original.shape)}"

    dequant_fp32 = dequant.float()
    original_fp32 = original.float()

    cos_sim = torch.nn.functional.cosine_similarity(
        dequant_fp32.flatten(), original_fp32.flatten(), dim=0
    ).item()
    max_err = (dequant_fp32 - original_fp32).abs().max().item()

    status = "OK" if cos_sim > 0.99 else f"WARN (cos={cos_sim:.6f})"
    return cos_sim, max_err, status


def build_bits_map(deploy_dir: Path) -> dict[str, int]:
    """Read index.json and classify each weight as INT4 or INT8."""
    index_path = deploy_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    layer_files = {f for w, f in index["weight_map"].items() if f.startswith("layer_")}
    deploy_tensors = {}
    for fname in sorted(layer_files):
        with safe_open(str(deploy_dir / fname), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                deploy_tensors[key] = sf.get_tensor(key)

    bits = {}
    for tn in deploy_tensors:
        if tn.endswith(".weight"):
            name = tn[:-7]  # strip ".weight"
            qw = deploy_tensors[tn]
            skey = name + ".weight_scale"
            scale = deploy_tensors.get(skey)
            if scale is not None and scale.dtype == torch.uint64:
                bits[name] = 4
            else:
                bits[name] = 8
    return bits


def main():
    parser = argparse.ArgumentParser(description="Verify deploy weight fidelity")
    parser.add_argument("--deploy_dir", required=True, help="Deploy output directory")
    parser.add_argument("--model_path", required=True, help="Original BF16 model directory")
    parser.add_argument("--layers", default="0,1,2,3", help="Layers to verify, e.g. '0,1,2,3'")
    args = parser.parse_args()

    deploy_dir = Path(args.deploy_dir)
    model_path = Path(args.model_path)
    layers = [int(x) for x in args.layers.split(",")]

    deploy_tensors = load_deploy_tensors(deploy_dir)
    bits_map = build_bits_map(deploy_dir)

    results = []
    for lidx in layers:
        prefix = f"model.layers.{lidx}."
        layer_names = sorted({
            key[:-7] for key in deploy_tensors
            if key.startswith(prefix) and key.endswith(".weight")
        })
        for name in layer_names:
            cos, err, status = verify_weight(name, deploy_tensors, model_path, bits_map)
            bits = bits_map.get(name, "?")
            results.append((f"L{lidx}", name.replace(prefix, ""), bits, cos, err, status))

    # Report
    print(f"{'Layer':<6} {'Weight':<35} {'Bits':<5} {'Cosine':<12} {'MaxErr':<12} {'Status'}")
    print("-" * 95)
    ok = warn = fail = skip = 0
    for layer, wname, bits, cos, err, status in results:
        print(f"{layer:<6} {wname:<35} {bits:<5} {cos:<12.6f} {err:<12.6e} {status}")
        if "OK" in status:
            ok += 1
        elif "WARN" in status:
            warn += 1
        elif "skip" in status:
            skip += 1
        else:
            fail += 1

    print(f"\nSummary: {ok} OK, {warn} WARN, {fail} FAIL, {skip} skipped")
    return 0 if fail == 0 and warn == 0 else 1


if __name__ == "__main__":
    exit(main())
