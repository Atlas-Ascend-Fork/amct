#!/usr/bin/env python3
"""Decompress INT8/INT4 deploy artifacts back to BF16 for eval verification.

Usage:
    python scripts/deploy_decompress.py \
        --deploy_dir /tmp/glm5.2-deploy-l0-3 \
        --model_path /mnt/workspace/gitCode/fujun19/model/GLM-5.2 \
        --output_dir /tmp/glm5.2-deploy-bf16
"""

import argparse
import json
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

from amct_pytorch.quantization.dtypes.int_impl import weight_dequant_int


def decompress_layers(deploy_dir: Path, output_dir: Path):
    """Decompress all layer_*.safetensors in deploy_dir to BF16 in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(deploy_dir.glob("layer_*.safetensors")):
        tensors = {}
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
            weight_keys = [k for k in keys if k.endswith(".weight")]
            for wk in weight_keys:
                name = wk[:-7]
                sk = name + ".weight_scale"
                bk = name + ".weight_bias"
                qw = sf.get_tensor(wk)
                sc = sf.get_tensor(sk) if sk in keys else None
                bi = sf.get_tensor(bk) if bk in keys else None
                bits = 4 if (sc is not None and sc.dtype == torch.uint64) else 8
                bf16 = weight_dequant_int(qw, sc, bi, bits=bits).to(torch.bfloat16)
                tensors[wk] = bf16
            # Keep unquantized weights (kv_b_proj, norm) as-is
            for k in keys:
                if k.endswith(".weight_scale") or k.endswith(".weight_bias"):
                    continue
                if k not in tensors:
                    tensors[k] = sf.get_tensor(k)
        save_file(tensors, str(output_dir / f.name))
        print(f"  Decompressed: {f.name} ({len(tensors)} tensors)")


def merge_index(model_path: Path, deploy_dir: Path, output_dir: Path):
    """Merge original index with deploy index: L0-3 from deploy, rest from original."""
    orig = json.load(open(model_path / "model.safetensors.index.json"))
    deploy = json.load(open(deploy_dir / "model.safetensors.index.json"))

    new_map = {}
    for wname, orig_file in orig["weight_map"].items():
        deploy_file = deploy["weight_map"].get(wname)
        if deploy_file and deploy_file.startswith("layer_"):
            new_map[wname] = deploy_file
        else:
            new_map[wname] = orig_file

    # Compute total_size for all referenced files
    total_size = 0
    for fname in sorted(set(new_map.values())):
        if fname.startswith("layer_"):
            fpath = output_dir / fname
        else:
            fpath = model_path / fname
        if fpath.exists():
            total_size += fpath.stat().st_size
        else:
            print(f"  Warning: referenced file not found: {fpath}")

    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_map}
    out_path = output_dir / "model.safetensors.index.json"
    json.dump(new_index, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"  Merged index: {out_path} ({len(new_map)} weights, {total_size / 1e9:.1f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Decompress deploy artifacts to BF16 for eval verification"
    )
    parser.add_argument("--deploy_dir", required=True, help="Deploy output directory")
    parser.add_argument("--model_path", required=True, help="Original BF16 model directory")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: deploy_dir + '-bf16')")
    args = parser.parse_args()

    deploy_dir = Path(args.deploy_dir)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(deploy_dir) + "-bf16")

    print(f"Deploy dir:  {deploy_dir}")
    print(f"Model path:  {model_path}")
    print(f"Output dir:  {output_dir}")

    print("\n[1/2] Decompressing layer files...")
    decompress_layers(deploy_dir, output_dir)

    print("\n[2/2] Merging index.json...")
    merge_index(model_path, deploy_dir, output_dir)

    print(f"\nDone. Output: {output_dir}/")
    print("Next: copy layer_*.safetensors and model.safetensors.index.json to original model dir")
    print("  cp {} {}/*".format(
        str(output_dir / "layer_*.safetensors"),
        str(model_path)
    ))
    print("  cp {} {}".format(
        str(output_dir / "model.safetensors.index.json"),
        str(model_path)
    ))


if __name__ == "__main__":
    main()
