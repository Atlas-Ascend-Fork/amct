"""直接基于「已缓存激活 + safetensors 权重」运行 OSPlus SmoothQuant 搜索。

本脚本用于 stage1 的「仅记录激活」模式（``run_stage1.sh`` 中 ``RECORD_ONLY=1``）
产出激活缓存之后，离线、可并行地补跑 scale 搜索。它刻意绕开完整的 HF 模型加载
（对 MiniMax-M2.7 约需十几分钟），因为每层的 qkv / o_proj 权重块彼此独立：

  1. 直接从模型目录的 safetensors 分片读取每层
     ``q_proj / k_proj / v_proj / o_proj`` 权重（BF16 -> float32）；
  2. 运行与 ``stage1_calibrate.py`` 完全一致的 ``OSPlusMigrator`` /
     ``OSPlusMigratorGQA`` 搜索；
  3. 将 ``layer_{i}_attn_scale.pt`` 与 ``layer_{i}_oproj_scale.pt`` 保存到
     激活缓存所在的 record_dir。

配合 ``run_stage1.sh`` 的多 worker + numactl 绑核，可在 CPU 上并行完成全部层的搜索。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401  registers npu backend
except ImportError:
    torch_npu = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_calibrate import (  # noqa: E402
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    OSPlusMigrator,
    OSPlusMigratorGQA,
)

HEAD_DIM = 128
Q_SIZE = NUM_ATTENTION_HEADS * HEAD_DIM  # 6144
KV_SIZE = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 1024


def _release(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_weights(
    model_dir: str,
    layer_idx: int,
    weight_map: dict[str, str],
    cache: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(combined_qkv_weight, o_proj_weight)`` as float32 CPU tensors.

    combined_qkv = cat([q, k, v], dim=0) shape = [Q_SIZE + 2*KV_SIZE, HIDDEN_SIZE]
    o_proj_weight                       shape = [HIDDEN_SIZE, Q_SIZE]
    """
    from safetensors import safe_open

    key_q = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
    key_k = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    key_v = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
    key_o = f"model.layers.{layer_idx}.self_attn.o_proj.weight"

    # Group keys by shard so each shard opens at most once.
    by_shard: dict[str, list[str]] = {}
    for k in (key_q, key_k, key_v, key_o):
        if k not in weight_map:
            raise KeyError(f"Key {k!r} not found in weight_map")
        by_shard.setdefault(weight_map[k], []).append(k)

    tensors: dict[str, torch.Tensor] = {}
    for shard_rel, keys in by_shard.items():
        shard_path = os.path.join(model_dir, shard_rel)
        with safe_open(shard_path, framework="pt") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k).to(torch.float32)

    # Shape checks.
    for k, want_rows in ((key_q, Q_SIZE), (key_k, KV_SIZE), (key_v, KV_SIZE)):
        if tensors[k].shape != (want_rows, HIDDEN_SIZE):
            raise ValueError(
                f"{k} has shape {tuple(tensors[k].shape)}; expected ({want_rows}, {HIDDEN_SIZE})"
            )
    if tensors[key_o].shape != (HIDDEN_SIZE, Q_SIZE):
        raise ValueError(
            f"{key_o} has shape {tuple(tensors[key_o].shape)}; expected ({HIDDEN_SIZE}, {Q_SIZE})"
        )

    qkv = torch.cat([tensors[key_q], tensors[key_k], tensors[key_v]], dim=0)
    return qkv, tensors[key_o]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run OS+SQ search from cached activations + safetensors weights."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="MiniMax-M2.7 BF16 模型目录（含 model.safetensors.index.json）。",
    )
    parser.add_argument(
        "--record_dir",
        required=True,
        help="Directory containing activations/ and where scales will be written.",
    )
    parser.add_argument("--num_layers", type=int, default=62)
    parser.add_argument(
        "--search_device",
        default="npu:0",
        help="Where to run the OS+ grid search (cpu / npu:0 / cuda:0).",
    )
    parser.add_argument(
        "--resume",
        type=int,
        choices=(0, 1),
        default=1,
        help="Skip layers that already have both scale files.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Optional comma-separated list / range of layer ids (e.g. '0,2,5-10').",
    )
    return parser.parse_args()


def _resolve_search_device(search_device: str) -> torch.device:
    device = torch.device(search_device)
    if device.type == "npu" and not (
        hasattr(torch, "npu") and torch.npu.is_available()
    ):
        print(f"[WARN] requested {device}, but NPU unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[WARN] requested {device}, but CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def _parse_layers(layers_arg: str | None, num_layers: int) -> list[int]:
    if not layers_arg:
        return list(range(num_layers))
    layers: list[int] = []
    for chunk in layers_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(chunk))
    return layers


def _filter_layers_for_resume(layers: list[int], record_dir: Path) -> list[int]:
    skipped = []
    keep = []
    for li in layers:
        a = record_dir / f"layer_{li}_attn_scale.pt"
        o = record_dir / f"layer_{li}_oproj_scale.pt"
        if a.is_file() and o.is_file():
            skipped.append(li)
        else:
            keep.append(li)
    if skipped:
        print(f"Resume: skipping {len(skipped)} already-done layers.")
    return keep


def _search_one_layer(
    model_dir: str,
    layer_idx: int,
    weight_map: dict[str, str],
    act_dir: Path,
    record_dir: Path,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run OS+ search for one layer. Returns (w_secs, g1_secs, g2_secs)."""
    t_w = time.perf_counter()
    qkv_w, o_w = load_weights(model_dir, layer_idx, weight_map)
    qkv_w = qkv_w.to(device)
    o_w = o_w.to(device)
    _sync(device)
    w_secs = time.perf_counter() - t_w

    attn_act = torch.load(
        act_dir / f"layer_{layer_idx}_attn_act.pt",
        map_location="cpu",
        weights_only=True,
    ).to(device=device, dtype=torch.float32)
    oproj_act = torch.load(
        act_dir / f"layer_{layer_idx}_oproj_act.pt",
        map_location="cpu",
        weights_only=True,
    ).to(device=device, dtype=torch.float32)

    t_g1 = time.perf_counter()
    attn_scale = OSPlusMigrator(attn_act, qkv_w).search()
    _sync(device)
    g1_secs = time.perf_counter() - t_g1
    del attn_act, qkv_w
    gc.collect()
    _release(device)

    t_g2 = time.perf_counter()
    oproj_scale = OSPlusMigratorGQA(
        oproj_act,
        o_w,
        num_kv_heads=NUM_KEY_VALUE_HEADS,
        num_attention_heads=NUM_ATTENTION_HEADS,
    ).search()
    _sync(device)
    g2_secs = time.perf_counter() - t_g2
    del oproj_act, o_w
    gc.collect()
    _release(device)

    torch.save(attn_scale.cpu(), record_dir / f"layer_{layer_idx}_attn_scale.pt")
    torch.save(oproj_scale.cpu(), record_dir / f"layer_{layer_idx}_oproj_scale.pt")
    del attn_scale, oproj_scale
    gc.collect()
    _release(device)
    return w_secs, g1_secs, g2_secs


def main():
    args = _parse_args()
    device = _resolve_search_device(args.search_device)
    print(f"Using search device: {device}")

    record_dir = Path(args.record_dir)
    act_dir = record_dir / "activations"
    if not act_dir.is_dir():
        raise FileNotFoundError(f"Missing activations/ under {record_dir}")

    layers = _parse_layers(args.layers, args.num_layers)
    if args.resume:
        layers = _filter_layers_for_resume(layers, record_dir)
    if not layers:
        print("Nothing to do.")
        return

    with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    t_overall = time.perf_counter()
    for idx, li in enumerate(layers):
        t_layer = time.perf_counter()
        w_secs, g1_secs, g2_secs = _search_one_layer(
            args.model_dir, li, weight_map, act_dir, record_dir, device
        )
        layer_secs = time.perf_counter() - t_layer
        elapsed = time.perf_counter() - t_overall
        done = idx + 1
        eta = (elapsed / done) * (len(layers) - done)
        print(
            f"Layer {li:>2}: w_load={w_secs:.1f}s g1={g1_secs:.1f}s g2={g2_secs:.1f}s "
            f"total={layer_secs:.1f}s | progress {done}/{len(layers)} "
            f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
        )

    print(f"\nDone. All scale files written under {record_dir}")


if __name__ == "__main__":
    main()
