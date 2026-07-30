"""OSPlus SmoothQuant Stage 2（BF16-only）：逐层应用等价缩放 scale 并导出融合后的
BF16 模型（不做 MXFP4 量化）。

本阶段读取原始 BF16 检查点与 Stage 1 产出的全部 scale，按等价变换公式逐层在 fp32 下
完成融合后再 cast 回 BF16，输出一份与原模型同 shape、同 dtype 的 BF16 检查点。

输出布局
--------
标准 HuggingFace safetensors 纯 BF16 检查点：在 GQA 约束下 SmoothQuant 缩放融合严格
成立，融合后模型与原 BF16 模型仅相差 BF16 舍入误差。``config.json`` 不携带任何
``quantization_config``，可作为普通 BF16 模型在 transformers / vLLM 中加载。

中间 BF16 导出的意义
--------------------
* 在任何低比特量化之前检查 / 评测 / 微调平滑后的 BF16 模型；
* 作为量化起点复用：在平滑后的权重上直接尝试 MXFP4 / W4A8 / W8A8 / INT4 等，
  无需重跑 Stage 1；
* 校验 OSPlus SmoothQuant 的数学正确性：本导出的输出应与原 BF16 模型在 BF16 舍入
  误差范围内一致。

融合位置
--------
  - Group 1：``input_layernorm``  -> q/k/v_proj   （沿 input channel）
  - Group 2：``v_proj``           -> o_proj       （GQA-aware）

实现要点
--------
* 在 ``from_pretrained`` 之前剥离源 BF16 检查点 ``config.json`` 中残留的
  ``quantization_config = {"quant_method": "fp8"}``，否则 transformers 会据此触发
  FP8 量化器，在无 GPU/XPU 的主机上拒绝加载；
* ``clear_accelerator_cache`` 兼容 NPU；
* 融合期间全部算术在 fp32 下完成后再 cast 回 BF16，避免 BF16 累计舍入误差污染等价变换。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file as safe_save_file
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from transformers.utils.hub import convert_file_size_to_int

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from stage1_calibrate import (  # noqa: E402
    HEAD_DIM,
    NUM_HEAD_REPEATS,
    NUM_KEY_VALUE_HEADS,
)


ATTN_IMPLEMENTATION = "eager"
IGNORE_STATE_DICT_KEYS = {"model.rotary_emb.inv_freq"}

COPY_FILES = [
    "chat_template.jinja",
    "configuration_minimax_m2.py",
    "generation_config.json",
    "modeling_minimax_m2.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "merges.txt",
]


def get_module(model, path: str):
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def clear_accelerator_cache() -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_scales(record_dir: str, num_layers: int) -> dict[str, torch.Tensor]:
    scales: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for layer_idx in range(num_layers):
        attn_path = os.path.join(record_dir, f"layer_{layer_idx}_attn_scale.pt")
        oproj_path = os.path.join(record_dir, f"layer_{layer_idx}_oproj_scale.pt")
        if not os.path.exists(attn_path):
            missing.append(attn_path)
            continue
        if not os.path.exists(oproj_path):
            missing.append(oproj_path)
            continue
        scales[f"layers.{layer_idx}.attn"] = torch.load(
            attn_path, map_location="cpu", weights_only=True
        )
        scales[f"layers.{layer_idx}.oproj"] = torch.load(
            oproj_path, map_location="cpu", weights_only=True
        )
    if missing:
        raise FileNotFoundError(
            f"Stage 1 scale files missing ({len(missing)}); first few: {missing[:3]}"
        )
    return scales


def apply_scales_to_layer(
    layer, layer_idx: int, scales: dict[str, torch.Tensor]
) -> None:
    """Fuse OS+SQ scales for one decoder layer (in-place), keeping intermediate
    arithmetic in fp32 to avoid BF16 round-off accumulation.

    Group 1: input_layernorm.weight /= attn_scale
             q/k/v_proj.weight *= attn_scale  (input channel)

    Group 2: v_proj.weight /= oproj_scale   (output channel, KV-head granular)
             o_proj.weight  *= oproj_scale  (input channel, Q-head granular)
    """
    attn_scale = scales[f"layers.{layer_idx}.attn"]
    oproj_scale = scales[f"layers.{layer_idx}.oproj"]

    ln = get_module(layer, "input_layernorm")
    attn_s = attn_scale.to(ln.weight.device).float()
    ln.weight.data = ln.weight.data.float().div_(attn_s).bfloat16()

    q_proj = get_module(layer, "self_attn.q_proj")
    k_proj = get_module(layer, "self_attn.k_proj")
    v_proj = get_module(layer, "self_attn.v_proj")
    q_proj.weight.data = (
        q_proj.weight.data.float()
        .mul_(attn_s.to(q_proj.weight.device).view(1, -1))
        .bfloat16()
    )
    k_proj.weight.data = (
        k_proj.weight.data.float()
        .mul_(attn_s.to(k_proj.weight.device).view(1, -1))
        .bfloat16()
    )

    o_proj = get_module(layer, "self_attn.o_proj")

    o_scale_full = oproj_scale.to(v_proj.weight.device).float()
    grouped = o_scale_full.view(NUM_KEY_VALUE_HEADS, NUM_HEAD_REPEATS, HEAD_DIM)
    v_scale = grouped[:, 0, :].reshape(-1)  # [num_kv_heads * head_dim]

    # v_proj eats both Group 1 (mul attn_s) and Group 2 (div v_scale);
    # do both in one fp32 pass for numerical accuracy.
    v_weight_fp32 = v_proj.weight.data.float()
    v_weight_fp32.mul_(attn_s.to(v_proj.weight.device).view(1, -1))
    v_weight_fp32.div_(v_scale.to(v_proj.weight.device).view(-1, 1))
    v_proj.weight.data = v_weight_fp32.bfloat16()
    if v_proj.bias is not None:
        v_proj.bias.data = (
            v_proj.bias.data.float().div_(v_scale.to(v_proj.bias.device)).bfloat16()
        )

    o_proj.weight.data = (
        o_proj.weight.data.float()
        .mul_(o_scale_full.to(o_proj.weight.device).view(1, -1))
        .bfloat16()
    )


def count_unique_parameters(model) -> int:
    total = 0
    seen: set[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]] = set()
    for name, param in model.named_parameters():
        if name in IGNORE_STATE_DICT_KEYS:
            continue
        storage = param.untyped_storage()
        key = (
            storage.data_ptr(),
            storage.nbytes(),
            param.storage_offset(),
            tuple(param.size()),
            tuple(param.stride()),
        )
        if key in seen:
            continue
        seen.add(key)
        total += param.numel()
    return total


def iter_export_tensors_bf16(model):
    """Yield ``(name, tensor)`` for every weight/buffer as plain BF16 on CPU."""
    exported_names: set[str] = set()
    for name, param in model.named_parameters():
        if name in IGNORE_STATE_DICT_KEYS:
            continue
        if param.device.type == "meta":
            continue
        exported_names.add(name)
        tensor = param.data.detach()
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        yield name, tensor.cpu().clone()

    for name, buf in model.named_buffers():
        if name in IGNORE_STATE_DICT_KEYS or name in exported_names:
            continue
        if buf.device.type == "meta":
            continue
        yield name, buf.data.detach().cpu().clone()


class StreamingShardWriter:
    """Stream BF16 shards to disk; bounded by ``max_shard_size`` per shard."""

    def __init__(
        self,
        output_dir: str,
        max_shard_size: str = "5GB",
        total_parameters: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.max_shard_size_bytes = convert_file_size_to_int(max_shard_size)
        self.total_parameters = total_parameters
        self.current_tensors: dict[str, torch.Tensor] = {}
        self.current_size_bytes = 0
        self.total_size_bytes = 0
        self.shard_files: list[str] = []
        self.shard_tensor_names: list[list[str]] = []

    @staticmethod
    def _tensor_size_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        tensor_size = self._tensor_size_bytes(tensor)
        if (
            self.current_tensors
            and self.current_size_bytes + tensor_size > self.max_shard_size_bytes
        ):
            self.flush()
        self.current_tensors[name] = tensor
        self.current_size_bytes += tensor_size
        self.total_size_bytes += tensor_size

    def flush(self) -> None:
        if not self.current_tensors:
            return
        shard_idx = len(self.shard_files) + 1
        tmp_name = f"model-{shard_idx:05d}.safetensors.tmp"
        tmp_path = os.path.join(self.output_dir, tmp_name)
        print(
            f"[Stage 2 BF16] Saving shard {shard_idx} "
            f"({len(self.current_tensors)} tensors, "
            f"{self.current_size_bytes / 1e9:.2f} GB)"
        )
        safe_save_file(self.current_tensors, tmp_path, metadata={"format": "pt"})
        self.shard_files.append(tmp_path)
        self.shard_tensor_names.append(list(self.current_tensors.keys()))
        self.current_tensors.clear()
        self.current_size_bytes = 0
        gc.collect()
        clear_accelerator_cache()

    def finalize(self) -> None:
        self.flush()
        num_shards = len(self.shard_files)
        if num_shards == 0:
            raise RuntimeError("No tensors were exported.")

        weight_map: dict[str, str] = {}
        for shard_idx, tmp_path in enumerate(self.shard_files, start=1):
            final_name = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
            final_path = os.path.join(self.output_dir, final_name)
            os.replace(tmp_path, final_path)
            for tensor_name in self.shard_tensor_names[shard_idx - 1]:
                weight_map[tensor_name] = final_name

        metadata = {"total_size": self.total_size_bytes}
        if self.total_parameters is not None:
            metadata["total_parameters"] = int(self.total_parameters)
        index = {"metadata": metadata, "weight_map": weight_map}
        with open(
            os.path.join(self.output_dir, SAFE_WEIGHTS_INDEX_NAME),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(index, f, ensure_ascii=False, indent=2)


def _strip_stale_quantization_config(cfg) -> None:
    """Remove leftover FP8 quantization_config from a BF16 source checkpoint."""
    stale_quant = getattr(cfg, "quantization_config", None)
    if stale_quant is None:
        return
    if isinstance(stale_quant, dict):
        stale_method = stale_quant.get("quant_method", "?")
    else:
        stale_method = getattr(stale_quant, "quant_method", "?")
    print(
        f"[Stage 2 BF16] Stripping stale config.quantization_config "
        f"(quant_method={stale_method}) from the source BF16 checkpoint."
    )
    try:
        delattr(cfg, "quantization_config")
    except (AttributeError, TypeError):
        cfg.quantization_config = None
    cfg.__dict__.pop("quantization_config", None)


def _load_bf16_model(model_dir: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    # Source BF16 checkpoint (e.g. MiniMax-M2.7-bf16) ships with a stale
    # ``quantization_config = {"quant_method": "fp8"}`` left over from the
    # block-fp8 ancestor; transformers honours that and triggers the FP8
    # quantizer, which then refuses to run without a GPU/XPU. Delete the
    # attribute (None is not enough -- ``pre_quantized`` is decided by
    # ``hasattr``).
    print(f"[Stage 2 BF16] Loading config from {model_dir} ...")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    _strip_stale_quantization_config(cfg)

    print(f"[Stage 2 BF16] Loading BF16 model from {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _apply_osplus_scales(model, record_dir: str) -> None:
    decoder_layers = get_module(model, "model.layers")
    num_layers = len(decoder_layers)
    scales = load_scales(record_dir, num_layers)
    print(f"[Stage 2 BF16] Loaded OS+SQ scales for {num_layers} layers.")
    print(
        "[Stage 2 BF16] Applying OS+SQ scale fusion "
        "(Group 1: ln \u2192 qkv ; Group 2: v \u2192 o, GQA-aware) "
        "in fp32 intermediate precision ..."
    )
    for layer_idx, layer in enumerate(tqdm(decoder_layers, desc="OS+SQ Fusion")):
        apply_scales_to_layer(layer, layer_idx, scales)
    del scales
    gc.collect()
    clear_accelerator_cache()


def _save_bf16_model_config(model, output_dir: str) -> None:
    # IMPORTANT: do NOT attach quantization_config -- this is a pure BF16
    # checkpoint. ``cfg`` already had it stripped above; remove from the
    # model.config object too in case transformers re-instated it.
    if hasattr(model.config, "quantization_config"):
        try:
            delattr(model.config, "quantization_config")
        except (AttributeError, TypeError):
            pass
    model.config.__dict__.pop("quantization_config", None)
    model.config.torch_dtype = "bfloat16"
    model.config.save_pretrained(output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)


def _stream_bf16_shards(model, output_dir: str, total_parameters: int) -> None:
    print("[Stage 2 BF16] Streaming BF16 shards (no quantization) ...")
    shard_writer = StreamingShardWriter(
        output_dir=output_dir,
        max_shard_size="5GB",
        total_parameters=total_parameters,
    )
    export_pbar = tqdm(
        iter_export_tensors_bf16(model), desc="Export tensors", unit="tensor"
    )
    for name, tensor in export_pbar:
        shard_writer.add_tensor(name, tensor)
        export_pbar.set_postfix_str(
            f"shard={shard_writer.current_size_bytes / 1e9:.2f}GB"
        )
    shard_writer.finalize()


def _copy_tokenizer_and_sidecar_files(files_src: str, output_dir: str) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        files_src, padding_side="left", trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)
    for filename in COPY_FILES:
        src = os.path.join(files_src, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[Stage 2 BF16] Copied {filename}")


def export_model_bf16(
    model_dir: str,
    record_dir: str,
    output_dir: str,
    model_files_dir: str | None = None,
) -> None:
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model = _load_bf16_model(model_dir)
    total_parameters = count_unique_parameters(model)
    _apply_osplus_scales(model, record_dir)
    _save_bf16_model_config(model, output_dir)
    _stream_bf16_shards(model, output_dir, total_parameters)

    files_src = model_files_dir if model_files_dir else model_dir
    _copy_tokenizer_and_sidecar_files(files_src, output_dir)
    print(f"[Stage 2 BF16] Done. Exported fused-scale BF16 model to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="OS+SQ Stage 2 (HF / NPU port, BF16-only): apply scales + "
        "export fused BF16 model (no quantization)."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the source BF16 HuggingFace checkpoint.",
    )
    parser.add_argument(
        "--record_dir",
        required=True,
        help="Directory containing Stage 1 OS+SQ scales (layer_{i}_*_scale.pt).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the exported fused BF16 model.",
    )
    parser.add_argument(
        "--model_files_dir",
        default=None,
        help=(
            "Directory with tokenizer / config files (chat_template, "
            "configuration_minimax_m2.py, etc.). Defaults to --model_dir."
        ),
    )
    args = parser.parse_args()

    export_model_bf16(
        model_dir=args.model_dir,
        record_dir=args.record_dir,
        output_dir=args.output_dir,
        model_files_dir=args.model_files_dir,
    )


if __name__ == "__main__":
    main()
