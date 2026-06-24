"""
Stage 2: apply SmoothQuant using recorded activations, then export MXFP4 weights.

This stage does not perform any calibration forward pass. It only:
1. loads the bf16 model,
2. loads Stage 1 activation records,
3. computes SmoothQuant scales,
4. fuses scales into weights,
5. quantizes eligible linear weights to MXFP4,
6. exports a HuggingFace safetensors checkpoint.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    ALPHA as DEFAULT_ALPHA,
    ATTN_IMPLEMENTATION,
    EXCLUDE_PATTERNS,
    GROUP_SIZE,
    IGNORE_STATE_DICT_KEYS,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    SCALE_CLAMP_MIN,
    SCALING_LAYERS,
    load_device_map as load_device_map_config,
    matches_exclude,
)
from mxfp4_quantizer import mxfp4_quantize


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

QUARK_VERSION = "0.11.1+210bbb7"


def get_module(model, path: str):
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _iter_scale_keys(num_layers: int):
    for layer_idx in range(num_layers):
        for cfg in SCALING_LAYERS:
            for linear_name in cfg["layers"]:
                yield f"layers.{layer_idx}.{linear_name}"


def _load_one_scale(record_dir: str, key: str) -> torch.Tensor:
    filename = os.path.join(record_dir, f"{key.replace('.', '_')}.pt")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing activation record: {filename}")
    return torch.load(filename, map_location="cpu", weights_only=True)


def load_act_scales(record_dir: str, num_layers: int) -> dict[str, torch.Tensor]:
    return {
        key: _load_one_scale(record_dir, key)
        for key in _iter_scale_keys(num_layers)
    }


def _compute_smooth_scale(
    act_scale: torch.Tensor, weight_scale: torch.Tensor, alpha: float
) -> torch.Tensor:
    return (act_scale.pow(alpha) / weight_scale.pow(1 - alpha)).clamp(
        min=SCALE_CLAMP_MIN
    )


def _smooth_attn_input(
    layer, layer_idx: int, act_scales: dict[str, torch.Tensor], alpha: float
) -> None:
    ln = get_module(layer, "input_layernorm")
    q_proj = get_module(layer, "self_attn.q_proj")
    k_proj = get_module(layer, "self_attn.k_proj")
    v_proj = get_module(layer, "self_attn.v_proj")
    q_w, k_w, v_w = q_proj.weight, k_proj.weight, v_proj.weight

    act_scale = (
        act_scales[f"layers.{layer_idx}.self_attn.q_proj"]
        .to(q_w.device)
        .to(q_w.dtype)
    )
    weight_scale = (
        torch.cat(
            [
                q_w.abs().max(dim=0, keepdim=True)[0],
                k_w.abs().max(dim=0, keepdim=True)[0],
                v_w.abs().max(dim=0, keepdim=True)[0],
            ],
            dim=0,
        )
        .max(dim=0)[0]
        .clamp(min=1e-5)
    )
    best_scale = _compute_smooth_scale(act_scale, weight_scale, alpha)

    ln.weight.data.div_(best_scale.to(ln.weight.device))
    for proj, w in ((q_proj, q_w), (k_proj, k_w), (v_proj, v_w)):
        proj.weight.data.mul_(best_scale.to(w.device).view(1, -1))


def _apply_gqa_aware_v_o(v_proj, o_proj, best_scale: torch.Tensor) -> None:
    head_dim = best_scale.numel() // NUM_ATTENTION_HEADS
    num_head_repeats = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
    grouped = best_scale.view(NUM_KEY_VALUE_HEADS, num_head_repeats, head_dim)
    prev_scales_tmp = grouped.max(dim=1, keepdim=True)[0]
    v_scales = prev_scales_tmp.reshape(-1)
    o_scales = prev_scales_tmp.expand(
        NUM_KEY_VALUE_HEADS, num_head_repeats, head_dim
    ).reshape(-1)
    v_proj.weight.data.div_(v_scales.to(v_proj.weight.device).view(-1, 1))
    if v_proj.bias is not None:
        v_proj.bias.data.div_(v_scales.to(v_proj.bias.device))
    o_proj.weight.data.mul_(o_scales.to(o_proj.weight.device).view(1, -1))


def _apply_plain_v_o(v_proj, o_proj, best_scale: torch.Tensor) -> None:
    v_proj.weight.data.div_(best_scale.to(v_proj.weight.device).view(-1, 1))
    if v_proj.bias is not None:
        v_proj.bias.data.div_(best_scale.to(v_proj.bias.device))
    o_proj.weight.data.mul_(best_scale.to(o_proj.weight.device).view(1, -1))


def _smooth_v_to_o(
    layer, layer_idx: int, act_scales: dict[str, torch.Tensor], alpha: float
) -> None:
    v_proj = get_module(layer, "self_attn.v_proj")
    o_proj = get_module(layer, "self_attn.o_proj")
    act_scale = (
        act_scales[f"layers.{layer_idx}.self_attn.o_proj"]
        .to(o_proj.weight.device)
        .to(o_proj.weight.dtype)
    )
    weight_scale = (
        o_proj.weight.abs().max(dim=0, keepdim=True)[0].squeeze(0).clamp(min=1e-5)
    )
    best_scale = _compute_smooth_scale(act_scale, weight_scale, alpha)

    num_head_repeats = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
    is_gqa = (
        v_proj.weight.shape[0] != best_scale.numel() and num_head_repeats != 1
    )
    if is_gqa:
        _apply_gqa_aware_v_o(v_proj, o_proj, best_scale)
    else:
        _apply_plain_v_o(v_proj, o_proj, best_scale)


def apply_smoothquant_to_layer(
    layer,
    layer_idx: int,
    act_scales: dict[str, torch.Tensor],
    alpha: float,
) -> None:
    _smooth_attn_input(layer, layer_idx, act_scales, alpha)
    _smooth_v_to_o(layer, layer_idx, act_scales, alpha)


def clear_accelerator_cache() -> None:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()


def iter_export_tensors(model):
    exported_names: set[str] = set()

    for name, param in model.named_parameters():
        if name in IGNORE_STATE_DICT_KEYS:
            continue
        exported_names.add(name)
        if name.endswith(".weight") and param.dim() == 2:
            module_name = name.rsplit(".weight", 1)[0]
            module = get_module(model, module_name)
            if isinstance(module, torch.nn.Linear) and not matches_exclude(
                module_name, EXCLUDE_PATTERNS
            ):
                packed, scale_e8m0 = mxfp4_quantize(
                    param.data.detach().cpu().float(), axis=-1
                )
                yield name, packed.contiguous()
                yield name.replace(".weight", ".weight_scale"), scale_e8m0.contiguous()
                continue
        yield name, param.data.detach().cpu().clone()

    for name, buf in model.named_buffers():
        if name in IGNORE_STATE_DICT_KEYS or name in exported_names:
            continue
        yield name, buf.data.detach().cpu().clone()


def count_unique_parameters(model) -> int:
    total = 0
    seen: set[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]] = set()

    for name, param in model.named_parameters():
        if name in IGNORE_STATE_DICT_KEYS:
            continue

        # MiniMax remote code can expose aliased parameters via distinct wrappers.
        # Deduplicate by underlying storage + view metadata so exported metadata
        # matches the actual unique parameter count.
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


class StreamingShardWriter:
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
            f"[Stage 2] Saving temporary shard {shard_idx} "
            f"({len(self.current_tensors)} tensors, {self.current_size_bytes / 1e9:.2f} GB)"
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


def _build_tensor_quant_cfg() -> dict:
    return {
        "ch_axis": -1,
        "dtype": "fp4",
        "group_size": GROUP_SIZE,
        "is_dynamic": False,
        "is_scale_quant": False,
        "mx_element_dtype": None,
        "observer_cls": "PerBlockMXObserver",
        "qscheme": "per_group",
        "round_method": "half_even",
        "scale_calculation_mode": "even",
        "scale_format": "e8m0",
        "scale_type": "float",
        "symmetric": None,
    }


def _build_algo_config(alpha: float) -> list[dict]:
    return [
        {
            "name": "smooth",
            "alpha": alpha,
            "scale_clamp_min": SCALE_CLAMP_MIN,
            "model_decoder_layers": "model.layers",
            "scaling_layers": SCALING_LAYERS,
        }
    ]


def _build_export_cfg() -> dict:
    return {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    }


def _build_global_quant_cfg(tensor_cfg: dict, input_cfg: dict) -> dict:
    return {
        "bias": None,
        "input_tensors": input_cfg,
        "output_tensors": None,
        "target_device": None,
        "weight": tensor_cfg,
    }


def build_quantization_config(alpha: float) -> dict:
    tensor_quant_cfg = _build_tensor_quant_cfg()
    input_quant_cfg = {**tensor_quant_cfg, "is_dynamic": True}
    return {
        "algo_config": _build_algo_config(alpha),
        "exclude": EXCLUDE_PATTERNS,
        "export": _build_export_cfg(),
        "global_quant_config": _build_global_quant_cfg(
            tensor_quant_cfg, input_quant_cfg
        ),
        "kv_cache_post_rope": False,
        "kv_cache_quant_config": {},
        "layer_quant_config": {},
        "layer_type_quant_config": {},
        "quant_method": "quark",
        "quant_mode": "eager_mode",
        "softmax_quant_spec": None,
        "version": QUARK_VERSION,
    }


def resolve_load_device_map(
    load_device_mode: str, device_map_file: str | None
) -> str | dict[str, str | int]:
    if device_map_file is not None:
        return load_device_map_config(device_map_file)
    if load_device_mode == "auto":
        return "auto"
    return {"": "cpu"}


def _load_bf16_model(model_dir: str, device_map):
    from transformers import AutoModelForCausalLM

    print(f"[Stage 2] Loading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _apply_smoothquant_all_layers(decoder_layers, act_scales, alpha: float) -> None:
    print(f"[Stage 2] Applying SmoothQuant with alpha={alpha}...")
    for layer_idx, layer in enumerate(tqdm(decoder_layers, desc="SmoothQuant")):
        apply_smoothquant_to_layer(layer, layer_idx, act_scales, alpha)


def _prepare_output_dir(output_dir: str, model, alpha: float) -> None:
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.config.quantization_config = build_quantization_config(alpha)
    model.config.save_pretrained(output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)


def _stream_export_weights(model, output_dir: str, total_parameters: int) -> None:
    print(
        "[Stage 2] Quantizing weights to MXFP4 and streaming shards to safetensors..."
    )
    shard_writer = StreamingShardWriter(
        output_dir=output_dir,
        max_shard_size="5GB",
        total_parameters=total_parameters,
    )
    export_pbar = tqdm(iter_export_tensors(model), desc="Export tensors", unit="tensor")
    for name, tensor in export_pbar:
        shard_writer.add_tensor(name, tensor)
        export_pbar.set_postfix_str(
            f"current_shard={shard_writer.current_size_bytes / 1e9:.2f}GB"
        )
    shard_writer.finalize()


def _copy_aux_files(model_dir: str, output_dir: str) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, padding_side="left", trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)

    for filename in COPY_FILES:
        src = os.path.join(model_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def export_model(
    model_dir: str,
    record_dir: str,
    output_dir: str,
    load_device_mode: str = "cpu",
    device_map_file: str | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    device_map = resolve_load_device_map(load_device_mode, device_map_file)
    model = _load_bf16_model(model_dir, device_map)
    total_parameters = count_unique_parameters(model)

    decoder_layers = get_module(model, "model.layers")
    act_scales = load_act_scales(record_dir, len(decoder_layers))
    print(f"[Stage 2] Loaded {len(act_scales)} activation scale tensors")

    _apply_smoothquant_all_layers(decoder_layers, act_scales, alpha)
    _prepare_output_dir(output_dir, model, alpha)

    del act_scales
    gc.collect()
    clear_accelerator_cache()

    _stream_export_weights(model, output_dir, total_parameters)
    _copy_aux_files(model_dir, output_dir)

    print(f"[Stage 2] Exporting to {output_dir}...")
    print(f"[Stage 2] Done. Exported model to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: SmoothQuant + MXFP4 export")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--record_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--load_device_map", default="cpu", choices=["cpu", "auto"])
    parser.add_argument("--device_map_file", default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    export_model(
        model_dir=args.model_dir,
        record_dir=args.record_dir,
        output_dir=args.output_dir,
        load_device_mode=args.load_device_map,
        device_map_file=args.device_map_file,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
