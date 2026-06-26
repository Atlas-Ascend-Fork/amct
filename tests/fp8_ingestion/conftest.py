# -*- coding: UTF-8 -*-
# Tests for FP8 original-weight ingestion (feat/fp8-weight-ingestion).
#
# Three scenarios:
#   S1 - BF16 model: _source_is_fp8=False, load_layer_weight unchanged (regression)
#   S2 - CT FP8 model with weight_scale: dequant via W * scale
#   S3 - DeepSeek-style FP8 model with weight_scale_inv: dequant via W / scale_inv
#
# All tests use synthetic safetensors fixtures; no real model checkpoint required.

import json
import os
import struct
import tempfile

import pytest
import torch
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Helpers to build minimal fake checkpoints
# ---------------------------------------------------------------------------

def _write_index(model_dir: str, weight_map: dict):
    index = {"metadata": {}, "weight_map": weight_map}
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


def _write_config(model_dir: str, quantization_config: dict = None):
    cfg = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "tie_word_embeddings": False,
    }
    if quantization_config:
        cfg["quantization_config"] = quantization_config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bf16_model_dir(tmp_path):
    """S1: plain BF16 checkpoint, no quantization_config."""
    d = str(tmp_path / "bf16_model")
    os.makedirs(d)
    _write_config(d)
    tensors = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight":   torch.randn(8, 16, dtype=torch.bfloat16),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, os.path.join(d, shard))
    _write_index(d, {k: shard for k in tensors})
    return d


@pytest.fixture()
def ct_fp8_model_dir(tmp_path):
    """S2: CT FP8 checkpoint with weight_scale (HF compressed-tensors standard)."""
    d = str(tmp_path / "ct_fp8_model")
    os.makedirs(d)
    _write_config(d, quantization_config={
        "format": "float-quantized",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {"weights": {"num_bits": 8, "type": "float"}, "targets": ["Linear"]}
        },
    })

    # Simulate per-channel FP8: weight (out, in) + scale (out,)
    w_bf16 = torch.randn(8, 16, dtype=torch.float32)
    scale = w_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0  # E4M3 max
    w_fp8 = (w_bf16 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale_1d = scale.squeeze(-1).to(torch.float32)

    tensors = {
        "model.layers.0.mlp.gate_proj.weight":       w_fp8,
        "model.layers.0.mlp.gate_proj.weight_scale":  scale_1d,
        "model.layers.0.mlp.up_proj.weight":          w_fp8.clone(),
        "model.layers.0.mlp.up_proj.weight_scale":    scale_1d.clone(),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, os.path.join(d, shard))
    _write_index(d, {k: shard for k in tensors})

    # Store original BF16 for numeric comparison
    tmp_path.__class__.bf16_ref = staticmethod(lambda: w_bf16)
    return d, w_bf16, scale_1d


@pytest.fixture()
def ds_fp8_model_dir(tmp_path):
    """S3: DeepSeek-style FP8 with weight_scale_inv (reciprocal scale)."""
    d = str(tmp_path / "ds_fp8_model")
    os.makedirs(d)
    _write_config(d, quantization_config={
        "format": "float-quantized",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {"weights": {"num_bits": 8, "type": "float"}, "targets": ["Linear"]}
        },
    })

    w_bf16 = torch.randn(8, 16, dtype=torch.float32)
    scale = w_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
    w_fp8 = (w_bf16 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale_inv_1d = (1.0 / scale.squeeze(-1)).to(torch.float32)

    tensors = {
        "model.layers.0.mlp.gate_proj.weight":           w_fp8,
        "model.layers.0.mlp.gate_proj.weight_scale_inv": scale_inv_1d,
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, os.path.join(d, shard))
    _write_index(d, {k: shard for k in tensors})
    return d, w_bf16, scale_inv_1d
