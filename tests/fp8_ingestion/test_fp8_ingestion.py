# -*- coding: UTF-8 -*-
# Tests for BaseModel FP8 weight ingestion.
#
# Run from the worktree root:
#   cd /mnt/data/x00823151/work/fp8-ingestion-wt
#   python -m pytest tests/fp8_ingestion/ -v

import os
import pytest
import torch

from amct_pytorch.common.models.llm.common.base import (
    BaseModel,
    _CT_FP8_FORMATS,
    _CT_QUANT_METHOD,
)


# ---------------------------------------------------------------------------
# Minimal stub — only needs _detect_source_fp8 and _dequant_fp8_state_dict
# ---------------------------------------------------------------------------

class _StubModel(BaseModel):
    """Lightweight BaseModel subclass that skips HF model loading."""

    def __init__(self, model_path):
        # Bypass full __init__; only wire up the fields we test.
        self.model_path = model_path
        self._source_is_fp8 = self._detect_source_fp8()

    # Abstract stubs — not called in these tests.
    def iter_ptq_units(self): ...
    def get_layer_weight_prefix(self, idx): ...
    def get_num_layers(self): ...
    def cls(self): ...


# ---------------------------------------------------------------------------
# S1: BF16 model — regression, no change
# ---------------------------------------------------------------------------

class TestS1BF16Regression:

    def test_detect_source_not_fp8(self, bf16_model_dir):
        m = _StubModel(bf16_model_dir)
        assert m._source_is_fp8 is False

    def test_dequant_passthrough(self, bf16_model_dir):
        """_dequant_fp8_state_dict must not alter BF16 tensors."""
        m = _StubModel(bf16_model_dir)
        w = torch.randn(4, 8, dtype=torch.bfloat16)
        sd = {"weight": w}
        out = m._dequant_fp8_state_dict(sd)
        assert out["weight"].data_ptr() == w.data_ptr(), "BF16 tensor must not be copied"
        assert out["weight"].dtype == torch.bfloat16

    def test_load_layer_weight_bf16_unchanged(self, bf16_model_dir):
        """load_layer_weight on a BF16 model returns BF16 tensors as-is."""
        m = _StubModel(bf16_model_dir)
        sd = m.load_layer_weight("model.layers.0.mlp.")
        assert len(sd) == 2
        for name, t in sd.items():
            assert t.dtype == torch.bfloat16, f"{name} should stay BF16"


# ---------------------------------------------------------------------------
# S2: CT FP8 — weight_scale (HF compressed-tensors standard)
# ---------------------------------------------------------------------------

class TestS2CTFp8WeightScale:

    def test_detect_source_is_fp8(self, ct_fp8_model_dir):
        model_dir, *_ = ct_fp8_model_dir
        m = _StubModel(model_dir)
        assert m._source_is_fp8 is True

    def test_scale_keys_stripped(self, ct_fp8_model_dir):
        """weight_scale entries must not appear in the output state_dict."""
        model_dir, *_ = ct_fp8_model_dir
        m = _StubModel(model_dir)
        sd = m.load_layer_weight("model.layers.0.mlp.")
        for name in sd:
            assert not name.endswith("_scale"), f"scale key leaked: {name}"

    def test_output_dtype_bfloat16(self, ct_fp8_model_dir):
        model_dir, *_ = ct_fp8_model_dir
        m = _StubModel(model_dir)
        sd = m.load_layer_weight("model.layers.0.mlp.")
        for name, t in sd.items():
            assert t.dtype == torch.bfloat16, f"{name} should be BF16 after dequant"

    def test_numeric_accuracy(self, ct_fp8_model_dir):
        """Dequantised weight should reconstruct original within FP8 rounding error."""
        model_dir, w_bf16_ref, scale_1d = ct_fp8_model_dir
        m = _StubModel(model_dir)
        sd = m.load_layer_weight("model.layers.0.mlp.")
        w_out = sd["gate_proj.weight"].float()
        # FP8 E4M3 relative error ≤ 2^{-3} ≈ 12.5%; allow 15% headroom
        rel_err = (w_out - w_bf16_ref).abs() / (w_bf16_ref.abs() + 1e-6)
        assert rel_err.mean().item() < 0.15, f"mean rel error too large: {rel_err.mean().item():.4f}"


# ---------------------------------------------------------------------------
# S3: DeepSeek-style _scale_inv (reciprocal scale)
# ---------------------------------------------------------------------------

class TestS3DeepSeekScaleInv:

    def test_detect_source_is_fp8(self, ds_fp8_model_dir):
        model_dir, *_ = ds_fp8_model_dir
        m = _StubModel(model_dir)
        assert m._source_is_fp8 is True

    def test_scale_inv_keys_stripped(self, ds_fp8_model_dir):
        model_dir, *_ = ds_fp8_model_dir
        m = _StubModel(model_dir)
        sd = m.load_layer_weight("model.layers.0.mlp.")
        for name in sd:
            assert not name.endswith("_scale_inv"), f"scale_inv key leaked: {name}"

    def test_output_dtype_bfloat16(self, ds_fp8_model_dir):
        model_dir, *_ = ds_fp8_model_dir
        m = _StubModel(model_dir)
        sd = m.load_layer_weight("model.layers.0.mlp.")
        for name, t in sd.items():
            assert t.dtype == torch.bfloat16

    def test_numeric_accuracy(self, ds_fp8_model_dir):
        model_dir, w_bf16_ref, _ = ds_fp8_model_dir
        m = _StubModel(model_dir)
        sd = m.load_layer_weight("model.layers.0.mlp.")
        w_out = sd["gate_proj.weight"].float()
        rel_err = (w_out - w_bf16_ref).abs() / (w_bf16_ref.abs() + 1e-6)
        assert rel_err.mean().item() < 0.15


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_no_scale_logs_warning(self, ct_fp8_model_dir, caplog):
        """FP8 weight without any scale tensor → warning, output still BF16."""
        model_dir, *_ = ct_fp8_model_dir
        m = _StubModel(model_dir)
        w_fp8 = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
        sd = {"weight": w_fp8}   # deliberately no scale key
        import logging
        with caplog.at_level(logging.WARNING):
            out = m._dequant_fp8_state_dict(sd)
        assert out["weight"].dtype == torch.bfloat16
        assert any("no scale" in r.message.lower() for r in caplog.records)

    def test_mixed_bf16_and_fp8(self, ct_fp8_model_dir):
        """State dict with mixed dtypes: FP8 dequanted, BF16 passed through."""
        model_dir, *_ = ct_fp8_model_dir
        m = _StubModel(model_dir)
        w_fp8 = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
        scale = torch.ones(4, dtype=torch.float32)
        w_bf16 = torch.randn(4, 8, dtype=torch.bfloat16)
        sd = {
            "mlp.gate_proj.weight": w_fp8,
            "mlp.gate_proj.weight_scale": scale,
            "mlp.norm.weight": w_bf16,
        }
        out = m._dequant_fp8_state_dict(sd)
        assert "mlp.gate_proj.weight_scale" not in out
        assert out["mlp.gate_proj.weight"].dtype == torch.bfloat16
        assert out["mlp.norm.weight"].dtype == torch.bfloat16
        assert out["mlp.norm.weight"].data_ptr() == w_bf16.data_ptr()
