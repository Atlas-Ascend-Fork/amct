import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
from types import SimpleNamespace
import amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe as mod
from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe


class _LayerBits:
    def __init__(self, w, a):
        self.default = SimpleNamespace(w=w, a=a)


class _FakePolicy:
    # top-level attn-linear bits
    w_bits = 8
    a_bits = 8
    def __getitem__(self, group):
        return {"moe.shared": _LayerBits(8, 8),
                "moe.routed": _LayerBits(8, 8)}[group]


def _make(quant_target, monkeypatch, tie=False):
    monkeypatch.setattr(mod, "ensure_bit_policy", lambda args: _FakePolicy())
    wf = Qwen3_5Moe.__new__(Qwen3_5Moe)
    wf.quant_target = list(quant_target)
    wf.args = SimpleNamespace()
    wf.config = SimpleNamespace(tie_word_embeddings=tie)
    return wf


LINEAR_ATTN_TARGETS = [
    r"re:.*\.linear_attn\.in_proj_qkv",
    r"re:.*\.linear_attn\.in_proj_z",
    r"re:.*\.linear_attn\.in_proj_a",
    r"re:.*\.linear_attn\.in_proj_b",
    r"re:.*\.linear_attn\.out_proj",
]


def test_bits_scheme_includes_linear_attn_regex(monkeypatch):
    wf = _make(["attn-linear", "moe"], monkeypatch)
    groups = wf.bits_scheme()
    linear_group = next(g for g in groups if "Linear" in g["targets"])
    for t in LINEAR_ATTN_TARGETS:
        assert t in linear_group["targets"], f"missing {t}"
    # MoEGMM group still present
    assert any("MoEGMM" in g["targets"] for g in groups)


def test_bits_scheme_no_linear_attn_when_moe_only(monkeypatch):
    wf = _make(["moe"], monkeypatch)
    groups = wf.bits_scheme()
    linear_group = next(g for g in groups if "Linear" in g["targets"])
    # moe-only (shared_expert) 不应带 linear_attn 正则
    for t in LINEAR_ATTN_TARGETS:
        assert t not in linear_group["targets"]


def test_deploy_ignore_layers_untied_has_lm_head(monkeypatch):
    wf = _make(["attn-linear", "moe"], monkeypatch, tie=False)
    assert "lm_head" in wf.generate_deploy_ignore_layers()


def test_deploy_ignore_layers_tied_omits_lm_head(monkeypatch):
    wf = _make(["attn-linear", "moe"], monkeypatch, tie=True)
    assert "lm_head" not in wf.generate_deploy_ignore_layers()
