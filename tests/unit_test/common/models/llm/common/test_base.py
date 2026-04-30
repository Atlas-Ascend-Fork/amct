from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.base import BaseModel


class _StubModel(BaseModel):
    """Subclass that bypasses BaseModel.__init__ to avoid HF model loading."""

    def __init__(self, args=None, quant_target=()):
        # Skip the parent constructor's HF / safetensors machinery.
        self.args = args
        self.quant_target = list(quant_target)
        self.position_ids = None
        self.attention_mask = None
        self.position_embeddings = None
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.model_path = "/fake/path"


# ---- _embed_base_prefix / get_embed_load_specs ---------------------------


def test_embed_base_prefix_default_is_model():
    assert _StubModel()._embed_base_prefix() == "model."


def _build_fake_hf_model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(8, 4)
    model.model.norm = nn.LayerNorm(4)
    model.lm_head = nn.Linear(4, 8, bias=False)
    return model


def test_get_embed_load_specs_uses_lm_head_prefix_when_untied():
    stub = _StubModel()
    stub.config = SimpleNamespace(tie_word_embeddings=False)
    stub.model = _build_fake_hf_model()
    specs = stub.get_embed_load_specs()
    prefixes = [p for _, p in specs]
    assert prefixes == ["model.embed_tokens.", "model.norm.", "lm_head."]


def test_get_embed_load_specs_uses_embed_prefix_when_tied():
    stub = _StubModel()
    stub.config = SimpleNamespace(tie_word_embeddings=True)
    stub.model = _build_fake_hf_model()
    specs = stub.get_embed_load_specs()
    prefixes = [p for _, p in specs]
    assert prefixes == ["model.embed_tokens.", "model.norm.", "model.embed_tokens."]


def test_get_embed_load_specs_falls_back_to_untied_when_attr_missing():
    stub = _StubModel()
    stub.config = SimpleNamespace()  # no tie_word_embeddings
    stub.model = _build_fake_hf_model()
    specs = stub.get_embed_load_specs()
    assert specs[2][1] == "lm_head."


# ---- get_block_forward_kwargs --------------------------------------------


def test_get_block_forward_kwargs_empty_when_nothing_set():
    stub = _StubModel(args=SimpleNamespace(device="cpu"))
    assert stub.get_block_forward_kwargs() == {}


def test_get_block_forward_kwargs_moves_tensors_to_args_device():
    stub = _StubModel(args=SimpleNamespace(device="cpu"))
    stub.position_ids = torch.tensor([[0, 1, 2]])
    stub.attention_mask = torch.ones(1, 3)
    stub.position_embeddings = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))

    kwargs = stub.get_block_forward_kwargs()
    assert set(kwargs) == {"position_ids", "attention_mask", "position_embeddings"}
    assert kwargs["position_ids"].device.type == "cpu"
    assert torch.equal(kwargs["position_ids"], stub.position_ids)
    pe0, pe1 = kwargs["position_embeddings"]
    assert torch.equal(pe0, stub.position_embeddings[0])
    assert torch.equal(pe1, stub.position_embeddings[1])


# ---- iter_ptq_units ------------------------------------------------------


def _make_block_with_attn_and_mlp():
    block = nn.Module()
    block.self_attn = nn.Linear(4, 4)
    block.mlp = nn.Linear(4, 4)
    return block


def test_iter_ptq_units_yields_attn_when_target_is_attn_linear():
    stub = _StubModel(quant_target=["attn-linear"])
    block = _make_block_with_attn_and_mlp()
    units = list(stub.iter_ptq_units(layer_idx=2, block=block))
    assert len(units) == 1
    u = units[0]
    assert u.kind == "attn"
    assert u.name == "self_attn"
    assert u.layer_idx == 2
    assert u.module is block.self_attn


def test_iter_ptq_units_picks_linear_attn_when_block_layer_type_is_linear_attention():
    stub = _StubModel(quant_target=["attn-cache"])
    block = nn.Module()
    block.linear_attn = nn.Linear(4, 4)
    block.layer_type = "linear_attention"
    units = list(stub.iter_ptq_units(layer_idx=0, block=block))
    assert units[0].name == "linear_attn"
    assert units[0].module is block.linear_attn


def test_iter_ptq_units_yields_mlp_when_target_is_mlp():
    stub = _StubModel(quant_target=["mlp"])
    block = _make_block_with_attn_and_mlp()
    units = list(stub.iter_ptq_units(layer_idx=1, block=block))
    assert len(units) == 1
    assert units[0].kind == "mlp" and units[0].name == "mlp"
    assert units[0].module is block.mlp


def test_iter_ptq_units_yields_each_expert_when_target_is_moe():
    stub = _StubModel(quant_target=["moe"])
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])
    units = list(stub.iter_ptq_units(layer_idx=0, block=block))
    assert [u.name for u in units] == ["expert_0", "expert_1", "expert_2"]
    assert all(u.kind == "moe" for u in units)
    assert [u.metadata["expert_idx"] for u in units] == [0, 1, 2]


def test_iter_ptq_units_raises_when_target_is_unsupported_and_block_lacks_mlp():
    stub = _StubModel(quant_target=["nonsense"])
    block = nn.Module()  # no mlp attribute
    with pytest.raises(ValueError, match="Unsupported quant target"):
        list(stub.iter_ptq_units(layer_idx=0, block=block))


# ---- save_block_hook_inputs ----------------------------------------------


def test_save_block_hook_inputs_raises_when_hook_name_is_none():
    stub = _StubModel(quant_target=["mlp"], args=SimpleNamespace(data_dir="/tmp"))
    with pytest.raises(ValueError, match="hook_name cannot be None"):
        stub.save_block_hook_inputs(act_stat={}, hook_name=None, layer_idx=0)


def test_save_block_hook_inputs_chooses_attn_save_target_when_attn_in_quant_target(monkeypatch):
    captured = {}

    def fake_save(act_stat, hook_name, save_target, layer_idx, data_dir):
        captured.update(
            act_stat=act_stat,
            hook_name=hook_name,
            save_target=save_target,
            layer_idx=layer_idx,
            data_dir=data_dir,
        )

    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.save_ptq_inps", fake_save
    )
    stub = _StubModel(quant_target=["attn-linear"], args=SimpleNamespace(data_dir="/d"))
    stub.save_block_hook_inputs({"k": 1}, hook_name="hook", layer_idx=4)
    assert captured["save_target"] == "attn"
    assert captured["layer_idx"] == 4
    assert captured["data_dir"] == "/d"
    assert captured["hook_name"] == "hook"


def test_save_block_hook_inputs_uses_quant_target_for_non_attn(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.save_ptq_inps",
        lambda *a, **k: captured.update(args=a, kwargs=k),
    )
    stub = _StubModel(quant_target=["mlp"], args=SimpleNamespace(data_dir="/d"))
    stub.save_block_hook_inputs({}, hook_name="hook", layer_idx=2)
    # Positional: (act_stat, hook_name, save_target, layer_idx, data_dir)
    assert captured["args"][2] == "mlp"


# ---- load_selected_layer_ptq_params --------------------------------------


def test_load_selected_layer_ptq_params_warns_on_missing_dir(monkeypatch):
    stub = _StubModel(
        quant_target=["mlp"],
        args=SimpleNamespace(
            attn_linear_param_dir="",
            attn_cache_param_dir="",
            moe_mlp_param_dir="",  # empty -> fallback
        ),
    )
    # Should never reach the store when the param_dir is empty.
    stub.ptq_param_store = SimpleNamespace(
        load_layer=lambda *a, **k: pytest.fail("should not be called"),
    )
    results = stub.load_selected_layer_ptq_params(0, block=nn.Module())
    assert results == {"mlp": {"loaded": [], "missing": []}}
    # Original quant_target preserved after the temporary swap.
    assert stub.quant_target == ["mlp"]


def test_load_selected_layer_ptq_params_routes_to_store_per_target():
    stub = _StubModel(
        quant_target=["attn-linear", "mlp"],
        args=SimpleNamespace(
            attn_linear_param_dir="/a",
            attn_cache_param_dir="",
            moe_mlp_param_dir="/m",
        ),
    )
    seen = []

    def fake_load_layer(layer_idx, block, param_dir, strict):
        seen.append((layer_idx, param_dir, list(stub.quant_target)))
        return {"loaded": ["x"], "missing": []}

    stub.ptq_param_store = SimpleNamespace(load_layer=fake_load_layer)
    results = stub.load_selected_layer_ptq_params(7, block=nn.Module())
    assert set(results) == {"attn-linear", "mlp"}
    # quant_target was temporarily set to a single-target list per call.
    assert seen[0][2] == ["attn-linear"]
    assert seen[1][2] == ["mlp"]
    # The original quant_target is restored.
    assert stub.quant_target == ["attn-linear", "mlp"]
