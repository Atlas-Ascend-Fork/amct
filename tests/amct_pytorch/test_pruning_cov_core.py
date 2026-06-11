#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""Unit tests targeting previously-uncovered core lines in the pruning package:
config validation/normalization, batch adapters (context), calib logit extraction,
finetune loss/device/warmup helpers, backend-compat helpers, and the paramless MoE
router patch. All CPU-only, deterministic, tiny inline modules -- no HF downloads,
no NPU.
"""

import torch

torch.set_num_threads(2)

import pytest
import torch.nn as nn

from amct_pytorch.pruning.config import (
    MethodSpec,
    PruneConfig,
    _normalize_method_spec,
    _validate_allocation,
    _validate_method_kwargs,
)
from amct_pytorch.pruning.context import PruneContext, default_batch_adapter
from amct_pytorch.pruning.calib import _unwrap_logits, calib_nll
from amct_pytorch.pruning.finetune import (
    _default_loss,
    _move_to_device,
    _set_warmup_lr,
    prune_finetune,
)
from amct_pytorch.pruning.compat import (
    ModelBackend,
    detect_backend,
    _try_set_int_attr,
    patch_common_config,
)
from amct_pytorch.pruning.methods._moe_mass_common import patch_router_module
from amct_pytorch.pruning.domains.moe import MoETarget


# ===========================================================================
# config.py -- _normalize_method_spec
# ===========================================================================
def test_normalize_from_methodspec_clones_kwargs():
    src = MethodSpec(name="m", kwargs={"prune_ratio": 0.5})
    out = _normalize_method_spec(src)  # line 31
    assert isinstance(out, MethodSpec)
    assert out.name == "m"
    assert out.kwargs == {"prune_ratio": 0.5}
    assert out.kwargs is not src.kwargs  # deep-ish copy of kwargs dict


def test_normalize_from_str():
    out = _normalize_method_spec("low_variance")  # line 33
    assert out.name == "low_variance"
    assert out.kwargs == {}


def test_normalize_from_dict_merges_extra_keys():
    out = _normalize_method_spec(
        {"name": "activation_count", "kwargs": {"a": 1}, "prune_ratio": 0.3}
    )
    assert out.name == "activation_count"
    # extra top-level keys folded into kwargs
    assert out.kwargs == {"a": 1, "prune_ratio": 0.3}


def test_normalize_dict_missing_name_raises():
    with pytest.raises(ValueError):  # line 36
        _normalize_method_spec({"kwargs": {"prune_ratio": 0.1}})


def test_normalize_bad_type_raises():
    with pytest.raises(TypeError):  # line 41
        _normalize_method_spec(123)


# ===========================================================================
# config.py -- _validate_method_kwargs
# ===========================================================================
def test_validate_kwargs_empty_name_raises():
    with pytest.raises(ValueError):  # line 46
        _validate_method_kwargs(MethodSpec(name="", kwargs={}))


def test_validate_kwargs_non_numeric_prune_ratio_raises():
    with pytest.raises(ValueError):  # lines 50-51
        _validate_method_kwargs(MethodSpec(name="m", kwargs={"prune_ratio": "abc"}))


def test_validate_kwargs_out_of_range_prune_ratio_raises():
    with pytest.raises(ValueError):  # lines 55 (>=1.0)
        _validate_method_kwargs(MethodSpec(name="m", kwargs={"prune_ratio": 1.0}))


def test_validate_kwargs_bad_seed_raises():
    with pytest.raises(ValueError):  # lines 59-62
        _validate_method_kwargs(MethodSpec(name="m", kwargs={"seed": "not_int"}))


def test_validate_kwargs_good_values_ok():
    # in-range prune_ratio + integer-like seed should pass without raising
    _validate_method_kwargs(
        MethodSpec(name="m", kwargs={"prune_ratio": 0.5, "seed": 7})
    )


# ===========================================================================
# config.py -- _validate_allocation
# ===========================================================================
def test_validate_allocation_bad_type_raises():
    with pytest.raises(TypeError):  # line 71
        _validate_allocation([("strategy", "uniform")])


def test_validate_allocation_ratio_out_of_range_raises():
    with pytest.raises(ValueError):  # line 80
        _validate_allocation({"ref_ratio": 1.5})


# ===========================================================================
# config.py -- PruneConfig.validate
# ===========================================================================
def test_validate_min_channels_raises():
    with pytest.raises(ValueError):  # line 119
        PruneConfig(min_channels=0).validate()


def test_validate_min_neurons_raises():
    with pytest.raises(ValueError):  # line 121
        PruneConfig(min_neurons=0).validate()


def test_validate_min_experts_raises():
    with pytest.raises(ValueError):  # line 123
        PruneConfig(min_experts=0).validate()


def test_validate_bad_missing_data_policy_raises():
    with pytest.raises(ValueError):  # lines 125
        PruneConfig(missing_data_policy="explode").validate()


def test_validate_bad_stage_error_policy_raises():
    with pytest.raises(ValueError):  # line 129
        PruneConfig(stage_error_policy="explode").validate()


def test_validate_good_config_ok():
    PruneConfig(methods={"dense": "low_variance"}).validate()


# ===========================================================================
# context.py -- default_batch_adapter & iter_model_inputs
# ===========================================================================
def test_iter_model_inputs_none_data_empty():
    ctx = PruneContext(data=None)
    assert list(ctx.iter_model_inputs()) == []  # line 35 path: return iter(())


def test_default_adapter_dict_to_kwargs():
    args, kwargs = default_batch_adapter({"input_ids": torch.zeros(2)})  # line 47
    assert args == ()
    assert "input_ids" in kwargs


def test_default_adapter_tensor():
    t = torch.zeros(3)
    args, kwargs = default_batch_adapter(t)  # line 49
    assert len(args) == 1 and torch.is_tensor(args[0])
    assert kwargs == {}


def test_default_adapter_tuple():
    args, kwargs = default_batch_adapter((1, 2))  # line 51
    assert args == (1, 2)
    assert kwargs == {}


def test_default_adapter_list():
    args, kwargs = default_batch_adapter([1, 2, 3])  # line 53
    assert args == (1, 2, 3)
    assert kwargs == {}


def test_default_adapter_unsupported_raises():
    with pytest.raises(TypeError):  # lines 54-57
        default_batch_adapter(object())


def test_iter_model_inputs_uses_adapter():
    # exercise the generator branch with real data
    ctx = PruneContext(data=[torch.zeros(2), torch.ones(2)])
    out = list(ctx.iter_model_inputs())
    assert len(out) == 2
    assert all(isinstance(a, tuple) and isinstance(k, dict) for a, k in out)


# ===========================================================================
# calib.py -- _unwrap_logits & calib_nll
# ===========================================================================
def test_unwrap_logits_tensor_passthrough():
    t = torch.zeros(2, 3)
    assert _unwrap_logits(t) is t


def test_unwrap_logits_from_tuple():
    t = torch.zeros(2, 3)
    assert _unwrap_logits((t, "extra")) is t  # line 35


def test_unwrap_logits_from_list():
    t = torch.zeros(2, 3)
    assert _unwrap_logits([t]) is t  # line 35


def test_unwrap_logits_from_attr():
    class Out:
        def __init__(self, logits):
            self.logits = logits

    t = torch.zeros(2, 3)
    assert _unwrap_logits(Out(t)) is t  # lines 37-38


def test_unwrap_logits_bad_raises():
    with pytest.raises(TypeError):  # line 39
        _unwrap_logits(object())


def test_calib_nll_empty_batches_raises():
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError):  # line 49
        calib_nll(model, [])


class _TokenLM(nn.Module):
    """Tiny LM: embed token ids -> logits over vocab."""

    def __init__(self, vocab=8, hidden=6):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, ids):
        return self.head(self.emb(ids))


def test_calib_nll_runs_on_integer_batches():
    model = _TokenLM()
    batches = [torch.randint(0, 8, (2, 5)) for _ in range(2)]
    val = calib_nll(model, batches)
    assert isinstance(val, float) and val >= 0.0


def test_calib_nll_float_batch_raises():
    model = _TokenLM()
    with pytest.raises(ValueError):
        calib_nll(model, [torch.randn(2, 5)])


# ===========================================================================
# finetune.py -- _default_loss
# ===========================================================================
class _LossOut:
    def __init__(self, loss=None):
        self.loss = loss


def test_default_loss_via_batch_adapter_ok():
    loss_tensor = torch.tensor(1.0, requires_grad=True)

    class M(nn.Module):
        def forward(self, *a, **k):
            return _LossOut(loss=loss_tensor)

    def adapter(batch):
        return (), {"x": batch}

    out = _default_loss(M(), torch.zeros(2), adapter)  # lines 48-51
    assert out is loss_tensor


def test_default_loss_via_batch_adapter_no_loss_raises():
    class M(nn.Module):
        def forward(self, *a, **k):
            return _LossOut(loss=None)

    def adapter(batch):
        return (), {}

    with pytest.raises(ValueError):  # lines 52-54
        _default_loss(M(), torch.zeros(2), adapter)


def test_default_loss_dict_labels():
    captured = {}

    class M(nn.Module):
        def forward(self, **batch):
            captured.update(batch)
            return _LossOut(loss=torch.tensor(2.0))

    batch = {"input_ids": torch.zeros(2), "labels": torch.zeros(2)}
    out = _default_loss(M(), batch, None)  # lines 56-57
    assert float(out) == 2.0
    assert "labels" in captured


def test_default_loss_dict_input_ids_only():
    seen = {}

    class M(nn.Module):
        def forward(self, **batch):
            seen.update(batch)
            return _LossOut(loss=torch.tensor(3.0))

    batch = {"input_ids": torch.zeros(2)}
    out = _default_loss(M(), batch, None)  # lines 58-59 (labels=input_ids injected)
    assert float(out) == 3.0
    assert "labels" in seen  # labels was injected from input_ids


def test_default_loss_dict_missing_keys_raises():
    class M(nn.Module):
        def forward(self, **batch):
            return _LossOut(loss=torch.tensor(0.0))

    with pytest.raises(ValueError):  # lines 60-62
        _default_loss(M(), {"foo": torch.zeros(2)}, None)


def test_default_loss_xy_tuple_classification():
    model = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    out = _default_loss(model, (x, y), None)  # lines 63-67
    assert out.ndim == 0  # scalar cross-entropy


class _LogitsOut:
    def __init__(self, logits):
        self.logits = logits


def test_default_loss_xy_tuple_with_logits_attr():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 3)

        def forward(self, x):
            return _LogitsOut(self.lin(x))  # line 66: out.logits branch

    m = M()
    out = _default_loss(m, (torch.randn(2, 4), torch.tensor([0, 1])), None)
    assert out.ndim == 0


def test_default_loss_unsupported_raises():
    with pytest.raises(ValueError):  # lines 68-70
        _default_loss(nn.Linear(2, 2), 12345, None)


# ===========================================================================
# finetune.py -- _move_to_device
# ===========================================================================
def test_move_to_device_dict():
    cpu = torch.device("cpu")
    batch = {"a": torch.zeros(2), "b": "str"}
    moved = _move_to_device(batch, cpu)  # lines 74-75
    assert torch.is_tensor(moved["a"])
    assert moved["b"] == "str"


def test_move_to_device_tuple_and_list():
    cpu = torch.device("cpu")
    out_t = _move_to_device((torch.zeros(2), torch.ones(2)), cpu)  # lines 76-78
    assert isinstance(out_t, tuple) and all(torch.is_tensor(x) for x in out_t)
    out_l = _move_to_device([torch.zeros(2)], cpu)
    assert isinstance(out_l, list)


def test_move_to_device_tensor_and_passthrough():
    cpu = torch.device("cpu")
    t = torch.zeros(2)
    assert torch.is_tensor(_move_to_device(t, cpu))  # line 79 tensor branch
    assert _move_to_device("scalar", cpu) == "scalar"  # line 79 passthrough


# ===========================================================================
# finetune.py -- _set_warmup_lr
# ===========================================================================
def test_set_warmup_lr_scales():
    params = [torch.zeros(1, requires_grad=True)]
    opt = torch.optim.SGD(params, lr=0.1)
    _set_warmup_lr(opt, lr=0.1, step=0, warmup=4)  # lines 100-102 scaling
    # step 0 of 4 -> lr * 1/4
    assert abs(opt.param_groups[0]["lr"] - 0.1 * 1 / 4) < 1e-9


def test_set_warmup_lr_noop_when_done():
    params = [torch.zeros(1, requires_grad=True)]
    opt = torch.optim.SGD(params, lr=0.1)
    _set_warmup_lr(opt, lr=0.1, step=10, warmup=4)  # early return branch (line 98-99)
    assert opt.param_groups[0]["lr"] == 0.1


# ===========================================================================
# finetune.py -- prune_finetune raises
# ===========================================================================
def test_prune_finetune_empty_data_raises():
    with pytest.raises(ValueError):  # line 154
        prune_finetune(nn.Linear(2, 2), [], steps=1)


def test_prune_finetune_no_trainable_params_raises():
    model = nn.Linear(2, 2)
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError):  # line 159
        prune_finetune(model, [(torch.randn(2, 2), torch.tensor([0, 1]))], steps=1)


def test_prune_finetune_runs_with_warmup():
    # end-to-end small run that exercises warmup scaling within the train loop
    model = nn.Linear(4, 3)
    data = [(torch.randn(2, 4), torch.tensor([0, 1])) for _ in range(2)]
    hist = prune_finetune(model, data, steps=3, lr=1e-2, warmup=2)
    assert hist["steps"] == 3
    assert len(hist["loss_history"]) == 3


# ===========================================================================
# compat.py -- detect_backend / _try_set_int_attr / patch_common_config
# ===========================================================================
def test_detect_backend_pretrained_module():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = object()

        def save_pretrained(self, *a, **k):
            pass

    backend = detect_backend(M())  # line 45 (pretrained-module)
    assert backend.name == "pretrained-module"
    assert backend.has_save_pretrained and backend.has_config


def test_detect_backend_plain_torch():
    backend = detect_backend(nn.Linear(2, 2))  # line 47 fallthrough
    assert backend.name == "torch"


def _module_named(fake_module):
    """Build an nn.Module instance whose __class__.__module__ is ``fake_module``."""

    class M(nn.Module):
        pass

    M.__module__ = fake_module
    return M()


def test_detect_backend_huggingface():
    backend = detect_backend(_module_named("transformers.models.foo"))  # line 41
    assert backend.name == "huggingface"


def test_detect_backend_modelscope():
    backend = detect_backend(_module_named("modelscope.models.bar"))  # line 43
    assert backend.name == "modelscope"


def test_try_set_int_attr_missing_attr_noop():
    obj = object()
    _try_set_int_attr(obj, "nonexistent", 5)  # line 55 early return
    assert not hasattr(obj, "nonexistent")


def test_try_set_int_attr_bad_value_swallowed():
    class C:
        pass

    obj = C()
    obj.field = 1
    _try_set_int_attr(obj, "field", "not_an_int")  # lines 58-59 except path
    assert obj.field == 1  # unchanged


def test_try_set_int_attr_ok():
    class C:
        pass

    obj = C()
    obj.field = 0
    _try_set_int_attr(obj, "field", "7")
    assert obj.field == 7


def test_patch_common_config_no_config_returns():
    patch_common_config(nn.Linear(2, 2))  # no .config -> early return


def test_patch_common_config_no_meta_returns():
    class C:
        pass

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = C()

    m = M()
    patch_common_config(m)  # config present, no _amct_prune_meta -> line 68 return


def test_patch_common_config_per_layer_widths_warns():
    class Cfg:
        intermediate_size = 100

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()

    m = M()
    m._amct_prune_meta = {"_dense_widths": {10, 20}}  # >1 distinct -> warn path (line 75)
    with pytest.warns(RuntimeWarning):
        patch_common_config(m)


def test_patch_common_config_uniform_sets_scalar():
    class Cfg:
        intermediate_size = 100

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()

    m = M()
    m._amct_prune_meta = {"dense_hidden_size": 42, "moe_num_experts": 3}
    patch_common_config(m)
    assert m.config.intermediate_size == 42


# ===========================================================================
# methods/_moe_mass_common.py -- patch_router_module
# ===========================================================================
class _Router(nn.Module):
    """Toy paramless router exposing group_limited_topk + expert_bias."""

    def __init__(self, num_experts=8):
        super().__init__()
        self.top_k = 2
        self.expert_bias = torch.zeros(num_experts)

        def group_limited_topk(scores):  # placeholder to be patched
            return None

        self.group_limited_topk = group_limited_topk


class _MoEBlock(nn.Module):
    def __init__(self, num_experts=8):
        super().__init__()
        self.gate = _Router(num_experts)
        # a parameter so infer_model_device works
        self.dummy = nn.Parameter(torch.zeros(1))


class _MoEModel(nn.Module):
    def __init__(self, num_experts=8):
        super().__init__()
        self.block = _MoEBlock(num_experts)


def test_patch_router_installs_simple_topk_and_bias():
    model = _MoEModel(num_experts=8)
    target = MoETarget(
        module_path="block",
        router_path="block.gate",
        experts_path="block.experts",
    )
    keep_idx = [0, 1, 2, 3]
    patch_router_module(model, target, keep_idx, top_k=2)  # lines 49-58

    router = model.block.gate
    # _simple_topk installed: returns torch.topk result over kept experts
    scores = torch.randn(5, 4)
    vals, idx = router.group_limited_topk(scores)
    assert vals.shape[-1] == 2  # min(top_k=2, scores.shape[-1]=4)
    # expert_bias reset to len(keep_idx)
    assert router.expert_bias.shape[0] == len(keep_idx)
    assert torch.count_nonzero(router.expert_bias) == 0


def test_patch_router_none_router_returns():
    class NoRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = NoRouter()

    model = Wrap()
    target = MoETarget(
        module_path="block", router_path=None, experts_path="block.experts"
    )
    # router is None (no .gate / .router) -> line 46 early return, no error
    patch_router_module(model, target, [0, 1], top_k=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
