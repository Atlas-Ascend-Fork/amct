from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .report import PruneReport


@dataclass(slots=True)
class ModelBackend:
    name: str
    has_save_pretrained: bool
    has_config: bool


COMMON_CONFIG_NAMES = {
    "intermediate_size",
    "ffn_hidden_size",
    "n_inner",
    "num_local_experts",
    "num_experts",
}


def detect_backend(model: nn.Module) -> ModelBackend:
    hint = getattr(model, "_ascend_prune_backend_hint", None)
    module_name = model.__class__.__module__.lower()
    has_save_pretrained = hasattr(model, "save_pretrained") and callable(getattr(model, "save_pretrained"))
    has_config = hasattr(model, "config")

    if hint in {"huggingface", "modelscope", "pretrained-module", "torch"}:
        name = str(hint)
    elif "transformers" in module_name:
        name = "huggingface"
    elif "modelscope" in module_name:
        name = "modelscope"
    elif has_save_pretrained and has_config:
        name = "pretrained-module"
    else:
        name = "torch"
    return ModelBackend(name=name, has_save_pretrained=has_save_pretrained, has_config=has_config)


def patch_common_config(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    meta = getattr(model, "_ascend_prune_meta", None)
    if not isinstance(meta, dict):
        return

    dense_hidden = meta.get("dense_hidden_size")
    moe_experts = meta.get("moe_num_experts")

    if dense_hidden is not None:
        for name in ("intermediate_size", "ffn_hidden_size", "n_inner"):
            if hasattr(config, name):
                try:
                    setattr(config, name, int(dense_hidden))
                except Exception:
                    pass

    if moe_experts is not None:
        for name in ("num_local_experts", "num_experts"):
            if hasattr(config, name):
                try:
                    setattr(config, name, int(moe_experts))
                except Exception:
                    pass


def save_model(model: nn.Module, save_dir: str | Path, report: PruneReport | None = None) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    patch_common_config(model)

    if hasattr(model, "save_pretrained") and callable(getattr(model, "save_pretrained")):
        model.save_pretrained(str(save_dir))
    else:
        torch.save(model.state_dict(), save_dir / "model.pt")
        try:
            torch.save(model, save_dir / "model_full.pt")
        except Exception:
            torch.save({"class_name": model.__class__.__name__, "state_dict": model.state_dict()}, save_dir / "model_full.pt")

    meta = getattr(model, "_ascend_prune_meta", {})
    if report is not None:
        meta = {**meta, "report": report.as_dict()}
    with open(save_dir / "ascend_prune_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return save_dir
