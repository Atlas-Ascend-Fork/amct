from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

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
    hint = getattr(model, "_amct_prune_backend_hint", None)
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


def _try_set_int_attr(obj: Any, name: str, value: Any) -> None:
    if not hasattr(obj, name):
        return
    try:
        setattr(obj, name, int(value))
    except (TypeError, ValueError):
        pass


def patch_common_config(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    meta = getattr(model, "_amct_prune_meta", None)
    if not isinstance(meta, dict):
        return

    dense_hidden = meta.get("dense_hidden_size")
    if dense_hidden is not None:
        for name in ("intermediate_size", "ffn_hidden_size", "n_inner"):
            _try_set_int_attr(config, name, dense_hidden)

    moe_experts = meta.get("moe_num_experts")
    if moe_experts is not None:
        for name in ("num_local_experts", "num_experts"):
            _try_set_int_attr(config, name, moe_experts)


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
            fallback = {"class_name": model.__class__.__name__, "state_dict": model.state_dict()}
            torch.save(fallback, save_dir / "model_full.pt")

    meta = getattr(model, "_amct_prune_meta", {})
    if report is not None:
        meta = {**meta, "report": report.as_dict()}
    with open(save_dir / "amct_prune_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return save_dir


def _report_from_dict(data: dict) -> PruneReport:
    from .report import StageEvent
    report = PruneReport(
        backend=data.get("backend", "unknown"),
        params_before=data.get("params_before", 0),
        params_after=data.get("params_after", 0),
    )
    for ev in data.get("events", []):
        report.events.append(StageEvent(
            stage=ev.get("stage", ""),
            method=ev.get("method", ""),
            module=ev.get("module", ""),
            detail=ev.get("detail", ""),
        ))
    report.warnings = list(data.get("warnings", []))
    for path, sparsity in data.get("per_layer_sparsity", {}).items():
        report.per_layer_sparsity[path] = float(sparsity)
    return report


def load_pruned_model(save_dir: str | Path) -> Tuple[nn.Module, PruneReport]:
    save_dir = Path(save_dir)
    meta_path = save_dir / "amct_prune_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    report_data = meta.get("report", {})
    report = _report_from_dict(report_data)

    model_full_path = save_dir / "model_full.pt"
    model_pt_path = save_dir / "model.pt"

    if model_full_path.exists():
        loaded = torch.load(model_full_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, nn.Module):
            return loaded, report
        if isinstance(loaded, dict) and "state_dict" in loaded:
            raise ValueError(
                "model_full.pt contains a state_dict without model architecture. "
                "Cannot reconstruct the model without the original class definition."
            )
        raise ValueError(f"Unexpected content in model_full.pt: {type(loaded)}")

    if model_pt_path.exists():
        raise ValueError(
            "Only model.pt (state_dict) found. Cannot reconstruct the model "
            "without the original class definition. Use model_full.pt instead."
        )

    raise FileNotFoundError(f"No model file found in {save_dir}")
