from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path
from typing import Any


def _safe_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_environment() -> dict[str, Any]:
    import torch

    packages = {
        name: {
            "installed": _safe_version(name) is not None,
            "version": _safe_version(name),
        }
        for name in ("torch", "transformers", "datasets", "modelscope")
    }

    cuda_info: dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        cuda_info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda_info,
    }


def save_runtime_environment(path: str | Path, environment: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2, ensure_ascii=False)
    return path


def save_tokenizer(tokenizer: Any, save_dir: str | Path) -> tuple[bool, str | None]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer is None:
        return False, "Tokenizer is None; skipping tokenizer save."
    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if not callable(save_pretrained):
        message = "Tokenizer does not expose save_pretrained(...); skipping tokenizer save."
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return False, message
    try:
        save_pretrained(str(save_dir))
    except Exception as exc:
        message = f"Failed to save tokenizer: {exc}"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return False, message
    return True, None
