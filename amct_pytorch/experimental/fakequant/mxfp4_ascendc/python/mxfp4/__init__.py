# -*- coding: UTF-8 -*-
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
"""
MXFP4 quant-dequant operator for Ascend NPU (experimental).

Registers torch.ops.amct.quant_dequant_mxfp4 and re-exports the Python wrapper.

Usage:
    import sys
    sys.path.insert(0, ".../mxfp4_ascendc/python")
    from mxfp4 import quant_dequant_mxfp4

    y = quant_dequant_mxfp4(x_npu)
    # or: torch.ops.amct.quant_dequant_mxfp4(x_flat, 1.0)
"""

from __future__ import annotations

__all__ = [
    "quant_dequant_mxfp4",
]

import ctypes
import os

import torch
import torch_npu  # noqa: F401 — registers PrivateUse1 backend

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

_SELF_TEST_DONE = False
_SELF_TEST_OK = False
_SELF_TEST_ERR: str | None = None


def _load_native_libs() -> None:
    """Load kernel .so then the Torch extension that registers torch.ops.amct.*."""
    try:
        _tnpu_lib = os.path.join(
            os.path.dirname(torch_npu.__file__), "lib", "libtorch_npu.so"
        )
        if os.path.isfile(_tnpu_lib):
            ctypes.CDLL(_tnpu_lib, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

    kernel_lib = os.path.join(_PKG_DIR, "libascendc_kernels_npu.so")
    if os.path.isfile(kernel_lib):
        try:
            ctypes.CDLL(kernel_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            raise RuntimeError(
                f"Failed to load Ascend-C kernel library '{kernel_lib}': {e}. "
                "Rebuild with `bash build.sh` (artifacts are copied next to this package)."
            ) from e
    else:
        raise RuntimeError(
            f"Missing Ascend-C kernel library at '{kernel_lib}'. "
            "Run `bash build.sh` from mxfp4_ascendc/ first."
        )

    ops_lib = os.path.join(_PKG_DIR, "libmxfp4_ops.so")
    if not os.path.isfile(ops_lib):
        raise RuntimeError(
            f"Missing Torch extension '{ops_lib}'. "
            "Run `bash build.sh` from mxfp4_ascendc/ first."
        )
    torch.ops.load_library(ops_lib)


def _self_test_or_raise() -> None:
    """Run a fixed-input MXFP4 QDQ on the kernel and verify the output.

    Input ones(32) yields analytically known output 0.75 for scale_factor=6.0.
    """
    global _SELF_TEST_DONE, _SELF_TEST_OK, _SELF_TEST_ERR

    if _SELF_TEST_DONE:
        if not _SELF_TEST_OK:
            raise RuntimeError(
                f"mxfp4 kernel previously failed self-test: {_SELF_TEST_ERR}"
            )
        return
    _SELF_TEST_DONE = True

    try:
        is_npu_available = bool(
            getattr(torch, "npu", None) and torch.npu.is_available()
        )
    except Exception:
        is_npu_available = False
    if not is_npu_available:
        _SELF_TEST_OK = True
        return

    try:
        x = torch.ones(32, dtype=torch.float32, device="npu")
        y = torch.ops.amct.quant_dequant_mxfp4(x, 1.0)
        torch.npu.synchronize()
        diff = (y.float() - 0.75).abs().max().item()
    except Exception as e:
        _SELF_TEST_OK = False
        _SELF_TEST_ERR = f"kernel raised during self-test: {e}"
        raise RuntimeError(
            f"mxfp4 kernel self-test FAILED while invoking the kernel: "
            f"{e}. Check that the compiled kernel SoC matches the runtime "
            f"hardware (`npu-smi info -t board -i 0 -c 0 | "
            f"grep 'NPU Name'`)."
        ) from e

    if not (diff < 1e-5):
        _SELF_TEST_OK = False
        _SELF_TEST_ERR = (
            f"input=ones(32), expected output all 0.75, max_abs_err={diff:.4e}"
        )
        raise RuntimeError(
            "mxfp4 kernel self-test FAILED: input=ones(32), "
            f"expected output all 0.75, max_abs_err={diff:.4e}. This usually "
            "means the compiled kernel SoC does not match the runtime "
            "hardware. Check SoC with "
            "`npu-smi info -t board -i 0 -c 0 | grep 'NPU Name'` and rebuild "
            "with the matching SOC_VERSION (e.g. "
            "SOC_VERSION=Ascend910_9392 bash build.sh)."
        )

    _SELF_TEST_OK = True


_load_native_libs()
_self_test_or_raise()


def __getattr__(name: str):
    """Lazy-export wrappers after native libs are loaded above.

    Importing ``.ops`` only after ``_load_native_libs()`` keeps
    ``torch.ops.amct.*`` registration ordered correctly, while avoiding a
    mid-module import that trips G.FMT.05 / wrong-import-position checks.
    """
    if name == "quant_dequant_mxfp4":
        from .ops import quant_dequant_mxfp4

        return quant_dequant_mxfp4
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
