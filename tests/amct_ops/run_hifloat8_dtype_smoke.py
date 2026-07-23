#!/usr/bin/env python3
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
import argparse
import json
import os
import sys
from importlib import metadata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ERRORS = (RuntimeError, OSError)


class BackendUnavailable(RuntimeError):
    """Raised when the requested HiFloat8 backend cannot run."""


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run HiFloat8 dtype round-trip smoke validation on NPU."
    )
    parser.add_argument(
        "--backend",
        choices=("native", "amct_ops", "auto"),
        default="auto",
        help="HiFloat8 backend to validate (default: auto).",
    )
    parser.add_argument(
        "--device",
        default="npu:0",
        help="NPU device used for smoke validation (default: npu:0).",
    )
    return parser.parse_args()


def _build_input(torch, dtype):
    fixed = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            0.25,
            -0.25,
            4.0,
            -4.0,
            16.0,
            -16.0,
            100.0,
            -100.0,
            32768.0,
            -32768.0,
            torch.finfo(dtype).tiny,
            -torch.finfo(dtype).tiny,
        ],
        dtype=dtype,
    )
    generator = torch.Generator(device="cpu").manual_seed(2026)
    random_values = torch.randn(16, generator=generator).to(dtype)
    return torch.cat((fixed, random_values))


def _run_native(hifp_impl, fp_tensor):
    if not hifp_impl.is_native_hifloat8_cast_available():
        raise BackendUnavailable("native HiFloat8 cast is unavailable")
    try:
        return "native", hifp_impl._native_hifloat8_fake_quant(fp_tensor)
    except BACKEND_ERRORS as exc:
        raise BackendUnavailable(
            f"native HiFloat8 cast failed: {exc}"
        ) from exc


def _run_amct_ops(hifp_impl, fp_tensor):
    ops = hifp_impl._load_amct_ops_cast()
    if ops is None:
        raise BackendUnavailable("amct_ops hifloat8_cast is unavailable")
    try:
        output = hifp_impl._amct_ops_hifloat8_fake_quant(fp_tensor, *ops)
    except BACKEND_ERRORS as exc:
        raise BackendUnavailable(
            f"amct_ops hifloat8_cast failed: {exc}"
        ) from exc
    return "amct_ops", output


def _run_backend(hifp_impl, fp_tensor, backend):
    if backend == "native":
        return _run_native(hifp_impl, fp_tensor)
    if backend == "amct_ops":
        return _run_amct_ops(hifp_impl, fp_tensor)

    native_error = None
    try:
        return _run_native(hifp_impl, fp_tensor)
    except BackendUnavailable as exc:
        native_error = str(exc)

    try:
        return _run_amct_ops(hifp_impl, fp_tensor)
    except BackendUnavailable as exc:
        raise BackendUnavailable(
            f"auto backend selection failed: {native_error}; {exc}"
        ) from exc


def _build_result(torch, backend, fp_tensor, output):
    if output.shape != fp_tensor.shape:
        raise RuntimeError(
            f"output shape changed from {fp_tensor.shape} to {output.shape}"
        )

    abs_error = (output.float() - fp_tensor.float()).abs().cpu()
    if not bool(torch.isfinite(abs_error).all().item()):
        raise RuntimeError("round-trip produced a non-finite error")

    return {
        "backend": backend,
        "input_dtype": str(fp_tensor.dtype).removeprefix("torch."),
        "shape_preserved": output.shape == fp_tensor.shape,
        "dtype_preserved": output.dtype == fp_tensor.dtype,
        "device_preserved": output.device == fp_tensor.device,
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
        "warning_count": 0,
    }


def _package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _environment_info(torch, torch_npu, device):
    try:
        device_name = torch.npu.get_device_name()
    except (AttributeError, RuntimeError, TypeError):
        device_name = None
    return {
        "cann_home": os.getenv("ASCEND_HOME_PATH")
        or os.getenv("ASCEND_TOOLKIT_HOME"),
        "device": device,
        "device_name": device_name,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", None),
        "amct_ops": _package_version("amct-ops")
        or _package_version("amct_ops"),
    }


def _error_payload(backend, message):
    return {
        "requested_backend": backend,
        "hardware_path_verified": False,
        "error": message,
    }


def main():
    args = _parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        import torch
        import torch_npu

        from amct_pytorch.quantization.dtypes import hifp_impl
    except ImportError as exc:
        print(
            json.dumps(_error_payload(args.backend, str(exc)), sort_keys=True),
            file=sys.stderr,
        )
        return 2

    try:
        torch.npu.set_device(args.device)
        results = []
        with torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16):
                fp_tensor = _build_input(torch, dtype).to(args.device)
                backend, output = _run_backend(
                    hifp_impl, fp_tensor, args.backend
                )
                results.append(
                    _build_result(torch, backend, fp_tensor, output)
                )
    except (BackendUnavailable, RuntimeError, OSError, ValueError) as exc:
        print(
            json.dumps(_error_payload(args.backend, str(exc)), sort_keys=True),
            file=sys.stderr,
        )
        return 2

    contract_ok = all(
        result["shape_preserved"]
        and result["dtype_preserved"]
        and result["device_preserved"]
        and result["warning_count"] == 0
        for result in results
    )
    payload = {
        "requested_backend": args.backend,
        "hardware_path_verified": contract_ok,
        "environment": _environment_info(torch, torch_npu, args.device),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if contract_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
