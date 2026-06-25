# Adapted from QuIP# (Cornell-RelaxML/quip-sharp), licensed under GPLv3.
# Source: https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py
# Modified for MiniCPM quantization on Ascend NPU.
import os
import time
from os.path import join, dirname

import torch
from safetensors import safe_open

_NATIVE_RUNTIME_PROBE_PRINTED = False

# Lightweight env-gated hadamard profiler (HARNESS_HAD_PROFILE=1). Accumulates
# call count + wall time (with NPU/CUDA sync) across _apply_fast_hadamard, the
# single chokepoint every hadamard call passes through. Used to break down the
# Stage2 fallback bottleneck (hadamard vs forward/backward vs other). Sync makes
# the per-call time honest at the cost of serialising — only enable for profiling.
_HAD_PROFILE = os.environ.get("HARNESS_HAD_PROFILE", "0") == "1"
_HAD_STATS = {"count": 0, "time": 0.0}


def had_profile_stats():
    return dict(_HAD_STATS)


def had_profile_reset():
    _HAD_STATS["count"] = 0
    _HAD_STATS["time"] = 0.0


def _sync():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()

try:
    import fast_hadamard_transform as _fast_had
except ImportError:
    try:
        from fast_hadamard_ascend import hadamard_transform as _fast_hadamard_transform
        from fast_hadamard_ascend import can_use_native_runtime as _can_use_native_runtime
        from fast_hadamard_ascend import native_runtime_not_ready_reason as _native_runtime_not_ready_reason
        from fast_hadamard_ascend import native_runtime_ready as _native_runtime_ready
    except ImportError:
        _fast_had = None
        _fast_hadamard_transform = None
        _can_use_native_runtime = None
        _native_runtime_not_ready_reason = None
        _native_runtime_ready = None
    else:
        _fast_had = None
else:
    _fast_hadamard_transform = _fast_had.hadamard_transform
    _can_use_native_runtime = None
    _native_runtime_not_ready_reason = None
    _native_runtime_ready = None

from lib import utils

hadamard_mats = {}
with safe_open(join(dirname(__file__), "hadamard.safetensors"), framework="pt", device="cpu") as f:
    k_sort = []
    for k in f.keys():
        k_sort.append((int(k), f.get_tensor(k)))
k_sort.sort(key=lambda x:x[0], reverse=True)
hadamard_mats = {x[0]: x[1] for x in k_sort}
del k_sort 


def get_hadK(n, transpose=False):
    hadK, K = None, None
    for prefetch_k in hadamard_mats.keys():
        if n % prefetch_k == 0:
            assert (is_pow2(n//prefetch_k))
            K = prefetch_k
            hadK = hadamard_mats[prefetch_k].T if transpose else hadamard_mats[prefetch_k]
            break
    return hadK, K


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output
    utils.clean()
    if K > 1:
        input = torch.bmm(hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)


def _apply_fast_hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
    global _NATIVE_RUNTIME_PROBE_PRINTED
    if _fast_hadamard_transform is None:
        raise RuntimeError(
            "Hadamard backend is unavailable. Install fast_hadamard_transform "
            "or fast_hadamard_ascend."
        )
    if not _NATIVE_RUNTIME_PROBE_PRINTED:
        runtime_ready = None if _native_runtime_ready is None else _native_runtime_ready()
        runtime_reason = "" if _native_runtime_not_ready_reason is None else _native_runtime_not_ready_reason()
        can_use_native = None if _can_use_native_runtime is None else _can_use_native_runtime(x)
        print(
            "[fast_hadamard probe] "
            f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
            f"runtime_ready={runtime_ready} can_use_native={can_use_native} "
            f"not_ready_reason={runtime_reason}",
            flush=True,
        )
        _NATIVE_RUNTIME_PROBE_PRINTED = True
    if _HAD_PROFILE:
        _sync()
        t0 = time.perf_counter()
        out = _fast_hadamard_transform(x, scale)
        _sync()
        _HAD_STATS["count"] += 1
        _HAD_STATS["time"] += time.perf_counter() - t0
        return out
    return _fast_hadamard_transform(x, scale)


def matmul_hadU_cuda(X, hadK, K, transpose=False):
    if _fast_hadamard_transform is None:
        return matmul_hadU(X, transpose=transpose)

    n = X.shape[-1]
    # Hadamard is run in the input dtype (the X.float() upcast was dropped for
    # speed). Production callers (LinearGuassianLinear / IndexLinear forward) all
    # upcast to fp32 first, so assert the contract:
    # a fp16 Hadamard on 3B's massive-activation outliers would silently corrupt
    # quantization (see project_m8_3b_npu_fp32_works). Relax to X.float() here if
    # an NPU path legitimately needs another dtype.
    assert X.dtype == torch.float32, (
        f"matmul_hadU_cuda expects fp32 input, got {X.dtype} "
        "(callers must upcast before the Hadamard transform; see review #2)"
    )
    if K == 1:
        return _apply_fast_hadamard(X.contiguous(), 1 / (n ** 0.5))

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.contiguous().view(-1, K, n // K)
    input = _apply_fast_hadamard(input.contiguous(), 1 / (n ** 0.5))
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(X.shape)
 

def matmul_hadUt_cuda(X, hadK, K):
    return matmul_hadU_cuda(X, hadK, K, transpose=True)

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)