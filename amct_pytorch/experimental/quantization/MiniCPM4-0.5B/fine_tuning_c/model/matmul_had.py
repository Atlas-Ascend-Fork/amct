import gc
from os.path import join, dirname

import torch
from safetensors import safe_open

try:
    import fast_hadamard_transform as _fast_had
except ImportError:
    try:
        from fast_hadamard_ascend import hadamard_transform as _fast_hadamard_transform
    except ImportError:
        _fast_had = None
        _fast_hadamard_transform = None
    else:
        _fast_had = None
else:
    _fast_hadamard_transform = _fast_had.hadamard_transform

torch._dynamo.config.cache_size_limit = 256
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


def _apply_fast_hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
    if _fast_hadamard_transform is None:
        raise RuntimeError(
            "Hadamard backend is unavailable. Install fast_hadamard_transform "
            "or fast_hadamard_ascend."
        )
    return _fast_hadamard_transform(x, scale)


def _fallback_hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Pure PyTorch Walsh-Hadamard transform on the last dimension (power-of-2 size)."""
    n = x.shape[-1]
    orig_shape = x.shape
    orig_dtype = x.dtype
    x = x.float().reshape(-1, n)
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :].clone()
        b = x[:, :, 1, :].clone()
        x[:, :, 0, :] = a + b
        x[:, :, 1, :] = a - b
        x = x.reshape(-1, n)
        h *= 2
    return (x * scale).to(orig_dtype).reshape(orig_shape)


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2,
                           input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output
    gc.collect()
    if K > 1:
        input = torch.bmm(hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)


def matmul_hadU_cuda(X, hadK, K, transpose=False):
    if _fast_hadamard_transform is None:
        return matmul_hadU(X, transpose=transpose)

    n = X.shape[-1]
    if K == 1:
        return _apply_fast_hadamard(X.contiguous(), n ** (-0.5))
    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().view(-1, K, n // K)
    input = _apply_fast_hadamard(input.contiguous(), n ** (-0.5))
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(X.shape)


def matmul_hadUt_cuda(X, hadK, K):
    return matmul_hadU_cuda(X, hadK, K, transpose=True)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)