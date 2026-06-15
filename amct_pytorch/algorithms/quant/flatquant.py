import torch
import torch.nn as nn

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY


def _random_orthogonal(size: int, *, device=None, dtype=None) -> torch.Tensor:
    matrix = torch.randn(size, size, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix)
    diag = torch.sign(torch.diag(r))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    q = q @ torch.diag(diag)
    if dtype is not None:
        q = q.to(dtype)
    return q


def _kronecker_matmul(x: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    init_shape = x.shape
    x = x.reshape(-1, left.shape[0], right.shape[0])
    x = torch.matmul(x, right.to(x))
    x = torch.matmul(left.to(x).transpose(0, 1), x)
    return x.reshape(init_shape)


class _InvFlatSingleTransform(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False)
        self.linear.weight.data.copy_(_random_orthogonal(size).to(self.linear.weight))
        self._eval_mode = False

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        init_shape = x.shape
        if not self._eval_mode:
            matrix = self.linear.weight
            if inv_t:
                matrix = torch.linalg.inv(matrix).transpose(0, 1)
        else:
            if inv_t:
                matrix = self.matrix_inv_t
            else:
                matrix = self.matrix

        matrix = matrix.to(device=x.device, dtype=x.dtype)
        x = x.reshape(-1, matrix.shape[0])
        return x.matmul(matrix).reshape(init_shape)

    def to_eval_mode(self):
        if not self._eval_mode:
            matrix = self.linear.weight
            matrix_inv_t = torch.linalg.inv(matrix).transpose(0, 1)
            self.matrix = nn.Parameter(matrix, requires_grad=False)
            self.matrix_inv_t = nn.Parameter(matrix_inv_t, requires_grad=False)
            del self.linear
            self._eval_mode = True


class _InvFlatDecomposeTransform(nn.Module):
    def __init__(self, left_size: int, right_size: int, add_diag: bool):
        super().__init__()
        self.add_diag = add_diag
        self.linear_left = nn.Linear(left_size, left_size, bias=False)
        self.linear_left.weight.data.copy_(_random_orthogonal(left_size).to(self.linear_left.weight))

        self.linear_right = nn.Linear(right_size, right_size, bias=False)
        self.linear_right.weight.data.copy_(_random_orthogonal(right_size).to(self.linear_right.weight))

        if add_diag:
            self.diag_scale = nn.Parameter(torch.ones(left_size * right_size, dtype=torch.float32), requires_grad=True)
        self._eval_mode = False

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        if self.add_diag:
            diag_scale = self.diag_scale.to(device=x.device, dtype=x.dtype)
            if inv_t:
                x = x / diag_scale
            else:
                x = x * diag_scale

        if not self._eval_mode:
            matrix_left, matrix_right = self.linear_left.weight, self.linear_right.weight
            if inv_t:
                matrix_left = torch.linalg.inv(matrix_left).transpose(0, 1)
                matrix_right = torch.linalg.inv(matrix_right).transpose(0, 1)
        else:
            matrix_left, matrix_right = self.matrix_left, self.matrix_right
            if inv_t:
                matrix_left, matrix_right = self.matrix_left_inv, self.matrix_right_inv
        return _kronecker_matmul(x, matrix_left, matrix_right)

    def to_eval_mode(self):
        if not self._eval_mode:
            self.matrix_left = nn.Parameter(self.linear_left.weight, requires_grad=False)
            self.matrix_right = nn.Parameter(self.linear_right.weight, requires_grad=False)
            self.matrix_left_inv = nn.Parameter(torch.linalg.inv(self.linear_left.weight).transpose(0, 1), requires_grad=False)
            self.matrix_right_inv = nn.Parameter(torch.linalg.inv(self.linear_right.weight).transpose(0, 1), requires_grad=False)
            del self.linear_left, self.linear_right
            self._eval_mode = True

class _SVDFlatSingleTransform(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear_u = nn.Linear(size, size, bias=False)
        self.linear_u.weight.data.copy_(_random_orthogonal(size).to(self.linear_u.weight))
        self.linear_u = nn.utils.parametrizations.orthogonal(self.linear_u, orthogonal_map="cayley", use_trivialization=False)
        self.linear_v = nn.Linear(size, size, bias=False)
        self.linear_v.weight.data.copy_(_random_orthogonal(size).to(self.linear_v.weight))
        self.linear_v = nn.utils.parametrizations.orthogonal(self.linear_v, orthogonal_map="cayley", use_trivialization=False)
        self.linear_diag = torch.nn.Parameter(torch.ones(size, dtype=torch.float32), requires_grad=True)

        self._eval_mode = False

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        init_shape = x.shape
        if not self._eval_mode:
            orthog_u =  self.linear_u.weight
            orthog_v = self.linear_v.weight
            linear_diag = self.linear_diag
            if inv_t:
                linear_diag = 1 / linear_diag
            matrix = orthog_u @ torch.diag(linear_diag) @ orthog_v.t()

        else:
            if inv_t:
                matrix = self.matrix_inv_t
            else:
                matrix = self.matrix

        matrix = matrix.to(device=x.device, dtype=x.dtype)
        x = x.reshape(-1, matrix.shape[0])
        return x.matmul(matrix).reshape(init_shape)

    def to_eval_mode(self):
        if not self._eval_mode:
            matrix = self.linear_u.weight @ torch.diag(self.linear_diag) @ self.linear_v.weight.t()
            matrix_inv_t = self.linear_u.weight @ torch.diag(1 / self.linear_diag) @ self.linear_v.weight.t()
            self.matrix = nn.Parameter(matrix, requires_grad=False)
            self.matrix_inv_t = nn.Parameter(matrix_inv_t, requires_grad=False)
            del self.linear_u, self.linear_diag, self.linear_v
            self._eval_mode = True


class _SVDFlatDecomposeTransform(nn.Module):
    def __init__(self, left_size: int, right_size: int, add_diag: bool):
        super().__init__()
        self.add_diag = add_diag
        self.linear_u_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_u_left.weight.data.copy_(_random_orthogonal(left_size).to(self.linear_u_left.weight))
        self.linear_u_left = nn.utils.parametrizations.orthogonal(self.linear_u_left, orthogonal_map="cayley", use_trivialization=False)
        self.linear_v_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_v_left.weight.data.copy_(_random_orthogonal(left_size).to(self.linear_v_left.weight))
        self.linear_v_left = nn.utils.parametrizations.orthogonal(self.linear_v_left, orthogonal_map="cayley", use_trivialization=False)
        self.linear_diag_left = torch.nn.Parameter(torch.ones(left_size, dtype=torch.float32), requires_grad=True)

        self.linear_u_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_u_right.weight.data.copy_(_random_orthogonal(right_size).to(self.linear_u_right.weight))
        self.linear_u_right = nn.utils.parametrizations.orthogonal(self.linear_u_right, orthogonal_map="cayley", use_trivialization=False)
        self.linear_v_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_v_right.weight.data.copy_(_random_orthogonal(right_size).to(self.linear_v_right.weight))
        self.linear_v_right = nn.utils.parametrizations.orthogonal(self.linear_v_right, orthogonal_map="cayley", use_trivialization=False)
        self.linear_diag_right = torch.nn.Parameter(torch.ones(right_size, dtype=torch.float32), requires_grad=True)

        if add_diag:
            self.diag_scale = nn.Parameter(torch.ones(left_size * right_size, dtype=torch.float32), requires_grad=True)
        self._eval_mode = False

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        if self.add_diag:
            diag_scale = self.diag_scale.to(device=x.device, dtype=x.dtype)
            if inv_t:
                x = x / diag_scale
            else:
                x = x * diag_scale

        if not self._eval_mode:
            matrix_u_left, matrix_u_right = self.linear_u_left.weight, self.linear_u_right.weight
            matrix_v_left, matrix_v_right = self.linear_v_left.weight, self.linear_v_right.weight
            linear_diag_left, linear_diag_right = self.linear_diag_left, self.linear_diag_right
            if inv_t:
                linear_diag_left, linear_diag_right = 1 / linear_diag_left, 1 / linear_diag_right
        else:
            matrix_left, matrix_right = self.matrix_left, self.matrix_right
            if inv_t:
                matrix_left, matrix_right = self.matrix_left_inv, self.matrix_right_inv
                return _kronecker_matmul(x, matrix_left, matrix_right)
        matrix_left = matrix_u_left @ torch.diag(linear_diag_left) @ matrix_v_left.t()
        matrix_right = matrix_u_right @ torch.diag(linear_diag_right) @ matrix_v_right.t()
        return _kronecker_matmul(x, matrix_left, matrix_right)

    def to_eval_mode(self):
        if not self._eval_mode:
            matrix_left = self.linear_u_left.weight @ torch.diag(self.linear_diag_left) @ self.linear_v_left.weight.t()
            matrix_right = self.linear_u_right.weight @ torch.diag(self.linear_diag_right) @ self.linear_v_right.weight.t()
            matrix_left_inv = self.linear_u_left.weight @ torch.diag(1 / self.linear_diag_left) @ self.linear_v_left.weight.t()
            matrix_right_inv = self.linear_u_right.weight @ torch.diag(1 / self.linear_diag_right) @ self.linear_v_right.weight.t()
            self.matrix_left = nn.Parameter(matrix_left, requires_grad=False)
            self.matrix_right = nn.Parameter(matrix_right, requires_grad=False)
            self.matrix_left_inv = nn.Parameter(matrix_left_inv, requires_grad=False)
            self.matrix_right_inv = nn.Parameter(matrix_right_inv, requires_grad=False)
            del self.linear_u_left, self.linear_diag_left, self.linear_v_left, self.linear_u_right, self.linear_diag_right, self.linear_v_right
            self._eval_mode = True

class _FlatDecomposeTransform(nn.Module):
    def __init__(self, left_size: int, right_size: int, add_diag: bool):
        super().__init__()
        self.left_size = left_size
        self.right_size = right_size
        self.add_diag = add_diag

        self.linear_left = nn.Linear(left_size, left_size, bias=False)
        self.linear_left.weight.data.copy_(_random_orthogonal(left_size).to(self.linear_left.weight))

        self.linear_right = nn.Linear(right_size, right_size, bias=False)
        self.linear_right.weight.data.copy_(_random_orthogonal(right_size).to(self.linear_right.weight))

        if add_diag:
            self.diag_scale = nn.Parameter(torch.ones(left_size * right_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        if self.add_diag:
            diag_scale = self.diag_scale.to(device=x.device, dtype=x.dtype)
            if inv_t:
                x = x / diag_scale
            else:
                x = x * diag_scale

        matrix_left = self.linear_left.weight
        matrix_right = self.linear_right.weight
        if inv_t:
            matrix_left = torch.linalg.inv(matrix_left).transpose(0, 1)
            matrix_right = torch.linalg.inv(matrix_right).transpose(0, 1)
        return _kronecker_matmul(x, matrix_left, matrix_right)
    

class _SVDOldFlatDecomposeTransform(nn.Module):
    def __init__(self, left_size: int, right_size: int, add_diag: bool):
        super().__init__()
        self.add_diag = add_diag
        self.linear_u_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_u_left.weight.data.copy_(_random_orthogonal(left_size).to(self.linear_u_left.weight))
        # self.linear_u_left = nn.utils.parametrizations.orthogonal(self.linear_u_left, orthogonal_map="cayley", use_trivialization=False)
        self.linear_v_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_v_left.weight.data.copy_(_random_orthogonal(left_size).to(self.linear_v_left.weight))
        # self.linear_v_left = nn.utils.parametrizations.orthogonal(self.linear_v_left, orthogonal_map="cayley", use_trivialization=False)
        self.linear_diag_left = torch.nn.Parameter(torch.ones(left_size, dtype=torch.float32), requires_grad=True)

        self.linear_u_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_u_right.weight.data.copy_(_random_orthogonal(right_size).to(self.linear_u_right.weight))
        # self.linear_u_right = nn.utils.parametrizations.orthogonal(self.linear_u_right, orthogonal_map="cayley", use_trivialization=False)
        self.linear_v_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_v_right.weight.data.copy_(_random_orthogonal(right_size).to(self.linear_v_right.weight))
        # self.linear_v_right = nn.utils.parametrizations.orthogonal(self.linear_v_right, orthogonal_map="cayley", use_trivialization=False)
        self.linear_diag_right = torch.nn.Parameter(torch.ones(right_size, dtype=torch.float32), requires_grad=True)

        if add_diag:
            self.diag_scale = nn.Parameter(torch.ones(left_size * right_size, dtype=torch.float32), requires_grad=True)
        # self._eval_mode = False

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        if self.add_diag:
            diag_scale = self.diag_scale.to(device=x.device, dtype=x.dtype)
            if inv_t:
                x = x / diag_scale
            else:
                x = x * diag_scale

        # if not self._eval_mode:
        matrix_u_left, matrix_u_right = self.linear_u_left.weight, self.linear_u_right.weight
        matrix_v_left, matrix_v_right = self.linear_v_left.weight, self.linear_v_right.weight
        linear_diag_left, linear_diag_right = self.linear_diag_left, self.linear_diag_right
        if inv_t:
            linear_diag_left, linear_diag_right = 1 / linear_diag_left, 1 / linear_diag_right
        # else:
        #     matrix_left, matrix_right = self.matrix_left, self.matrix_right
        #     if inv_t:
        #         matrix_left, matrix_right = self.matrix_left_inv, self.matrix_right_inv
        #         return _kronecker_matmul(x, matrix_left, matrix_right)
        matrix_left = matrix_u_left @ torch.diag(linear_diag_left) @ matrix_v_left.t()
        matrix_right = matrix_u_right @ torch.diag(linear_diag_right) @ matrix_v_right.t()
        return _kronecker_matmul(x, matrix_left, matrix_right)
    

@ALGO_REGISTRY.register(
    name="flatquant",
    description="Learnable affine structure transform inspired by FlatQuant",
    targets=("structure",),
)
class FlatQuant(nn.Module):
    """
    Minimal FlatQuant core for the current framework:
    learn a structure-level affine transform on activations and apply its
    inverse-transpose to linear weights.

    This keeps the main idea from FlatQuant's fast learnable affine transforms,
    but does not port their full wrapper/model-tool/training stack.
    """

    def __init__(self, args, ctx):
        super().__init__()
        self.args = args
        self.dim_size = ctx.dim_size
        self.matrix_size = ctx.matrix_size or getattr(args, "flat_matrix_size", 128)
        self.add_diag = bool(getattr(args, "flat_add_diag", True))

        if self.dim_size is None:
            raise ValueError("FlatQuant requires AlgoBuildContext.dim_size.")

        use_decompose = (
            self.matrix_size is not None
            and self.matrix_size > 1
            and self.dim_size > self.matrix_size
            and self.dim_size % self.matrix_size == 0
        )

        if use_decompose:
            left_size = self.dim_size // self.matrix_size
            right_size = self.matrix_size
            # self.transform = _FlatDecomposeTransform(left_size, right_size, self.add_diag)  # old code
            # self.transform = _InvFlatDecomposeTransform(left_size, right_size, self.add_diag) # inv clean code
            self.transform = _SVDFlatDecomposeTransform(left_size, right_size, self.add_diag) # svd clean code
        else:
            self.transform = _InvFlatSingleTransform(self.dim_size)

    def forward(self, x: torch.Tensor, inv_t: bool = False, name: str = None) -> torch.Tensor:
        return self.transform(x, inv_t=inv_t)

    def transform_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self.transform(weight, inv_t=True)

    def trainable_params(self):
        return list(self.parameters())

    def export_ptq_params(self):
        return {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
        }

    def load_ptq_params(self, params):
        # exported_eval_mode = any(
        #     name.startswith("transform.matrix")
        #     for name in params
        # )
        # if exported_eval_mode and not getattr(self.transform, "_eval_mode", False):
        #     to_eval_mode = getattr(self.transform, "to_eval_mode", None)
        #     if not callable(to_eval_mode):
        #         raise ValueError(
        #             "The saved FlatQuant parameters are in eval mode, "
        #             "but the current transform cannot enter eval mode."
        #         )
        #     to_eval_mode()

        named_params = dict(self.named_parameters())
        for name, value in params.items():
            if name not in named_params:
                continue
                # raise KeyError(
                #     f"Parameter '{name}' is not found in FlatQuant."
                # ) 
            param = named_params[name]
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))