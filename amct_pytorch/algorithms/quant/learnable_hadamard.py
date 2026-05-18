import torch
import torch.nn as nn

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY


def get_random_orthg(size: int, *, device=None, dtype=None) -> torch.Tensor:
    matrix = torch.randn(size, size, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix)
    diag = torch.sign(torch.diag(r))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    q = q @ torch.diag(diag)
    if dtype is not None:
        q = q.to(dtype)
    return q


class InvSingleTransMatrix(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False)
        self.linear.weight.data.copy_(get_random_orthg(size).to(self.linear.weight))
        self._eval_mode = False
        self.register_buffer("_matrix", None, persistent=False)
        self.register_buffer("_matrix_inv_t", None, persistent=False)

    def forward(self, inp: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        init_shape = inp.shape
        matrix = self.get_matrix(inv_t=inv_t).to(device=inp.device, dtype=inp.dtype)
        inp = inp.reshape(-1, matrix.shape[0])
        return inp.matmul(matrix).reshape(init_shape)

    def get_matrix(self, inv_t: bool = False) -> torch.Tensor:
        if self._eval_mode and self._matrix is not None and self._matrix_inv_t is not None:
            return self._matrix_inv_t if inv_t else self._matrix

        matrix = self.linear.weight
        if inv_t:
            return torch.linalg.inv(matrix).transpose(0, 1)
        return matrix

    def to_eval_mode(self):
        if self._eval_mode:
            return
        matrix = self.linear.weight.detach()
        matrix_inv_t = torch.linalg.inv(matrix).transpose(0, 1)
        self._matrix = matrix
        self._matrix_inv_t = matrix_inv_t
        self._eval_mode = True

    def __repr__(self):
        matrix = self._matrix if self._eval_mode and self._matrix is not None else self.linear.weight
        return f"InvSingleTransMatrix(eval_mode={self._eval_mode}, matrix.shape={tuple(matrix.shape)})"


@ALGO_REGISTRY.register(
    name="learnable_had",
    description="Learnable hadamard-like structure transform",
    targets=("structure",),
)
class LearnableHadamard(nn.Module):
    def __init__(self, args, ctx):
        super().__init__()
        self.args = args
        self.size = ctx.matrix_size
        self.transform = InvSingleTransMatrix(self.size)

    def forward(self, x: torch.Tensor, inv_t: bool = False, name: str = None) -> torch.Tensor:
        return self.transform(x, inv_t=inv_t)

    def trainable_params(self):
        return list(self.parameters())

    def export_ptq_params(self):
        return {
            "transform_weight": self.transform.linear.weight.detach().cpu(),
        }

    def load_ptq_params(self, params):
        self.transform.linear.weight.data.copy_(
            params["transform_weight"].to(
                device=self.transform.linear.weight.device,
                dtype=self.transform.linear.weight.dtype,
            )
        )
