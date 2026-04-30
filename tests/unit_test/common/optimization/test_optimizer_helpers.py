import pytest
import torch

from amct_pytorch.common.optimization.optimizer import (
    Cayley_loop,
    matrix_norm_one,
    norm,
    qr_retraction,
    unit,
)


def test_norm_returns_l2_keepdim():
    v = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    out = norm(v, dim=1)
    assert out.shape == (2, 1)
    assert torch.allclose(out, torch.tensor([[5.0], [5.0]]))


def test_norm_asserts_2d_input():
    with pytest.raises(AssertionError):
        norm(torch.zeros(3))


def test_unit_normalizes_to_unit_vectors_and_returns_norm():
    v = torch.tensor([[3.0, 4.0]])
    normalized, n = unit(v)
    assert torch.allclose(n, torch.tensor([[5.0]]))
    assert torch.allclose(normalized.norm(dim=1), torch.tensor([1.0]), atol=1e-5)


def test_unit_handles_zero_vector_via_eps():
    v = torch.zeros(1, 3)
    normalized, n = unit(v, eps=1e-8)
    assert n.item() == pytest.approx(0.0)
    # eps avoids division by zero -> result is near-zero (not NaN).
    assert torch.isfinite(normalized).all()


def test_matrix_norm_one_is_max_column_abs_sum():
    # column sums of |W|: col0=6, col1=15
    w = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert matrix_norm_one(w).item() == pytest.approx(15.0)


def test_cayley_loop_at_zero_step_returns_x_transposed():
    x = torch.eye(3, 2)             # 3x2 (n, p)
    w = torch.zeros(3, 3)
    tan = torch.zeros(3, 2)
    out = Cayley_loop(x, w, tan, t=0.0)
    # When t=0 -> Y = X; the function returns Y.t() with shape (p, n).
    assert out.shape == (2, 3)
    assert torch.allclose(out, x.t())


def test_qr_retraction_yields_orthonormal_rows():
    # qr_retraction expects p-by-n with p<=n. After call it transposes back so
    # the returned tensor has the same orientation.
    tan = torch.randn(3, 5)
    q = qr_retraction(tan.clone())
    # Row vectors should be orthonormal.
    assert q.shape == (3, 5)
    assert torch.allclose(q @ q.t(), torch.eye(3), atol=1e-5)
