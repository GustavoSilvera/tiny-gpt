import torch
from typing import Optional
from utils import *


# mathematical trick in self attention to optimize the incremental averaging
# goal: want xBOW[b, t] = mean_{i <= t} x[b, i] # incremental averaging


def compute_xBOW_naive(x: torch.Tensor) -> torch.Tensor:
    # the general idea of incremental averaging, but this is very slow. Don't use!
    B, T, C = x.shape
    xBOW: torch.Tensor = torch.zeros_like(x, device=device)
    for b in range(B):
        for t in range(T):
            xBOW[b, t] = torch.mean(x[b, : t + 1], axis=0, dtype=torch.float)
    assert xBOW.shape == (B, T, C)
    return xBOW


def compute_xBOW_matmul(x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    # can accomplish this via matrix multiplication with a lower-triangular matrix where all the rows
    # sum to 1 and nonzero row elements are equal
    # [1.00, 0.00, 0.00]
    # [0.50, 0.50, 0.00]
    # [0.33, 0.33, 0.33]
    inc_avg_mat: torch.Tensor = torch.tril(
        torch.ones(T, T, device=device)
    )  # weighted summing
    inc_avg_mat /= inc_avg_mat.sum(axis=1, keepdim=True)
    # (B x T x T) @ (B x T x C) -> (B x ((T x T) @ (T x c))) -> (B x T x C) :)
    xBOW: torch.Tensor = inc_avg_mat @ x
    assert xBOW.shape == (B, T, C)
    return xBOW


def compute_xBOW_softmax(
    x: torch.Tensor, z: Optional[torch.Tensor] = None
) -> torch.Tensor:
    B, T, C = x.shape
    # enables more flexibility than compute_xBOW_matmul by allowins the initial weights to be
    # more interesting than just 1's and 0's
    tri_lower: torch.Tensor = torch.tril(
        torch.ones(T, T, device=device)
    )  # weighted summing
    # clamping/preventing tokens from talking to the future
    tri_lower[tri_lower == 0] = -float("inf")  # exp(-inf) -> 0 to not contribute
    if z is not None:
        # allow element wise product (more interesting than just 1's)
        tri_lower = z * tri_lower
        tri_lower[tri_lower.isinf()] = -float("inf")  # ensure only -inf not +inf
        assert tri_lower.shape == z.shape  # broadcast to z's shape
    inc_avg_mat = torch.nn.functional.softmax(tri_lower, dim=-1)
    xBOW: torch.Tensor = inc_avg_mat @ x
    assert xBOW.shape == (B, T, C)
    return xBOW


x = torch.randn(10, 4, 3, device=device)  # for example
xBOW1 = compute_xBOW_naive(x)
xBOW2 = compute_xBOW_matmul(x)
xBOW3 = compute_xBOW_softmax(x)
print(
    "Bag of words correctness: ",
    torch.allclose(xBOW1, xBOW2) and torch.allclose(xBOW1, xBOW3),
)
