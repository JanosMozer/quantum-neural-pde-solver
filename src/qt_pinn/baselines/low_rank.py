"""Classical baseline generator 1: a random low-rank projection.

Redesigned 2026-07-31: the original version (a learned nn.Linear(r, out_dim)) could not
go below out_dim+1 trainable parameters even at rank 1, useless once the real target
(Option 1's actual 82 parameters) turned out to be far smaller than out_dim=418. Fixed
by flipping which side is learned: a FIXED random matrix (not trained) maps a small
learned coefficient vector up to out_dim. This gives exactly target_param_count learned
parameters for any target, including ones smaller than out_dim, and is structurally
closer to Option 1 anyway (few learned numbers, fixed transform, large output).
"""

import torch
import torch.nn as nn


class LowRankGenerator(nn.Module):
    """out = A @ coeffs, where A is a fixed random (out_dim, target_param_count) matrix
    and coeffs is the only learned parameter, so trainable params == target_param_count
    exactly, whatever that target is.
    """

    def __init__(self, out_dim: int, target_param_count: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        r = max(1, target_param_count)
        self.rank = r
        A = torch.randn(out_dim, r) / r ** 0.5  # scaled so output magnitude doesn't grow with r
        self.register_buffer("A", A)
        self.coeffs = nn.Parameter(torch.randn(r) * 0.1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self) -> torch.Tensor:
        return self.A @ self.coeffs


if __name__ == "__main__":
    for target, out_dim in [(82, 418), (418, 418), (1000, 418)]:
        gen = LowRankGenerator(out_dim=out_dim, target_param_count=target)
        out = gen()
        assert out.shape == (out_dim,), f"expected ({out_dim},), got {out.shape}"
        actual = gen.param_count()
        assert actual == target, f"expected exact match, got {actual} for target {target}"
        loss = out.sum()
        loss.backward()
        assert gen.coeffs.grad is not None and torch.any(gen.coeffs.grad != 0), "bad gradient"
        print(f"PASS target={target} out_dim={out_dim}: actual_params={actual} (exact), "
              f"shape ok, gradients ok")
