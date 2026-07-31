"""Classical baseline generator 1: a random low-rank projection.

Parametrized by target_param_count, not a hardcoded number, since Janos's Phase 1a fix
will change what that number actually is. Given a target output vector length and a
parameter budget, solves for a rank r such that a (out_dim, r) x (r,) factorization uses
approximately that many trainable parameters.
"""

import torch
import torch.nn as nn


class LowRankGenerator(nn.Module):
    """Generates a flat weight vector of length out_dim from a small trainable seed,
    via a (out_dim, r) learned basis. Total trainable parameters ~= out_dim * r + r.
    """

    def __init__(self, out_dim: int, target_param_count: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        # no bias: with one, out_dim alone already costs out_dim params, which can
        # overshoot small targets before rank even enters the picture.
        r = max(1, round(target_param_count / (out_dim + 1)))
        self.rank = r
        self.basis = nn.Linear(r, out_dim, bias=False)
        self.seed = nn.Parameter(torch.randn(r) * 0.1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self) -> torch.Tensor:
        return self.basis(self.seed)


if __name__ == "__main__":
    for target, out_dim in [(418, 418), (1000, 418)]:
        gen = LowRankGenerator(out_dim=out_dim, target_param_count=target)
        out = gen()
        assert out.shape == (out_dim,), f"expected ({out_dim},), got {out.shape}"
        actual = gen.param_count()
        rel_err = abs(actual - target) / target
        # integer rank means we can only approximate the target, not hit it exactly;
        # 35% is generous on purpose for this generic self-test with arbitrary dummy
        # targets, the real ablation tunes rank against the actual corrected count.
        assert rel_err < 0.35, f"param count {actual} too far from target {target} (rel err {rel_err:.2f})"
        loss = out.sum()
        loss.backward()
        grads = [p.grad for p in gen.parameters()]
        assert all(g is not None for g in grads), "missing gradient"
        assert all(torch.any(g != 0) for g in grads), "zero gradient"
        print(f"PASS target={target} out_dim={out_dim}: rank={gen.rank} actual_params={actual} "
              f"(rel err {rel_err:.2%}), shape ok, gradients ok")
