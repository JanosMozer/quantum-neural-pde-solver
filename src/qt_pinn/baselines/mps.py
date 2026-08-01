"""Classical baseline generator 2: a trainable matrix product state (MPS).

Parametrized by target_param_count, not a hardcoded number (same reason as low_rank.py).
Uses quimb only to construct a well-formed random MPS (correct bond structure); the
forward-pass contraction is done by hand in plain torch, per the Gate 3 finding: quimb's
own to_dense() contraction does not support torch tensors that require grad on this
installed version (cotengra falls back to a numpy path and crashes). The manual
contraction was cross-checked exactly against quimb's own reference in
test_quimb_autograd.py before being reused here.
"""

import torch
import torch.nn as nn
import quimb.tensor as qtn


def _mps_to_dense(tensors: list[torch.Tensor]) -> torch.Tensor:
    result = tensors[0].transpose(0, 1)  # (bond, phys) -> (phys, bond)
    n = len(tensors)
    for i, t in enumerate(tensors[1:], start=1):
        result = torch.tensordot(result, t, dims=([-1], [0]))
        if i < n - 1:
            result = result.transpose(-2, -1)
    return result.reshape(-1)


def _param_count_for(num_sites: int, bond_dim: int) -> int:
    mps = qtn.MPS_rand_state(L=num_sites, bond_dim=bond_dim, dtype="float64")
    return sum(t.data.size for t in mps.tensors)


def _pick_num_sites_and_bond_dim(out_dim: int, target_param_count: int) -> tuple[int, int]:
    """Smallest num_sites with 2**num_sites >= out_dim, then whichever bond_dim (from a
    small grid) gives a total parameter count closest to the target. Direct search over
    actually-constructed tensor sizes, not a closed-form approximation.
    """
    num_sites = 1
    while 2 ** num_sites < out_dim:
        num_sites += 1
    num_sites = max(num_sites, 2)  # need at least 2 sites for the chain contraction to make sense

    best_bond_dim, best_diff = 1, None
    for bond_dim in range(1, 33):
        count = _param_count_for(num_sites, bond_dim)
        diff = abs(count - target_param_count)
        if best_diff is None or diff < best_diff:
            best_bond_dim, best_diff = bond_dim, diff
    return num_sites, best_bond_dim


class MPSGenerator(nn.Module):
    def __init__(self, out_dim: int, target_param_count: int, seed: int = 0) -> None:
        super().__init__()
        self.out_dim = out_dim
        num_sites, bond_dim = _pick_num_sites_and_bond_dim(out_dim, target_param_count)
        self.num_sites = num_sites
        self.bond_dim = bond_dim

        # quimb's MPS_rand_state already normalizes the overall contracted state to unit
        # norm; scaling each of the num_sites tensors individually (previously *0.1) compounds
        # multiplicatively across the contraction (0.1**num_sites), collapsing both the output
        # and its gradient to numerical zero for any num_sites much above a handful. Use
        # quimb's own init unscaled.
        # seed=... is required: MPS_rand_state draws from its own RNG, not covered by
        # torch.manual_seed, confirmed by two unseeded calls returning different tensors.
        seed_mps = qtn.MPS_rand_state(L=num_sites, bond_dim=bond_dim, dtype="float64", seed=seed)
        self.tensors = nn.ParameterList([
            nn.Parameter(torch.tensor(t.data.copy(), dtype=torch.float64))
            for t in seed_mps.tensors
        ])

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self) -> torch.Tensor:
        dense = _mps_to_dense(list(self.tensors))
        return dense[: self.out_dim]


if __name__ == "__main__":
    for target, out_dim in [(82, 418), (418, 418), (1000, 418)]:
        gen = MPSGenerator(out_dim=out_dim, target_param_count=target)
        out = gen()
        assert out.shape == (out_dim,), f"expected ({out_dim},), got {out.shape}"
        actual = gen.param_count()
        rel_err = abs(actual - target) / target
        assert rel_err < 0.35, f"param count {actual} too far from target {target} (rel err {rel_err:.2f})"
        loss = out.sum()
        loss.backward()
        grads = [p.grad for p in gen.parameters()]
        assert all(g is not None for g in grads), "missing gradient"
        # not just nonzero: catches the real bug found 2026-07-31, a compounding *0.1
        # per-tensor init scale that left gradients technically nonzero but ~1e-8, dead
        # in practice. 1e-4 is well above float32 training noise, well below a healthy signal.
        assert all(g.norm().item() > 1e-4 for g in grads), "gradient too small to be useful, not just nonzero"
        assert out.std().item() > 1e-4, "output magnitude too small to be useful"
        # reproducibility: found 2026-08-01 that MPS_rand_state ignored torch.manual_seed
        # entirely (own RNG source); same seed must now give bit-identical init.
        gen2 = MPSGenerator(out_dim=out_dim, target_param_count=target, seed=0)
        assert torch.allclose(gen().detach(), gen2().detach()), "same seed must reproduce exactly"
        print(f"PASS target={target} out_dim={out_dim}: num_sites={gen.num_sites} "
              f"bond_dim={gen.bond_dim} actual_params={actual} (rel err {rel_err:.2%}), "
              f"shape ok, gradients ok")
