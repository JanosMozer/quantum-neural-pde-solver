"""Gate 3: the one real unknown in this plan. Does a quimb-constructed MPS support PyTorch
gradients well enough to train an MPS generator later.

Finding: quimb's own `to_dense()` contraction (via cotengra) does not work with torch
tensors that require grad on this installed version (cotengra falls back to a numpy
tensordot path and crashes). Workaround, and it's simple enough that it's arguably the
better choice anyway: use quimb only to *construct* a well-formed random MPS (correct
bond structure), then contract it to a dense vector by hand with plain torch.tensordot,
bypassing quimb's contraction engine entirely. Cross-checked against quimb's own (numpy,
no-grad) to_dense() for correctness before trusting the gradient path.
"""

import numpy as np
import torch
import quimb.tensor as qtn


def mps_to_dense_manual(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Contract an open-boundary MPS (site tensors in quimb's convention: interior sites
    shaped (bond_left, bond_right, phys), boundary sites shaped (bond, phys)) into a
    dense vector, by hand, so gradients flow through plain torch ops only.
    """
    result = tensors[0].transpose(0, 1)  # (bond, phys) -> (phys, bond)
    n = len(tensors)
    for i, t in enumerate(tensors[1:], start=1):
        result = torch.tensordot(result, t, dims=([-1], [0]))
        if i < n - 1:
            # t was an interior tensor: contracting added (bond_right, phys) at the end
            result = result.transpose(-2, -1)
    return result.reshape(-1)


def main() -> None:
    L, bond_dim = 4, 2

    # ground truth: quimb's own contraction, plain numpy, no grad involved
    ref_mps = qtn.MPS_rand_state(L=L, bond_dim=bond_dim, dtype="float64", seed=0)
    ref_dense = ref_mps.to_dense().reshape(-1)

    # same tensors, converted to torch parameters, contracted by hand
    torch_tensors = [
        torch.tensor(np.asarray(t.data), requires_grad=True) for t in ref_mps.tensors
    ]
    dense = mps_to_dense_manual(torch_tensors)

    assert np.allclose(dense.detach().numpy(), ref_dense, atol=1e-10), (
        "manual contraction disagrees with quimb's own to_dense() ground truth"
    )
    print("PASS: manual torch contraction matches quimb's own to_dense() exactly")

    loss = (dense ** 2).sum()
    loss.backward()

    grads = [t.grad for t in torch_tensors]
    assert all(g is not None for g in grads), "some tensor got no gradient at all"
    assert all(torch.any(g != 0) for g in grads), "some tensor's gradient is all zero"
    print(f"PASS: gradients flow through all {len(grads)} MPS tensors, nonzero")


if __name__ == "__main__":
    main()
