## 2026-07-31 — Phase 1b: circuit classification + classical baseline utilities built and gated

Environment: `qnn/ablation/`, `uv`-managed Python 3.12.12 venv (system Python was 3.14.4, too new for reliable pennylane/torch/quimb wheels; sidestepped rather than fought). No GPU on this laptop, none needed, everything here is a 6-qubit state (64 numbers), trivial for any CPU.

### The real result (Gate 2)

Classified Janos's actual circuit (imported directly from his repo, not reimplemented: 6 qubits, 2 `StronglyEntanglingLayers`, range-based CNOT). Two things measured at every contiguous cut: the exact Schmidt rank (every technically-nonzero singular value) and the *effective* rank (how many singular values are needed to capture 99% of the state's norm). The gap between these two numbers turned out to be the whole story.

Raw Schmidt rank says "fully entangled" at the middle cut for both random-init and trained parameters (8 out of a possible 8). That's misleading on its own, generic states are almost never exactly low-rank, a few tiny but technically-nonzero singular values are enough to claim "full rank" without meaning anything physically. The effective rank tells the real story:

- **Random init:** effective rank 2-3 out of 8 at every cut. Entropy is only 3-9% of the theoretical maximum.
- **Janos's actual trained checkpoint (`run_0004`):** effective rank **1 at every single cut**. Entropy under 3% of maximum everywhere. The trained state is, for practical purposes, a product state.

### What this means, plainly

A quantum state with effective rank 1 at every cut needs no entanglement at all to reproduce to 99% fidelity, a classical description factors it exactly. This isn't a marginal case, it's about as classically simulable as a state can be while still being called entangled at all. The classification's prediction, stated before any ablation is run: expect no measurable advantage from this quantum generator over a classical one, for this specific circuit configuration.

### Why, mechanically (the honest caveat)

This isn't surprising once you look at *why*: 2 layers is shallow, and Janos's own init scheme draws angles from `N(0, 0.1)`, i.e. rotations very close to the identity. Small-angle initialization is a known, deliberate way to dodge barren plateaus, and it was already flagged in this project's own literature review (Cerezo et al.) that exactly this kind of BP-avoiding design tends to land in the classically-simulable regime. This result is that pattern showing up concretely, in our own circuit, not a new phenomenon.

**This is specific to today's configuration** (6 qubits, 2 layers, this init scale). If Janos's Phase 1a fix changes any of qubit count, depth, or init scale while replacing the oversized projection layer, this classification needs to be rerun before it's trusted for the final ablation. Logged as a fact about today's circuit, not a permanent verdict.

### Gate 3 finding, worth keeping: quimb + torch autograd

`quimb`'s own `to_dense()` contraction (via its `cotengra` backend) does not work with torch tensors that require grad on the installed versions (`quimb==1.14.0`, torch `2.13.0+cu130`): it falls back to a numpy `tensordot` path partway through and crashes on `Tensor that requires grad`. Workaround, simple enough it's arguably the better design anyway: use `quimb` only to construct a well-formed random MPS (correct bond dimensions), then contract it to a dense vector by hand with plain `torch.tensordot` in the code, not through quimb's contraction engine. Cross-checked the hand-rolled contraction against quimb's own (numpy, no-grad) `to_dense()` output, exact match, before trusting it for anything gradient-based. This workaround is now baked into `baselines/mps.py`.

### What was skipped

`colloc.py` (a tiny collocation-point sampler mirroring Janos's `make_colloc`/`make_bc`), listed in the implementation plan as a "shared utility," turned out not to be needed by any self-test actually written, none of Gate 1-4's checks required running a forward pass through the target PINN. Not built. Add it when something actually needs it, not because a plan mentioned it.

### Status

Gates 0-4 all pass. Classical baseline generators (`baselines/low_rank.py`, `baselines/mps.py`) are built, parametrized by `target_param_count`, and self-tested at two dummy sizes, not yet run against Janos's real corrected parameter count since that depends on his Phase 1a fix landing. Ready to size and run for real the moment it does.
