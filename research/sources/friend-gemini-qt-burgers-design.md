## Friend's Gemini design chat — Quantum-Train PINN for 2D Burgers' equation

**URL:** https://gemini.google.com/share/b2b9b53575d9?skid=190575b9-b992-4b06-8156-93ad3b87ce5e (via share.gemini.google/bPi9OFILS1fK)
**Type:** Not a paper — an informal Gemini design conversation. Treat every claim below as unverified until cross-checked against primary QML literature. Fetched via headless Chrome (`--dump-dom`) since the share link requires JS render.

### What it proposes (5-component architecture)
1. **Classical Fourier feature front-end** — maps `(x,y,t) ∈ R³` to `γ(x) = [cos(2πBx), sin(2πBx)]` to fight spectral bias (standard SIREN/Fourier-features trick, not novel).
2. **PQC ansatz** — strongly-entangling hardware-efficient circuit, `N` qubits, `Rz-Ry-Rz` rotations per layer + circular CNOT entangling layer.
3. **Weight-mapping projection** — measures `⟨Z_i⟩` for each qubit (N real numbers), then a classical linear layer `W_c = Aq + b` expands these into the full classical PINN weight matrices. This is exactly the **"Quantum-Train" (QT)** paradigm attributed (by Gemini, unverified) to recent QML literature — needs primary-source check.
4. **Physics-informed loss** — standard PINN: PDE residual (autograd) + IC + BC terms.
5. **Two-stage optimization** — Adam (1-1.5k steps) then L-BFGS to push residual `<1e-5`.

Post-training: run the circuit once, extract `W_c` as static float tensors, **discard the quantum circuit entirely**, deploy a pure classical `nn.Module`. So the quantum part is only a training-time parameter generator, not part of inference.

### Compute claims (RTX 5090, 32GB, pennylane-lightning[gpu] + cuQuantum)
- VRAM ≈ `B × 3 × 2^N × 8 bytes` (B=batch, 3 = adjoint-diff state copies, 2^N = state vector size, FP32).
- Their recommendation for 2D Burgers': **10-12 qubits** is the sweet spot (>4096 basis states, <0.5s/step); 24 qubits is VRAM-fine (~3.2GB) but **compute-bound**, not memory-bound.
- Quote: *"What takes 5 minutes at 10 qubits might take three weeks of continuous 100% GPU utilization at 24 qubits. You will not hit an out-of-memory (OOM) error; you will hit a wall of agonizingly slow epoch times."*

### Bottlenecks Gemini itself raised (self-critique, second turn) — these are the fracture-relevant leads
1. **Exponential time wall** — state-vector sim cost, independent of the VRAM fit.
2. **Barren plateaus** — quote: *"the variance of the gradient scales as O(1/2^n)... at 24 qubits your gradients will be so infinitesimally small... the Adam optimizer will just bounce around in numerical noise."*
3. **"Translation illusion" / information bottleneck** — quote: *"A massive debate in QML research right now is whether Quantum-Train architectures actually benefit from quantum entanglement, or if the classical mapping MLP is just acting as a clever classical Low-Rank matrix factorization tool."* **This is the sharpest candidate problem here** — precise, has a real attack (ablate entanglement / compare against a classical low-rank baseline at matched parameter count), and a structural reason (Type C shape) if true.
4. **Frequency saturation** — PDE high-frequency content is capped by how many times `(x,t)` is re-uploaded (data re-uploading literature), ties back into bottleneck #1 and #2 at depth.
5. **Hyperparameter fragility** — periodic angle landscape (`θ ∈ [0,2π]`), amplified vs classical PINN fragility.
- Their bottom line: keep to 8-14 qubits, 1D/2D smooth PDEs (heat/wave/Burgers), not 3D turbulence.

### Feynman summary (what I actually understand so far)
A quantum circuit's job here isn't to compute anything at inference time — it's a fancy random-but-structured number generator used *during training only*, exploiting the fact that unitary constraints (`U†U=I`) force whatever comes out to be smooth and low-rank. The open question Gemini raised is exactly the one I'd ask as an ML person: a random projection + linear read-out is a well-known classical trick (random features, LoRA, hypernetworks). What, if anything, does forcing that projection to be unitary (i.e. quantum) buy you over just sampling a random *classical* orthogonal/low-rank matrix? If the answer is "nothing measurable", the entire architecture reduces to an elaborate, GPU-expensive way of doing classical low-rank hypernetwork weight generation — which would itself be a legitimate, well-scoped finding (Type C/E), just not the one being marketed. **This needs primary-source verification before it's trusted as an open problem** — next step: find the actual Quantum-Train papers and read what they measured, not what Gemini summarized.

### Status — UPDATED 2026-07-30, partially verified
Primary source found and read in full: Liu et al. 2024, arXiv:2402.16465, "Training Classical Neural Networks by Quantum Machine Learning" — see `ours/sources/liu2024_quantum_train.md` for full detail. Two corrections to this design doc:

1. **Measurement scheme mismatch.** The real Quantum-Train method reads out the full `2^N`-dimensional basis-state probability distribution, not per-qubit `⟨Z_i⟩` (only N numbers). The friend's design uses the cheaper `⟨Z_i⟩` readout — a legitimate variant, but not literally "Quantum-Train" as published. Worth flagging to the peer.
2. **"Massive debate" claim is an overstatement.** Gemini's framing of an active, named debate about entanglement-vs-classical-low-rank was NOT found in any primary source. What IS real and verified: the QT authors' own paper contains a complexity-theoretic necessity argument (SampBQP ≠ SampBPP) for why a QNN can't be replaced by a classical generator — but they immediately caveat it: matrix-product-state (tensor network) methods CAN efficiently replicate this "for 1D or geometrically local Hamiltonian" circuits. That caveat, not a field-wide debate, is the real and precise crack — see the hypothesis note in `ours/hypotheses/`.

No published prior art was found combining QT-style weight generation with PINNs (closest: TE-QPINN keeps the quantum circuit in the inference loop — architecturally the opposite of QT's train-then-discard design).
