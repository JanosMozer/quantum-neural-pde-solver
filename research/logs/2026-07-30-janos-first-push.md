## 2026-07-30 — Janos's first push (branch `jani`, commit cd2a76b)

726 lines, one commit: `training/{fourier,qnn_generator,pinn_target,physics_loss}.py`, `scripts/{train,export_weights,inference}.py`, `config.yaml`, 4 checkpointed runs.

### What's actually there
A complete, working end-to-end scaffold, not a stub. Fourier feature map (3 to 6 dims), a PennyLane quantum weight generator, a classical target PINN that consumes externally-supplied weights, a physics loss with correct-looking second-order Burgers residuals via `torch.autograd.grad`, a two-stage Adam-then-LBFGS trainer, and a script that exports static weights and runs pure-classical inference with zero PennyLane dependency at deployment. Every file has a runnable `assert`-based self-check. This is good engineering, better than a first pass usually is.

### Three concrete findings

**1. Readout scheme resolved.** Uses `qml.expval(qml.PauliZ(i))` per qubit, 6 numbers total. Confirms the cheap variant, not Liu et al.'s literal full-$2^N$-distribution readout. Open question from the implementation plan is now closed by Janos's actual choice.

**2. Ansatz connectivity checked against PennyLane's own docs, not assumed.** `StronglyEntanglingLayers` uses CNOT with connectivity `(i, (i+r) mod M)`, range `r` varying per layer, confirmed non-nearest-neighbor for `r>1`. Good, this is what the classification step against Cerezo et al.'s criteria needs, not a 1D chain.

**3. Parameter accounting is broken, and this is the headline finding.** `TOTAL_WEIGHTS` (the target PINN's weight count) is 418 (112+272+34). The quantum circuit itself has 36 trainable parameters (`StronglyEntanglingLayers.shape(2, 6)`). But the classical linear projection layer mapping the 6 qubit readouts to 418 weights, `nn.Linear(6, 418)`, has `6*418+418 = 2926` parameters on its own. Total generator parameter count: `36 + 2926 = 2962`, against a target network of 418 weights. The generator is roughly seven times larger than what it generates, and the classical piece of it outnumbers the quantum piece roughly 80 to 1.

This is not a matched-parameter-count compression scheme in its current form. It is not compression at all. Liu et al.'s actual result compresses a bigger classical network into a *smaller* generator (6690 to 728, a 10.8% ratio); this does the reverse. The planned ablation (quantum generator vs. classical generator, matched parameter count) needs this fixed before it means anything, because right now the dominant cost of the "quantum" generator is an oversized classical layer, which would make any classical baseline trivially competitive for reasons that have nothing to do with quantum simulability.

### Secondary finding: training does not look converged
`run_0004`'s `solution_plot.png` shows the network's own `t=0` prediction visibly does not match the exact initial condition it was trained against, and the time-slice curves look noisy rather than physically smooth. 500 combined Adam+LBFGS steps have not gotten this to fit even the boundary condition, let alone the PDE.

### Device note, not a problem
Uses `default.qubit` (CPU, PennyLane's built-in simulator) with `diff_method="backprop"`, not `lightning.gpu`. At 6 qubits this is the right call, GPU acceleration buys nothing at this size. Relevant only if qubit count scales up to fix finding 3.

### What this changes
Janos's task is not "train the thing", it is "make the generator actually a generator, and get it to converge" — fixing finding 3 (likely by scaling qubit count so `2^N` or the quantum-plus-projection parameter count is genuinely smaller than the target network, or restructuring the projection) and finding 4 (debugging convergence) are both real, open, non-trivial technical problems, not implementation grunt work.
