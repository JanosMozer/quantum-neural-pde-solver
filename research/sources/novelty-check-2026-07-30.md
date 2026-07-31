## Adversarial novelty kill-search — 2026-07-30

**Goal:** actively try to find a published paper or public repo that already does: PQC trained → classical layer maps measurements to classical NN weights → quantum circuit discarded → deployed on a PDE/PINN task → with an ablation/comparison against a classical weight-generator baseline (random projection, low-rank, hypernetwork, tensor-network/MPS) at matched parameter count. Two independent agents, told explicitly to try to kill the claim, not confirm it.

### Agent 1 — phrase/keyword sweep (~15 angles, full text checked for every candidate)
**NOT FOUND.** Three close relatives found and must be cited/distinguished in any writeup:
- **arXiv:2505.09395** (typhoon trajectory forecasting, May 2026) — genuine QT mechanism (generate weights, discard quantum, deploy classical) AND does ablate vs. classical compression baselines (pruning, weight-sharing), beating them. Quote: *"this approach leverages QNNs to generate the weights of a target classical neural network model during the training process... the trained model remains fully classical."* Missing: wrong domain (geoscience time-series, not PDE), wrong baseline family (compression methods, not weight-generators).
- **arXiv:2606.04679** (June 2026) — benchmarks Burgers'/Allen-Cahn/KdV directly. Missing: quantum circuit stays active at inference (never discarded), ablation is a hyperparameter sweep not a classical-baseline comparison.
- **arXiv:2606.18713** (Photonic Quantum Neural Fields for PINN, June 2026) — covers 7 PDE families including Burgers', runs frozen/shuffled-circuit ablations. Missing: circuit stays live at inference, ablation isn't matched-parameter classical-generator comparison.

### Agent 2 — author-network/venue sweep (4 angles)
**NOT FOUND**, and this is the stronger check:
1. **The QT authors themselves** (Liu, Kuo, Hsieh) have extended Quantum-Train to: LSTM/flood prediction, RL (QTRL), federated learning, tensor-network mapping, deepfake-audio CNN, distributed MARL, typhoon forecasting, differentiable architecture search — eight-plus domains across 2024-2026 — and never to PDE/PINN. If the people who invented the mechanism haven't tried the obvious PDE extension in 2+ years of actively extending it elsewhere, that's real signal.
2. **Venues checked** (QCE 2025 program, NeurIPS/ICLR 2025 workshops, Quantum Machine Intelligence journal) — nothing.
3. **QPINN groups' own papers, full text, related/future-work sections** — TE-QPINN, QCPINN, CV-qumode multi-variable PDE PINN, Wavelet-PIQNN, Quantum-Circuit-Enhanced PINN, Quantum-Enhanced-Convergence PINN — none mention Quantum-Train as related or future work; all keep the quantum circuit permanently in the inference loop, architecturally the opposite design choice.
4. **GitHub** — official QT repo (github.com/Hon-Hai-Quantum-Computing/QuantumTrain) lists MNIST/CIFAR/flood/RL examples only, zero PDE mentions, zero relevant open issues.

### Verdict
Novelty survives a genuinely hard, two-angle, full-text adversarial search (~19 distinct attack angles combined). Not just "we didn't happen to find it" — the mechanism's own inventors have spent 2+ years extending it to nearly everything except this, and three 2026 papers circling adjacent territory all miss it on either the mechanism (discard-after-training) or the ablation (matched-parameter classical baseline) axis. Residual risk: absence-of-evidence is never proof; the three close-relative papers must be cited and explicitly distinguished from in any writeup, not treated as irrelevant.
