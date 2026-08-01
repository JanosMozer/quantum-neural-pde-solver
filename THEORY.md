# QT-QPINN — Complete Theory Reference

A deep-dive into every mathematical concept, technique, and design decision in this project, plus what you would use to go further.

---

## Table of Contents

1. [The Physical Problem — 2D Viscous Burgers' Equation](#1-the-physical-problem)
2. [Physics-Informed Neural Networks (PINNs)](#2-physics-informed-neural-networks)
3. [Fourier Feature Maps — Fixing the Spectral Bias](#3-fourier-feature-maps)
4. [Quantum Computing Foundations](#4-quantum-computing-foundations)
5. [Variational Quantum Circuits](#5-variational-quantum-circuits)
6. [The Quantum Hypernetwork Design](#6-the-quantum-hypernetwork-design)
7. [Backpropagation Through Quantum Circuits](#7-backpropagation-through-quantum-circuits)
8. [Two-Stage Optimisation: Adam + L-BFGS](#8-two-stage-optimisation)
9. [Loss Function Design](#9-loss-function-design)
10. [Metrics and Evaluation](#10-metrics-and-evaluation)
11. [Known Failure Modes](#11-known-failure-modes)
12. [What to Try Next — Techniques That Could Be Applied](#12-what-to-try-next)

---

## 1. The Physical Problem

### 2D Viscous Burgers' Equation

Burgers' equation is one of the simplest nonlinear PDEs that shares structure with the Navier-Stokes equations — it has nonlinear advection and viscous diffusion, making it a standard benchmark for PDE solvers.

The coupled 2D system:

```
f_u := ∂u/∂t + u·∂u/∂x + v·∂u/∂y − ν·(∂²u/∂x² + ∂²u/∂y²) = 0
f_v := ∂v/∂t + u·∂v/∂x + v·∂v/∂y − ν·(∂²v/∂x² + ∂²v/∂y²) = 0
```

Where:
- `u(x, y, t)` — x-component of velocity
- `v(x, y, t)` — y-component of velocity
- `ν = 0.01/π ≈ 0.00318` — kinematic viscosity (fixed benchmark value)
- Domain: `x, y ∈ [-1, 1]`, `t ∈ [0, 1]`

**Term-by-term physics:**

| Term | Name | Effect |
|---|---|---|
| `∂u/∂t` | temporal derivative | rate of change in time |
| `u·∂u/∂x + v·∂u/∂y` | nonlinear advection | velocity field transporting itself — causes shocks |
| `ν·(∂²u/∂x² + ∂²u/∂y²)` | viscous diffusion (Laplacian) | smooths out sharp gradients, dissipates energy |

**Initial condition used:**
```
u(x, y, 0) = sin(πx)·cos(πy)
v(x, y, 0) = −cos(πx)·sin(πy)
```

This is a smooth, divergence-free vortex field (`∂u/∂x + ∂v/∂y = 0`) that evolves under advection and diffusion. The solution stays smooth for all `t > 0` at this viscosity — no shock formation. That makes it a clean benchmark: the network can be verified analytically at `t=0` and must reproduce physically consistent evolution for `t > 0`.

**Why ν = 0.01/π specifically?** This is the standard from Raissi et al. (2019). Lower ν makes the problem harder (sharper gradients, longer correlation lengths) and eventually causes shocks — a fundamentally different numerical regime.

---

## 2. Physics-Informed Neural Networks

### The Core Idea

A standard neural network learns from labelled data `{(x_i, y_i)}`. A PINN learns from **physics** — the PDE itself. No simulation data is needed. The network is trained to be a function that satisfies the differential equation everywhere in the domain.

**The key insight:** if a neural network `f_θ(x, y, t)` is differentiable (which all smooth NNs are), you can compute its partial derivatives exactly using automatic differentiation, then penalise the PDE residual directly.

### Collocation Point Sampling

The domain is sampled at `N` random interior points:
```
(xᵢ, yᵢ, tᵢ) ~ Uniform([-1,1]² × [0,1])
```

These are called **collocation points**. The PDE residual is enforced at these points. More points → better coverage of the domain → more accurate solution, but slower training.

### The Loss Function

```
L_total = L_pde + λ · L_bc

L_pde = (1/N) Σ [ f_u(xᵢ,yᵢ,tᵢ)² + f_v(xᵢ,yᵢ,tᵢ)² ]

L_bc  = (1/M) Σ [ (û(xⱼ,yⱼ,0) − u_exact(xⱼ,yⱼ))²
                 + (v̂(xⱼ,yⱼ,0) − v_exact(xⱼ,yⱼ))² ]
```

`λ` (lambda_bc in config) weights how strongly to enforce the initial condition relative to the PDE interior.

### Why Automatic Differentiation, Not Finite Differences?

Finite differences approximate `∂u/∂x ≈ (u(x+h) − u(x−h)) / 2h`, introducing truncation error `O(h²)`. Autograd computes the **exact** symbolic derivative of the computation graph — no approximation error. This is critical for PDEs requiring second derivatives, where finite difference error compounds.

The `create_graph=True` flag in `torch.autograd.grad` is essential: it builds a second computation graph over the first derivatives, enabling second-order derivatives to be differentiated through again during backpropagation.

### Historical Context

PINNs were introduced by Raissi, Perdikaris & Karniadakis (2019) in *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"*, Journal of Computational Physics. This paper showed that neural networks can solve PDEs without training data, purely from the physics.

---

## 3. Fourier Feature Maps

### The Spectral Bias Problem

Standard MLPs suffer from **spectral bias** (Rahaman et al., 2019): they learn low-frequency components of a function much faster than high-frequency ones. For PDEs with oscillatory or multi-scale solutions, an untransformed MLP often converges to a poor approximation.

**Empirical observation:** give an MLP the raw coordinates `(x, y, t)` and it struggles to learn anything oscillatory like `sin(πx)`.

### The Fix: Random Fourier Features

Tancik et al. (2020) showed that mapping inputs through a random Fourier feature embedding before the MLP completely solves spectral bias for coordinate-based networks.

The mapping:

```
γ(x) = [ sin(2π·B·x), cos(2π·B·x) ]
```

Where `B ∈ ℝ^(d_input × d_freq)` is sampled from `N(0, σ²)`.

For our case:
- Input `x = (x, y, t) ∈ ℝ³`
- `B ∈ ℝ^(3×3)` sampled from `N(0, σ²)`
- Output `γ(x) ∈ ℝ⁶` — three sin features, three cos features

**Why does this work?** The dot products `B·x` create random linear combinations of the input coordinates. Applying sin/cos lifts them onto the unit circle in a high-dimensional space where frequency content is explicitly encoded. The MLP then only needs to learn which combinations of these pre-computed frequencies produce the solution — a much easier task.

**The σ parameter:** controls the bandwidth of sampled frequencies.
- `σ` too small → only low frequencies represented → misses fine structure
- `σ` too large → frequencies too high → noisy, hard to optimise
- `σ = 1.0` is a good default; tune upward for solutions with sharp gradients

**The seed must be fixed** because `B` is drawn at initialisation and baked into the model's `forward` pass. If you retrain with a different seed, `B` changes and the Fourier basis is different — the frozen static weights from an old run become meaningless.

---

## 4. Quantum Computing Foundations

### Qubits

A classical bit is 0 or 1. A **qubit** is a quantum system that can be in a **superposition** of both:

```
|ψ⟩ = α|0⟩ + β|1⟩      where |α|² + |β|² = 1
```

`α` and `β` are complex amplitudes. The system is in both states simultaneously until measured. Upon measurement, it collapses to `|0⟩` with probability `|α|²` or `|1⟩` with probability `|β|²`.

Geometrically, a qubit lives on the **Bloch sphere** — a unit sphere where the north pole is `|0⟩` and the south pole is `|1⟩`. Any point on the surface is a valid quantum state.

### N-Qubit Systems

With `N` qubits, the state lives in a `2^N`-dimensional complex Hilbert space:

```
|ψ⟩ = Σ αᵢ|i⟩    i ∈ {0,1}^N
```

For `N=6` (this project), the state space is `2⁶ = 64`-dimensional. This exponential scaling is the source of potential quantum advantage — classically storing this state requires 64 complex numbers, but a quantum computer *is* that state.

### Quantum Gates

Gates are **unitary matrices** `U` (`U†U = I`) applied to qubits. They rotate the quantum state:

**Single-qubit gates used in this project:**

- **RY(θ)** — Y-rotation by angle θ:
  ```
  RY(θ) = [[cos(θ/2), −sin(θ/2)],
            [sin(θ/2),  cos(θ/2)]]
  ```
  Used in AngleEmbedding to encode classical data as rotation angles.

**Two-qubit gates:**

- **CNOT** — controlled-NOT: flips target qubit if control is `|1⟩`. Creates entanglement between qubits. StronglyEntanglingLayers uses CNOT-like interactions between all qubits.

### Measurement and Expectation Values

We don't directly read out the quantum state (that would collapse it). Instead we measure **observables** — Hermitian operators representing physical quantities.

The **Pauli-Z operator**:
```
Z = [[1,  0],
     [0, -1]]
```

Its expectation value on qubit `i`:
```
⟨Z_i⟩ = ⟨ψ|Z_i|ψ⟩ ∈ [-1, 1]
```

This is the probability of measuring `|0⟩` minus the probability of measuring `|1⟩`. It's a classical real number that the quantum circuit outputs — the interface between quantum and classical worlds. In this project, 6 qubits give 6 Pauli-Z values, forming a 6-element classical vector.

---

## 5. Variational Quantum Circuits

### What They Are

A **Variational Quantum Circuit (VQC)** is a parameterised quantum circuit `U(θ)` where `θ` is a vector of learnable rotation angles. The circuit transforms an input state and produces measurements that depend on `θ`. Training means optimising `θ` to minimise a classical loss function.

```
Classical input x → Encoding U_enc(x) → Variational U_var(θ) → Measurement → Classical output
```

This is exactly a quantum-classical hybrid: the quantum circuit does the computation, a classical optimiser adjusts `θ`.

### AngleEmbedding

Encodes a classical vector `x ∈ ℝ^N` into a quantum state by using each component as a rotation angle:

```
U_enc(x) = RY(x₁) ⊗ RY(x₂) ⊗ ... ⊗ RY(x_N)
```

Each qubit is independently rotated on the Bloch sphere by the corresponding input value. This is a **product state** — no entanglement yet. Entanglement is created by the variational layers that follow.

**Limitations:** AngleEmbedding is a single data-encoding layer. It can only encode `N_QUBITS` values. More expressive alternatives include **data re-uploading** (see Section 12).

### StronglyEntanglingLayers

Developed by Pennylane (Schuld et al.), this is a hardware-efficient ansatz consisting of:

1. A layer of single-qubit rotations `Rot(φ, θ, ω)` on every qubit (3 parameters per qubit)
2. A layer of CNOT gates connecting every qubit to a "distance-d" neighbour, cycling through distances 1, 2, 3, ... to create all-to-all entanglement

For `N_QUBITS=6`, `N_LAYERS=2`:
- Parameters per layer: `6 × 3 = 18` rotation angles
- Total circuit parameters: `2 × 18 = 36` (the `q_weights` tensor shape is `(2, 6, 3)`)

The entangling pattern ensures every qubit influences every other qubit within `L` layers — this is called a **light cone** in quantum information theory.

**Expressibility:** A circuit is **expressible** if it can approximate any unitary in `U(2^N)`. StronglyEntanglingLayers is designed to be highly expressible — it covers a large fraction of the full unitary group for moderate layer counts.

**Entanglement capability:** The ability to generate entanglement between qubits. The CNOT structure creates multi-qubit correlations that cannot be reproduced by any classical product-state model of the same parameter count. This is the core claim of quantum advantage.

---

## 6. The Quantum Hypernetwork Design

### The Hypernetwork Concept

A **hypernetwork** (Ha, Dai & Le, 2016) is a network that generates the weights of another network. Instead of training the target network directly, you train the hypernetwork — which can be much smaller, or impose structure on the generated weights.

In this project:
- **Hypernetwork:** `QuantumWeightGenerator` (quantum circuit + linear projection)
- **Target network:** `TargetPINN` (classical MLP)

The quantum circuit acts as a structured, low-dimensional hypernetwork. Its `36 + 418×6` parameters generate all `418` MLP weights on every forward pass.

### Why This Design?

**Standard PINN:** train MLP weights directly (418 parameters, all classical).

**QT-QPINN:** train quantum circuit angles + projection layer, which generate MLP weights (36 circuit angles + `418×6` projection = ~2550 parameters).

This seems like more parameters, not fewer — so what's the point?

1. **Quantum expressibility:** The 6 Pauli-Z values are not 6 independent scalars. They come from a `64`-dimensional entangled quantum state. The information density in those 6 numbers is fundamentally different from 6 classical neurons — they encode correlations across exponentially many basis states.

2. **Structured weight generation:** The quantum circuit enforces a kind of implicit regularisation. Weights generated by a smooth manifold (the quantum state space) cannot be arbitrary — they lie on a structured submanifold of `ℝ^418`.

3. **Research motivation:** Demonstrating that quantum circuits can replace classical weight generation, and that the resulting PDE solutions are accurate. This is a proof-of-concept for quantum advantage in scientific ML.

### The Full Forward Pass

```
Training step:
  1. Sample (x, y, t) collocation points
  2. QuantumWeightGenerator.forward():
       a. inputs = zeros(6)                        → default angle vector
       b. AngleEmbedding(inputs) on 6 qubits       → encode input
       c. StronglyEntanglingLayers(q_weights)       → parameterised evolution
       d. [⟨Z₀⟩, ..., ⟨Z₅⟩] → stack → float32   → 6 classical values
       e. proj(z_vals) → flat(418,)                → linear map
       f. slice into {W1(112,), W2(272,), W3(34,)} → MLP weight dict
  3. TargetPINN.forward(x, y, t, weights):
       a. FourierFeatureMap([x,y,t]) → (N, 6) features
       b. F.linear(feats, W1, b1) → tanh           → (N, 16)
       c. F.linear(h, W2, b2) → tanh               → (N, 16)
       d. F.linear(h, W3, b3)                       → (N, 2): u, v
  4. compute_burgers_loss(u, v, x, y, t):
       a. autograd: u_t, u_x, u_y, u_xx, u_yy, v_t, v_x, v_y, v_xx, v_yy
       b. f_u = u_t + u·u_x + v·u_y − ν·(u_xx+u_yy)
       c. f_v = v_t + u·v_x + v·v_y − ν·(v_xx+v_yy)
       d. pde_loss = mean(f_u²) + mean(f_v²)
       e. bc_loss  = MSE at IC points
  5. total = pde_loss + λ·bc_loss
  6. backward() → gradients flow to q_weights and proj
  7. optimizer.step()
```

---

## 7. Backpropagation Through Quantum Circuits

### The Problem

Standard backprop requires computing `∂L/∂θ` for every parameter. In a classical network this is a chain rule through matrix multiplications. In a quantum circuit the computation is a sequence of unitary matrix exponentials — how do you differentiate through that?

### Method 1: Parameter Shift Rule (Analytical, Hardware-Compatible)

For any gate of the form `G(θ) = exp(−i θ/2 · P)` where `P` is a Pauli operator:

```
∂⟨O⟩/∂θ = ½ · [⟨O⟩(θ + π/2) − ⟨O⟩(θ − π/2)]
```

The gradient is computed by running the circuit **twice** with shifted parameters — no approximation, exact analytical gradient. This works on real quantum hardware because it only requires circuit evaluations.

**Cost:** 2 circuit evaluations per parameter. For 36 parameters, that's 72 evaluations per gradient step.

### Method 2: Backprop (Used in This Project)

PennyLane's `diff_method="backprop"` uses classical automatic differentiation through a **statevector simulator**. The full quantum state `|ψ⟩ ∈ ℂ^64` is stored in memory, and JAX/PyTorch computes the gradient by differentiating through the matrix multiplications that implement the gates.

**Cost:** Single forward pass, same as classical AD. Fast on CPU for small circuits.

**Trade-off:** Cannot run on real quantum hardware (requires full statevector). Only valid for simulation. For hardware deployment, switch to `diff_method="parameter-shift"`.

### Method 3: Adjoint Differentiation

A memory-efficient variant that runs the circuit forward and then backward in reverse, similar to classical backprop but adapted for unitary operations. Used by PennyLane's `lightning.qubit` device for larger circuits.

---

## 8. Two-Stage Optimisation

### Why Two Stages?

Neural network loss landscapes are non-convex. First-order methods (Adam) explore broadly but converge slowly near minima. Second-order methods (L-BFGS) converge precisely but are expensive and get stuck in bad basins if started poorly.

The combination: Adam finds a good basin, L-BFGS refines to a sharp minimum within it.

### Stage 1: Adam

**Adam** (Kingma & Ba, 2014) is an adaptive gradient method with momentum:

```
m_t = β₁·m_{t-1} + (1−β₁)·g_t          (first moment — gradient)
v_t = β₂·v_{t-1} + (1−β₂)·g_t²         (second moment — squared gradient)
θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε) (update)
```

Where `m̂_t`, `v̂_t` are bias-corrected estimates. The adaptive per-parameter learning rate means parameters with consistently large gradients get smaller steps (prevents oscillation) and sparse parameters get larger steps.

**Why Adam first:** it's robust to poor initialisations, handles noisy gradients well, and makes fast initial progress across the loss landscape.

### Stage 2: L-BFGS

**L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a quasi-Newton method. It approximates the inverse Hessian `H⁻¹` using the last `history_size` gradient vectors:

```
θ_{t+1} = θ_t − H⁻¹_t · g_t
```

Newton's method (`H⁻¹g`) would give the exact step to the minimum of a quadratic approximation. L-BFGS approximates this using only `O(n·history_size)` memory instead of `O(n²)` for the full Hessian.

**Strong Wolfe line search:** at each step, L-BFGS performs a 1D line search along the Newton direction to find a step size satisfying:
1. Sufficient decrease (Armijo condition)
2. Curvature condition (Wolfe condition)

This is why the closure is called multiple times per outer step — the line search evaluates the loss at different step sizes.

**Why L-BFGS second:** near a good minimum, the loss landscape is approximately quadratic, and L-BFGS converges superlinearly (faster than any fixed-order method). Adam cannot compete here.

**The closure call count explained:** with `lbfgs.steps=50` and `lbfgs.max_iter=20`, you see ~1000 closure calls. Each outer step may call the closure 20 times for the line search — this is normal. The loss you see printed is the final value after the line search succeeds.

---

## 9. Loss Function Design

### MSE vs. Other Norms

We use mean squared error throughout. Why not mean absolute error (L1)?

- L2 (MSE) is differentiable everywhere; L1 has a subgradient at zero
- L2 penalises large residuals quadratically — outlier collocation points with high PDE violation are penalised harder, which guides the network toward globally consistent solutions
- L2 is the natural norm for Sobolev spaces H^1, which is where PDE solutions live mathematically

### The λ Weighting

```
L_total = L_pde + λ · L_bc
```

These two losses have different scales. `L_pde` involves second derivatives of the network — typically large values early in training. `L_bc` is an MSE against known values — typically small if the IC is smooth.

Without λ, the optimiser ignores BC (it's numerically small relative to the PDE residual). `λ = 10–50` re-balances them.

**Adaptive weighting** (advanced): some works compute λ automatically each step as `λ = mean(|∇L_pde|) / mean(|∇L_bc|)` — normalising the gradient magnitudes. This is more robust than a fixed λ.

### Why the IC and Not Boundary Conditions?

Burgers' equation on `[-1,1]²` with periodic or Dirichlet boundaries requires specifying values at spatial edges. We use only the **initial condition** (IC at `t=0`) and let the network extrapolate to `t>0`. The IC is strong enough to uniquely determine the solution given the PDE — this is the Cauchy problem.

For a fully constrained problem, you would also enforce:
- Periodic BC: `u(-1, y, t) = u(1, y, t)` and `u(x, -1, t) = u(x, 1, t)`
- Or Dirichlet BC: specify `u` on all four spatial edges

---

## 10. Metrics and Evaluation

### Training Metrics

The three columns printed during training:

```
total = pde_loss + λ · bc_loss
```

**pde_loss** is the most important. It measures how well the network satisfies the Burgers PDE at the collocation points:

```
pde_loss = (1/N) Σ (f_u² + f_v²)
```

The pointwise PDE residual at a single point is `√pde_loss`. At final value `0.00017`:
```
√0.00017 ≈ 0.013  →  1.3% average PDE violation per point
```

**bc_loss** reaching `0.000000` means the IC is satisfied to floating-point precision — essentially exact.

### Relative L2 Error (Gold Standard)

The training loss is evaluated on training points. The true quality metric is the **relative L2 error** on a held-out grid:

```
ε_rel = ‖u_pred − u_ref‖₂ / ‖u_ref‖₂
```

Where `u_ref` is either a high-resolution numerical solution or an analytical solution.

For this IC, there is no closed-form analytical solution for `t > 0`, but you can generate a reference using a classical high-order solver (e.g., spectral methods with 256² grid points).

**Target values:**
- `ε_rel < 1%` — publication quality for a small network
- `ε_rel < 0.1%` — competitive with classical solvers at this resolution

### Visual Checks from the Plot

1. **t=0 curve matches dashed IC exactly** → bc_loss is truly zero
2. **Smooth monotonic decay of amplitude** → physical diffusion is captured
3. **No oscillations between curves** → no spectral aliasing, good generalisation
4. **Curves don't cross each other unexpectedly** → temporal ordering is respected
5. **Solution at late t (t=1) converges toward zero** → viscous dissipation is correct

---

## 11. Known Failure Modes

### Barren Plateaus

The most serious problem in quantum ML. McClean et al. (2018) proved that for random initialisations of deep circuits, the variance of the gradient vanishes exponentially in the number of qubits:

```
Var[∂L/∂θ] ~ O(1/2^N)
```

For `N=6`, this variance is `~1/64` — small but not zero. For `N=20`, it's `~1/10^6` — the gradient signal is completely buried in noise. This means quantum ML does not automatically scale by adding qubits.

**Mitigations in this project:**
- Small circuit (`N=6`) where gradients are still meaningful
- Shallow depth (`N_LAYERS=2`) minimises the plateau effect
- Small initialisation (`q_weights * 0.1`) keeps the circuit near identity initially

**Mitigations in research:**
- Layer-by-layer training (train one layer at a time)
- Problem-inspired ansätze instead of random circuits
- Classical shadows for efficient gradient estimation
- Quantum Natural Gradient (QNG)

### Mode Collapse

The quantum generator produces the same weights regardless of the input — the circuit gets stuck in a state where all qubits give the same output. Detectable by watching `z_vals` in the generator: if all 6 are identical, you have collapse.

Fix: perturb `q_weights` initialisation with larger noise, or add an input-dependent embedding.

### λ Imbalance

If `lambda_bc` is too small: the network learns the PDE interior but ignores the IC — the solution is a valid Burgers solution but the wrong one (wrong initial conditions). The plot will show the t=0 curve far from the dashed IC reference.

If `lambda_bc` is too large: the network memorises the IC but learns nothing about the PDE interior. The pde_loss stays high while bc_loss is zero.

---

## 12. What to Try Next

### A. Data Re-uploading

Current design: input is encoded once, then variational layers run. In **data re-uploading** (Pérez-Salinas et al., 2020), the input is re-injected at every variational layer:

```
Layer 1: Encode(x) → Variational(θ₁)
Layer 2: Encode(x) → Variational(θ₂)   ← same x, new angles
Layer 3: Encode(x) → Variational(θ₃)
```

This makes the circuit a universal function approximator — analogous to depth in classical networks. It dramatically increases the effective frequency content of the quantum function and removes a key bottleneck of single-encoding circuits.

**How to implement:** in `qnn_generator.py`, interleave `qml.AngleEmbedding` between each `StronglyEntanglingLayers`.

### B. Quantum Natural Gradient (QNG)

Standard gradient descent uses the Euclidean geometry of parameter space. QNG uses the **Fubini-Study metric** — the natural geometry of the quantum state space — to compute:

```
θ_{t+1} = θ_t − α · F⁺ · g_t
```

Where `F` is the quantum Fisher information matrix (QFI). This is analogous to natural gradient in classical ML but exactly adapted to quantum geometry. Steps in QNG are geometry-aware — a step of size `δ` in parameter space corresponds to a controlled change in the quantum state.

**Effect:** dramatically faster convergence, especially near barren plateaus where vanilla gradients are tiny.

**Implementation:** `qml.QNGOptimizer` in PennyLane.

### C. Adaptive Collocation (RAR)

Instead of random uniform sampling, **Residual-Adaptive Refinement** (Wu et al., 2022) places more collocation points where the PDE residual is large:

```
1. Train on uniform grid
2. Evaluate residual on a dense evaluation grid
3. Sample new points proportional to residual magnitude
4. Retrain with augmented point set
5. Repeat
```

This is critical for problems with sharp gradients or shocks — the network gets more training signal where it struggles.

### D. Fourier Frequency Curriculum

Instead of fixed σ for the frequency matrix, start training with low σ (low frequencies) and progressively increase it. This mirrors the idea in multi-scale learning — learn coarse structure first, then refine. Can prevent the network from getting stuck in high-frequency noise early.

### E. Larger Quantum Circuits and Better Ansätze

| Ansatz | Properties |
|---|---|
| StronglyEntanglingLayers (current) | General-purpose, good for CPU prototyping |
| QAOA-inspired | Problem-structure-aware, may match PDE symmetries |
| Hamiltonian Variational Ansatz (HVA) | Encodes physical symmetries, avoids barren plateaus |
| Hardware-Efficient Ansatz (HEA) | Minimal gate depth for real hardware |
| Tensor-network-inspired | Efficient for 1D problems, scales better |

### F. Physics-Aware Loss Weighting

Replace fixed `lambda_bc` with a self-adaptive scheme (Wang et al., 2022):

```python
lambda_bc = mean(|∇_θ L_pde|) / mean(|∇_θ L_bc|)
```

Computed each step. Ensures BC and PDE gradients have equal influence on the optimiser — eliminates the manual tuning of `lambda_bc`.

### G. Transfer Learning

Train on a low-viscosity (easy) problem, then fine-tune on high-viscosity (hard). Or train on 1D Burgers first, then transfer to 2D. The quantum circuit weights provide a good initialisation rather than random angles.

### H. Ensemble of Quantum Circuits

Instead of one circuit generating weights, train `K` circuits and combine their outputs:

```
{W1, W2, W3} = (1/K) Σ_k gen_k()
```

Or weight the ensemble by a learned attention mechanism. This reduces variance in weight generation and provides implicit regularisation.

### I. Inverse Problem

The current setup is the **forward problem**: given the PDE and IC, find `u(x,y,t)`. PINNs naturally extend to **inverse problems**: given some observations of `u`, identify unknown parameters (e.g., ν).

This is where PINNs show the most advantage over classical solvers — the same framework handles both without modification. Just add ν as a learnable parameter.

### J. Deploying on Real Quantum Hardware

To run the quantum circuit on IBM, Google, or IonQ hardware:
1. Change `diff_method="backprop"` to `diff_method="parameter-shift"` (hardware-compatible)
2. Change device to `qml.device("qiskit.ibmq", ...)` or similar
3. Add noise mitigation (zero-noise extrapolation, readout error correction)
4. Reduce `N_LAYERS` — hardware has finite coherence time (circuit depth limit)
5. Transpile gates to native gate set of the hardware

The exported `static_weights.pt` remains fully classical — hardware is only needed for training.

---

## Key Papers

| Paper | What It Introduces |
|---|---|
| Raissi et al. 2019 | Physics-Informed Neural Networks |
| Tancik et al. 2020 | Fourier Features for coordinate networks |
| McClean et al. 2018 | Barren plateaus in quantum ML |
| Mitarai et al. 2018 | Parameter shift rule |
| Pérez-Salinas et al. 2020 | Data re-uploading |
| Ha, Dai & Le 2016 | Hypernetworks |
| Wang et al. 2022 | Self-adaptive loss weights for PINNs |
| Wu et al. 2022 | Residual-adaptive refinement for PINNs |
| Cerezo et al. 2021 | Review of variational quantum algorithms |
| Schuld & Petruccione 2021 | *Machine Learning with Quantum Computers* (book) |
| Kingma & Ba 2014 | Adam optimiser |

---

## Glossary

| Term | Definition |
|---|---|
| Ansatz | A parameterised circuit architecture (German: "approach") |
| Barren plateau | Region of parameter space where gradients vanish exponentially |
| Collocation points | Spatial-temporal points where the PDE is enforced |
| Entanglement | Quantum correlation between qubits with no classical analogue |
| Expressibility | How much of the unitary space a circuit can reach |
| Hypernetwork | A network that generates another network's weights |
| QNode | PennyLane's quantum node — a differentiable quantum circuit |
| Qubit | Quantum bit; superposition of 0 and 1 |
| Spectral bias | MLPs' tendency to learn low frequencies first |
| Statevector | The full `2^N`-dimensional complex vector representing an N-qubit state |
| Strong Wolfe | Line search conditions ensuring sufficient decrease and curvature |
| Superposition | A qubit being in a quantum combination of both 0 and 1 simultaneously |
| Unitary | A matrix U where U†U = I; preserves quantum state normalization |
| VQC | Variational Quantum Circuit — a parameterised quantum circuit used for ML |
