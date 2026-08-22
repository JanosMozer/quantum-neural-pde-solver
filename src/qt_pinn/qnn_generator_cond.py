"""Conditioned weight generators: nu -> MLP weights.

Motivation
----------
The earlier NS generator (`qnn_generator_ns.py`) evaluates its circuit on a
constant input, so it emits ONE fixed probability vector. A fixed state
prepared by a fixed circuit can be computed classically once and cached, so
that design cannot exhibit quantum advantage even in principle -- it is a
reparameterisation of a single weight vector, not a computation.

Here the generator is a genuine map: it takes a PDE parameter (kinematic
viscosity nu) and returns the MLP weights that solve NS for that nu. The
quantum circuit therefore has to represent a *function* over a family of
tasks, which is the setting where circuit expressivity is a well-posed
question.

Encoding
--------
Data re-uploading (Perez-Salinas et al. 2020): nu is re-encoded before every
variational block. Frequencies follow Schuld, Sweke & Meyer (2021): the
circuit realises a Fourier series whose spectrum is set by the encoding
scales. Default is a *linear* ladder w_i = (i+1)·π so that, over the
normalised nu∈[0,1] interval, no qubit wraps more than ~n/2 times
(RY period 4π). The geometric ladder w_i = 2^i·π is available but aliases
heavily for n≳6 and is a known confounder, not a feature.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import pennylane as qml


def make_freq_ladder(n_qubits: int, mode: str = "linear") -> torch.Tensor:
    """Encoding frequency scales (radians) for AngleEmbedding / sin-cos features.

    linear:   (i+1)·π  — mild wrapping, default
    geometric: 2^i ·π — exponentially wide spectrum; aliases for n≳6
    """
    if mode == "linear":
        ladder = torch.arange(1, n_qubits + 1, dtype=torch.float32)
    elif mode == "geometric":
        ladder = torch.tensor([2.0 ** i for i in range(n_qubits)], dtype=torch.float32)
    else:
        raise ValueError(f"unknown freq_mode={mode!r}; use 'linear' or 'geometric'")
    return ladder * math.pi


def normalize_nu(
    nu: torch.Tensor,
    nu_lo: float,
    nu_hi: float,
    encode: str = "linear",
) -> torch.Tensor:
    """Map nu -> z in [0,1].  log encoding matches log-uniform training draws."""
    nu = nu.reshape(-1, 1)
    if encode == "linear":
        return (nu - nu_lo) / (nu_hi - nu_lo)
    if encode == "log":
        return (torch.log(nu) - math.log(nu_lo)) / (math.log(nu_hi) - math.log(nu_lo))
    raise ValueError(f"unknown nu_encode={encode!r}")


def weight_sizes(in_dim: int, h1: int, h2: int, out_dim: int = 3) -> tuple[int, int, int]:
    return (in_dim * h1 + h1, h1 * h2 + h2, h2 * out_dim + out_dim)


def target_init_vector(in_dim: int, h1: int, h2: int, out_dim: int = 3,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """Flat vector holding a standard Xavier init for the target MLP.

    Used as the bias of the generator's output layer so the hypernetwork
    *starts* at a healthy classical initialisation (Chang, Flokas & Lipson,
    ICLR 2020). Without this the generated weights have arbitrary scale and
    the target network saturates or explodes before any learning happens.
    """
    parts = []
    for fan_in, fan_out in ((in_dim, h1), (h1, h2), (h2, out_dim)):
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        w = (torch.rand(fan_out * fan_in, generator=generator) * 2 - 1) * bound
        parts.append(w)
        parts.append(torch.zeros(fan_out))
    return torch.cat(parts)


class _WeightSplitter(nn.Module):
    """Shared plumbing: flat vector -> {W1, W2, W3} dict."""

    def __init__(self, in_dim: int, h1: int, h2: int, out_dim: int = 3) -> None:
        super().__init__()
        self.dims = (in_dim, h1, h2, out_dim)
        self.w1_size, self.w2_size, self.w3_size = weight_sizes(in_dim, h1, h2, out_dim)
        self.total_weights = self.w1_size + self.w2_size + self.w3_size

    def _split(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        a, b = self.w1_size, self.w1_size + self.w2_size
        return {"W1": flat[..., :a], "W2": flat[..., a:b], "W3": flat[..., b:]}

    def _init_head(self, head: nn.Linear, scale: float = 0.01) -> None:
        """Start the generator at the Xavier init, with a small learnable
        deviation that the nu-dependent signal can grow into."""
        with torch.no_grad():
            head.weight.mul_(scale)
            head.bias.copy_(target_init_vector(*self.dims))


def _make_circuit(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(nu_encoded: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """nu_encoded: (n_qubits,) or (batch, n_qubits) rotation angles."""
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for l in range(n_layers):
            # data re-uploading: the parameter is re-injected at every layer
            qml.AngleEmbedding(nu_encoded, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[l : l + 1], wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    return circuit


class ConditionedQuantumGenerator(_WeightSplitter):
    """nu -> re-uploading circuit -> basis probabilities -> MLP weights.

    Trainable: circuit angles (n_layers, n_qubits, 3), input scaling, projection.
    """

    ARCH = "reupload"

    def __init__(
        self,
        in_dim: int,
        h1: int = 32,
        h2: int = 32,
        out_dim: int = 3,
        n_qubits: int = 8,
        n_layers: int = 3,
        bottleneck_width: int = 64,
        nu_range: tuple[float, float] = (0.05, 0.5),
        freq_mode: str = "linear",
    ) -> None:
        super().__init__(in_dim, h1, h2, out_dim)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_states = 2 ** n_qubits
        self.nu_lo, self.nu_hi = nu_range
        self.freq_mode = freq_mode
        self._circuit = _make_circuit(n_qubits, n_layers)

        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.freq_scale = nn.Parameter(make_freq_ladder(n_qubits, freq_mode))

        self.proj = nn.Sequential(
            nn.Linear(self.n_states, bottleneck_width),
            nn.Tanh(),
            nn.Linear(bottleneck_width, self.total_weights),
        )
        self._init_head(self.proj[-1])

    def _encode(self, nu: torch.Tensor) -> torch.Tensor:
        """Map nu -> (batch, n_qubits) angles, normalised to a stable range."""
        nu = nu.reshape(-1, 1)
        z = (nu - self.nu_lo) / (self.nu_hi - self.nu_lo)  # ~[0,1]
        return z * self.freq_scale.unsqueeze(0)

    def forward(self, nu: torch.Tensor) -> dict[str, torch.Tensor]:
        """nu: (batch,) -> weight dict with leading batch dim."""
        out_device = next(self.proj.parameters()).device
        angles = self._encode(nu)
        # PennyLane's statevector simulator runs on CPU; cross the device
        # boundary here. .to() is differentiable, so grads still reach freq_scale.
        probs = self._circuit(angles.cpu(), self.q_weights.cpu()).float()
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        # probs sum to 1 over 2^n states, so each entry is O(2^-n) and the
        # projection would see a vanishing signal. Rescale to zero-mean O(1).
        probs = probs.to(out_device) * self.n_states - 1.0
        return self._split(self.proj(probs))


class ConditionedClassicalGenerator(_WeightSplitter):
    """Parameter-matched classical control: nu -> MLP -> MLP weights.

    Mirrors the quantum path exactly (same nu normalisation, same frequency
    ladder as a fixed sinusoidal feature map, same projection head), so the
    only difference is whether the intermediate representation comes from an
    entangled state or a classical nonlinearity.
    """

    ARCH = "reupload"

    def __init__(
        self,
        in_dim: int,
        h1: int = 32,
        h2: int = 32,
        out_dim: int = 3,
        n_qubits: int = 8,
        bottleneck_width: int = 64,
        nu_range: tuple[float, float] = (0.05, 0.5),
        freq_mode: str = "linear",
        hidden: int | None = None,
    ) -> None:
        super().__init__(in_dim, h1, h2, out_dim)
        self.n_states = 2 ** n_qubits
        self.nu_lo, self.nu_hi = nu_range
        self.freq_mode = freq_mode
        self.freq_scale = nn.Parameter(make_freq_ladder(n_qubits, freq_mode))

        # stand-in for the circuit: nu -> n_states "probabilities"
        hidden = hidden if hidden is not None else n_qubits * 4
        self.encoder = nn.Sequential(
            nn.Linear(2 * n_qubits, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.n_states),
            nn.Softmax(dim=-1),   # match the simplex constraint of qml.probs
        )
        self.proj = nn.Sequential(
            nn.Linear(self.n_states, bottleneck_width),
            nn.Tanh(),
            nn.Linear(bottleneck_width, self.total_weights),
        )
        self._init_head(self.proj[-1])

    def forward(self, nu: torch.Tensor) -> dict[str, torch.Tensor]:
        nu = nu.reshape(-1, 1)
        z = (nu - self.nu_lo) / (self.nu_hi - self.nu_lo)
        ang = z * self.freq_scale.unsqueeze(0)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        probs = self.encoder(feats) * self.n_states - 1.0   # matched rescaling
        return self._split(self.proj(probs))


def _make_expectation_circuit(n_qubits: int, n_layers: int):
    """Re-uploading circuit with Pauli-Z expectation readout (dim = n_qubits)."""
    dev = qml.device("default.qubit", wires=n_qubits)
    obs = [qml.PauliZ(w) for w in range(n_qubits)]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(nu_encoded: torch.Tensor, weights: torch.Tensor) -> list:
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for l in range(n_layers):
            qml.AngleEmbedding(nu_encoded, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[l : l + 1], wires=range(n_qubits))
        return [qml.expval(o) for o in obs]

    return circuit


class ConditionedQuantumGeneratorV2(_WeightSplitter):
    """v2: log(nu) encoding + Z expectations -> weights.

    Motivation (from ns_par_q_s0 failure mode):
      - Training samples nu log-uniformly but v1 encoded linear nu -> extrap-lo
        is off-manifold in input space.
      - Full 2^n probability vector is a harsh readout; Z expectations are the
        standard QML interface and give smooth, O(1) features in [-1, 1].
    """

    ARCH = "expect"

    def __init__(
        self,
        in_dim: int,
        h1: int = 32,
        h2: int = 32,
        out_dim: int = 3,
        n_qubits: int = 6,
        n_layers: int = 6,
        bottleneck_width: int = 64,
        nu_range: tuple[float, float] = (0.05, 0.5),
        freq_mode: str = "linear",
        nu_encode: str = "log",
    ) -> None:
        super().__init__(in_dim, h1, h2, out_dim)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feat_dim = n_qubits
        self.nu_lo, self.nu_hi = nu_range
        self.freq_mode = freq_mode
        self.nu_encode = nu_encode
        self._circuit = _make_expectation_circuit(n_qubits, n_layers)

        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.freq_scale = nn.Parameter(make_freq_ladder(n_qubits, freq_mode))

        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, bottleneck_width),
            nn.Tanh(),
            nn.Linear(bottleneck_width, self.total_weights),
        )
        self._init_head(self.proj[-1])

    def _encode(self, nu: torch.Tensor) -> torch.Tensor:
        z = normalize_nu(nu, self.nu_lo, self.nu_hi, self.nu_encode)
        return z * self.freq_scale.unsqueeze(0)

    def forward(self, nu: torch.Tensor) -> dict[str, torch.Tensor]:
        out_device = next(self.proj.parameters()).device
        angles = self._encode(nu)
        raw = self._circuit(angles.cpu(), self.q_weights.cpu())
        feats = torch.stack(raw, dim=-1).float().to(out_device)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return self._split(self.proj(feats))


class ConditionedClassicalGeneratorV2(_WeightSplitter):
    """Matched v2 control: log(nu) sin/cos -> MLP -> n_qubits pseudo-expectations."""

    ARCH = "expect"

    def __init__(
        self,
        in_dim: int,
        h1: int = 32,
        h2: int = 32,
        out_dim: int = 3,
        n_qubits: int = 6,
        bottleneck_width: int = 64,
        nu_range: tuple[float, float] = (0.05, 0.5),
        freq_mode: str = "linear",
        nu_encode: str = "log",
        hidden: int | None = None,
    ) -> None:
        super().__init__(in_dim, h1, h2, out_dim)
        self.feat_dim = n_qubits
        self.nu_lo, self.nu_hi = nu_range
        self.freq_mode = freq_mode
        self.nu_encode = nu_encode
        self.freq_scale = nn.Parameter(make_freq_ladder(n_qubits, freq_mode))

        hidden = hidden if hidden is not None else n_qubits * 4
        self.encoder = nn.Sequential(
            nn.Linear(2 * n_qubits, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.feat_dim),
            nn.Tanh(),   # match Z expectation range [-1, 1]
        )
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, bottleneck_width),
            nn.Tanh(),
            nn.Linear(bottleneck_width, self.total_weights),
        )
        self._init_head(self.proj[-1])

    def forward(self, nu: torch.Tensor) -> dict[str, torch.Tensor]:
        z = normalize_nu(nu, self.nu_lo, self.nu_hi, self.nu_encode)
        ang = z * self.freq_scale.unsqueeze(0)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self._split(self.proj(self.encoder(feats)))


if __name__ == "__main__":
    IN_DIM = 10
    nu = torch.tensor([0.05, 0.2, 0.5])

    q = ConditionedQuantumGenerator(IN_DIM, n_qubits=8, bottleneck_width=64)
    w = q(nu)
    assert w["W1"].shape == (3, q.w1_size), w["W1"].shape
    # the whole point: different nu must give different weights
    spread = (w["W1"][0] - w["W1"][2]).abs().max().item()
    assert spread > 1e-6, "quantum generator ignores its input"
    print(f"PASS quantum: total={q.total_weights} params={sum(p.numel() for p in q.parameters()):,} "
          f"nu-sensitivity={spread:.4f}")

    c = ConditionedClassicalGenerator(IN_DIM, n_qubits=8, bottleneck_width=64)
    wc = c(nu)
    assert wc["W1"].shape == (3, c.w1_size)
    spread_c = (wc["W1"][0] - wc["W1"][2]).abs().max().item()
    assert spread_c > 1e-6
    print(f"PASS classical: params={sum(p.numel() for p in c.parameters()):,} "
          f"nu-sensitivity={spread_c:.4f}")

    q2 = ConditionedQuantumGeneratorV2(IN_DIM, n_qubits=6, n_layers=6, bottleneck_width=64)
    w2 = q2(nu)
    spread2 = (w2["W1"][0] - w2["W1"][2]).abs().max().item()
    assert spread2 > 1e-6
    print(f"PASS quantum v2: params={sum(p.numel() for p in q2.parameters()):,} "
          f"nu-sensitivity={spread2:.4f}")

    c2 = ConditionedClassicalGeneratorV2(IN_DIM, n_qubits=6, bottleneck_width=64)
    wc2 = c2(nu)
    assert wc2["W1"].shape == (3, c2.w1_size)
    print(f"PASS classical v2: params={sum(p.numel() for p in c2.parameters()):,}")
