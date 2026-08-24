"""Input-conditioned Burgers PINNs: VQC vs parameter-matched classical.

Both models compute (u,v) = f(x,y,t) at each collocation point. The quantum
path uses data re-uploading; the classical path uses the same angle features
with sin/cos + a matched small MLP. This is the architecturally valid test
that was missing from the hypernetwork experiments (Tier 0-C).

Hard-IC ansatz
--------------
With soft IC, u=v=0 is an exact Burgers solution with zero PDE residual, so
both models collapse into it and the PDE metric becomes meaningless. Setting
hard_ic=True uses

    u(x,y,t) = u_IC(x,y) + t * N_u(x,y,t)

so the IC holds by construction (bc_loss == 0) and the trivial solution is
unreachable. ic_fn is injected by the caller to keep the PDE's initial
condition defined in one place (pdes/burgers2d/physics_loss.py).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import pennylane as qml

OUT_DIM = 2

ICFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def make_burgers_freq_ladder(n_qubits: int) -> torch.Tensor:
    """Dimensionless encoding frequencies for encode_burgers_coords.

    Angles are pi * z * freq with z in [0,1]. Qubit i encodes channel i%3
    (x, y, t) at harmonic 1 + i//3.

    x and y use even freqs 2,4,6,... which reproduce the IC exactly
    (sin(2*pi*zx) = -sin(pi*x) for x in [-1,1]) and are periodic over the
    domain. t uses odd freqs 1,3,5,...: the solution is not periodic in time,
    and any even freq would wrap a whole number of periods over t in [0,1],
    making t=0 and t=1 indistinguishable to the circuit.
    """
    freqs = []
    for i in range(n_qubits):
        harmonic = 1 + i // 3
        freqs.append(2.0 * harmonic if i % 3 < 2 else 2.0 * harmonic - 1.0)
    return torch.tensor(freqs, dtype=torch.float32)


def encode_burgers_coords(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    n_qubits: int,
    freq_scale: torch.Tensor,
) -> torch.Tensor:
    """Map (x,y,t) on the Burgers domain to (batch, n_qubits) rotation angles.

    Domain: x,y in [-1,1], t in [0,1]. Normalised to z in [0,1] then scaled by
    a learnable frequency ladder.
    """
    zx = (x.reshape(-1, 1) + 1.0) * 0.5
    zy = (y.reshape(-1, 1) + 1.0) * 0.5
    zt = t.reshape(-1, 1)
    base = [zx, zy, zt]
    cols = [base[i % 3] for i in range(n_qubits)]
    z = torch.cat(cols, dim=-1)
    return math.pi * z * freq_scale.unsqueeze(0)


def _make_burgers_circuit(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)
    obs = [qml.PauliZ(w) for w in range(n_qubits)]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(angles: torch.Tensor, weights: torch.Tensor) -> list:
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for l in range(n_layers):
            qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[l : l + 1], wires=range(n_qubits))
        return [qml.expval(o) for o in obs]

    return circuit


def _match_hidden(in_dim: int, out_dim: int, target: int) -> int:
    """Smallest hidden h with in*h+h + h*out+out >= target (single hidden layer)."""
    for h in range(1, 256):
        if in_dim * h + h + h * out_dim + out_dim >= target:
            return h
    return 256


class _HardICMixin:
    """Applies u = u_IC + t * N when hard_ic is enabled."""

    hard_ic: bool
    ic_fn: ICFn | None

    def _init_hard_ic(self, hard_ic: bool, ic_fn: ICFn | None) -> None:
        if hard_ic and ic_fn is None:
            raise ValueError("hard_ic=True requires ic_fn (e.g. physics_loss.ic_values)")
        self.hard_ic = hard_ic
        self.ic_fn = ic_fn

    def _apply_ansatz(
        self,
        raw: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if not self.hard_ic:
            return raw
        u0, v0 = self.ic_fn(x, y)
        ic = torch.stack([u0.reshape(-1), v0.reshape(-1)], dim=-1)
        return ic + t.reshape(-1, 1) * raw


class BurgersVQCPINN(_HardICMixin, nn.Module):
    """(x,y,t) -> (u,v) via a re-uploading VQC + linear head."""

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 4,
        hard_ic: bool = False,
        ic_fn: ICFn | None = None,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self._circuit = _make_burgers_circuit(n_qubits, n_layers)
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.freq_scale = nn.Parameter(make_burgers_freq_ladder(n_qubits))
        self.head = nn.Linear(n_qubits, OUT_DIM)
        self._init_hard_ic(hard_ic, ic_fn)

    def _raw(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dev = self.head.weight.device
        angles = encode_burgers_coords(x, y, t, self.n_qubits, self.freq_scale)
        out = self._circuit(angles.cpu(), self.q_weights.cpu())
        feats = torch.stack(out, dim=-1).float().to(dev)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return self.head(feats)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._apply_ansatz(self._raw(x, y, t), x, y, t)


class BurgersClassicalPINN(_HardICMixin, nn.Module):
    """Matched classical control: same angles -> sin/cos -> MLP -> (u,v)."""

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 4,
        hidden: int | None = None,
        hard_ic: bool = False,
        ic_fn: ICFn | None = None,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers  # used only for param matching
        self.freq_scale = nn.Parameter(make_burgers_freq_ladder(n_qubits))
        in_dim = 2 * n_qubits
        if hidden is None:
            q_params = n_layers * n_qubits * 3 + n_qubits * OUT_DIM + OUT_DIM
            hidden = _match_hidden(in_dim, OUT_DIM, q_params)
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, OUT_DIM),
        )
        self._init_hard_ic(hard_ic, ic_fn)

    @classmethod
    def matched_to(
        cls,
        q_model: BurgersVQCPINN,
        hard_ic: bool | None = None,
        ic_fn: ICFn | None = None,
    ) -> BurgersClassicalPINN:
        n_q = sum(p.numel() for p in q_model.parameters())
        hidden = _match_hidden(2 * q_model.n_qubits, OUT_DIM, n_q)
        c = cls(
            q_model.n_qubits,
            q_model.n_layers,
            hidden=hidden,
            hard_ic=q_model.hard_ic if hard_ic is None else hard_ic,
            ic_fn=q_model.ic_fn if ic_fn is None else ic_fn,
        )
        with torch.no_grad():
            c.freq_scale.copy_(q_model.freq_scale)
        return c

    def _raw(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ang = encode_burgers_coords(x, y, t, self.n_qubits, self.freq_scale)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.net(feats)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._apply_ansatz(self._raw(x, y, t), x, y, t)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pdes.burgers2d.physics_loss import ic_values

    x = torch.linspace(-1, 1, 8)
    y = torch.zeros(8)
    t = torch.linspace(0, 1, 8)

    q = BurgersVQCPINN(6, 4)
    c = BurgersClassicalPINN.matched_to(q)
    assert q(x, y, t).shape == (8, 2) and c(x, y, t).shape == (8, 2)
    nq, nc = count_params(q), count_params(c)
    assert nc >= nq, f"classical {nc} must match/exceed quantum {nq}"
    spread = (q(x, y, t)[0] - q(x, y, torch.ones(8))[0]).abs().max().item()
    assert spread > 1e-6, "VQC must depend on t"
    print(f"PASS soft-IC: q={nq} params  c={nc} params  t-sensitivity={spread:.4f}")

    # freq ladder must reproduce the IC fundamental and not alias in time
    ladder = make_burgers_freq_ladder(6)
    xg = torch.linspace(-1, 1, 32)
    yg = torch.linspace(-1, 1, 32)
    ang = encode_burgers_coords(xg, yg, torch.zeros(32), 6, ladder)
    assert torch.allclose(torch.sin(ang[:, 0]), -torch.sin(math.pi * xg), atol=1e-5), \
        "qubit 0 must carry the IC fundamental in x"
    assert torch.allclose(torch.cos(ang[:, 1]), -torch.cos(math.pi * yg), atol=1e-5), \
        "qubit 1 must carry the IC fundamental in y"
    # every time channel must separate t=0 from t=1 modulo 2*pi
    a0 = encode_burgers_coords(xg, yg, torch.zeros(32), 6, ladder)
    a1 = encode_burgers_coords(xg, yg, torch.ones(32), 6, ladder)
    for ch in range(2, 6, 3):
        gap = ((a1[:, ch] - a0[:, ch]) % (2 * math.pi)).abs().min().item()
        assert gap > 1e-3, f"time channel on qubit {ch} aliases over t in [0,1]"
    print("PASS freq ladder: IC fundamental exact, time channels non-aliasing")

    # hard IC: exact at t=0 for both models
    for name, m in [("quantum", BurgersVQCPINN(6, 4, hard_ic=True, ic_fn=ic_values))]:
        cm = BurgersClassicalPINN.matched_to(m)
        for tag, model in [(name, m), ("classical", cm)]:
            out = model(x, y, torch.zeros(8))
            u0, v0 = ic_values(x, y)
            err = (out[:, 0] - u0).abs().max() + (out[:, 1] - v0).abs().max()
            assert err < 1e-6, f"{tag} hard IC not exact at t=0: {err}"
            assert model(x, y, torch.ones(8)).abs().sum() > 0
        print("PASS hard-IC: exact IC at t=0 for quantum and classical")
