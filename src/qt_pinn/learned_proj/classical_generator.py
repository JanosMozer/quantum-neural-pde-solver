"""Classical analog of QuantumWeightGeneratorLP.

Replaces the quantum circuit with a trainable 512-dim latent vector
(softmax-normalised to match the probability output of the circuit).
Architecture is otherwise IDENTICAL — same bottleneck projection.

Fair ablation: if this trains equally well, the quantum circuit is redundant.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

import torch
import torch.nn as nn
from qt_pinn.qnn_generator import N_STATES, W1_SIZE, W2_SIZE, W3_SIZE, TOTAL_WEIGHTS


class ClassicalWeightGenerator(nn.Module):
    """Trainable latent vector → same bottleneck projection as LP generator.

    Params: 512 (latent) + ~33K (proj) ≈ 33.5K total
    vs quantum LP:    81 (circuit) + ~33K (proj) ≈ 33K total
    """

    def __init__(self) -> None:
        super().__init__()
        self.latent = nn.Parameter(torch.randn(N_STATES))   # replaces circuit output
        self.proj = nn.Sequential(
            nn.Linear(N_STATES, 64),
            nn.Tanh(),
            nn.Linear(64, TOTAL_WEIGHTS),
        )

    def forward(self, inputs=None) -> dict[str, torch.Tensor]:
        probs = torch.softmax(self.latent, dim=0)   # normalise like circuit probabilities
        flat  = self.proj(probs)
        return {
            "W1": flat[:W1_SIZE],
            "W2": flat[W1_SIZE: W1_SIZE + W2_SIZE],
            "W3": flat[W1_SIZE + W2_SIZE:],
        }


if __name__ == "__main__":
    gen = ClassicalWeightGenerator()
    w   = gen()
    n   = sum(p.numel() for p in gen.parameters())
    print(f"PASS  W1={w['W1'].shape} W2={w['W2'].shape} W3={w['W3'].shape}")
    print(f"Params: {n:,}  (latent={gen.latent.numel()}, proj={n - gen.latent.numel()})")
