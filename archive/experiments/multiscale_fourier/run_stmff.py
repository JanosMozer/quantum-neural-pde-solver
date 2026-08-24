"""Cheap falsification test: does splitting (x,y) and t into separate Fourier
embeddings (Wang, Wang, Perdikaris arXiv:2012.10047's ST-MFF, Mx=Mt=1 case) beat
our current single joint (x,y,t) embedding, and does it need the adaptive-lambda
pairing their own ablation showed was necessary (see research/logs/
2026-08-01-multiscale-fourier-features.md for the full source-verified writeup)?

Mx=Mt=1 (one scale per axis, not per-axis multi-resolution) is parameter-count
neutral: W1/W2 are reused for both branches unchanged, W3's input stays H2-wide
since the branches are combined by element-wise product, not concatenation. So
this isolates the "does splitting the axis help" question from any budget
confound -- same TOTAL_WEIGHTS as the current architecture (1346).

Uses DirectGenerator (uncompressed, matches Step 2 of the session's established
practice: test architecture changes on the cheapest generator first).

Usage: python run_stmff.py [--split] [--adaptive] [--steps N] [--sigma_xy F] [--sigma_t F]
"""

import sys
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2_ablation"))

import torch
import torch.nn.functional as F
from qt_pinn.qnn_generator import IN_DIM, H1, H2, OUT_DIM, TOTAL_WEIGHTS
from pdes.burgers2d.physics_loss import compute_burgers_loss
from run_ablation import DirectGenerator

N_FREQ = IN_DIM // 2  # 3, matches the current joint Fourier map's frequency count


class SplitAxisFourierMap(torch.nn.Module):
    """Separate Fourier embeddings for (x,y) and t, each producing IN_DIM features
    (so W1 can be shared, unchanged, across both branches)."""

    def __init__(self, sigma_xy: float, sigma_t: float, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("B_xy", torch.randn(2, N_FREQ) * sigma_xy)
        self.register_buffer("B_t", torch.randn(1, N_FREQ) * sigma_t)

    def embed_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        proj = torch.stack([x, y], dim=-1) @ self.B_xy
        return torch.cat([torch.sin(2 * torch.pi * proj), torch.cos(2 * torch.pi * proj)], dim=-1)

    def embed_t(self, t: torch.Tensor) -> torch.Tensor:
        proj = t.unsqueeze(-1) @ self.B_t
        return torch.cat([torch.sin(2 * torch.pi * proj), torch.cos(2 * torch.pi * proj)], dim=-1)


class JointFourierMap(torch.nn.Module):
    """Baseline: today's single joint (x,y,t) embedding, reimplemented locally
    so the baseline run in this script doesn't depend on config.yaml's fourier
    section (keeps both arms of the comparison controlled by the same --sigma_xy
    flag, not two different config sources)."""

    def __init__(self, sigma: float, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("B", torch.randn(3, N_FREQ) * sigma)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        proj = torch.stack([x, y, t], dim=-1) @ self.B
        return torch.cat([torch.sin(2 * torch.pi * proj), torch.cos(2 * torch.pi * proj)], dim=-1)


class SplitAxisPINN(torch.nn.Module):
    """ST-MFF (Mx=Mt=1): x,y and t embedded separately, run through the SAME
    shared hidden layers, combined by element-wise product before the output layer."""

    def __init__(self, sigma_xy: float, sigma_t: float) -> None:
        super().__init__()
        self.fourier = SplitAxisFourierMap(sigma_xy, sigma_t)

    def _unpack(self, weights):
        w1 = weights["W1"][:IN_DIM * H1].reshape(H1, IN_DIM)
        b1 = weights["W1"][IN_DIM * H1:]
        w2 = weights["W2"][:H1 * H2].reshape(H2, H1)
        b2 = weights["W2"][H1 * H2:]
        w3 = weights["W3"][:H2 * OUT_DIM].reshape(OUT_DIM, H2)
        b3 = weights["W3"][H2 * OUT_DIM:]
        return w1, b1, w2, b2, w3, b3

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, weights) -> torch.Tensor:
        w1, b1, w2, b2, w3, b3 = self._unpack(weights)

        feats_xy = self.fourier.embed_xy(x, y)
        feats_t = self.fourier.embed_t(t)

        h_xy = torch.tanh(F.linear(feats_xy, w1, b1))
        h_t = torch.tanh(F.linear(feats_t, w1, b1))
        h_xy = torch.tanh(F.linear(h_xy, w2, b2))
        h_t = torch.tanh(F.linear(h_t, w2, b2))
        combined = h_xy * h_t  # element-wise product, per the paper's ST-MFF formula
        return F.linear(combined, w3, b3)


class JointPINN(torch.nn.Module):
    """Baseline: same unpack logic, single joint embedding, standard 2-layer forward."""

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.fourier = JointFourierMap(sigma)

    def _unpack(self, weights):
        w1 = weights["W1"][:IN_DIM * H1].reshape(H1, IN_DIM)
        b1 = weights["W1"][IN_DIM * H1:]
        w2 = weights["W2"][:H1 * H2].reshape(H2, H1)
        b2 = weights["W2"][H1 * H2:]
        w3 = weights["W3"][:H2 * OUT_DIM].reshape(OUT_DIM, H2)
        b3 = weights["W3"][H2 * OUT_DIM:]
        return w1, b1, w2, b2, w3, b3

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, weights) -> torch.Tensor:
        w1, b1, w2, b2, w3, b3 = self._unpack(weights)
        feats = self.fourier(x, y, t)
        h = torch.tanh(F.linear(feats, w1, b1))
        h = torch.tanh(F.linear(h, w2, b2))
        return F.linear(h, w3, b3)


def make_colloc(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n).uniform_(-1, 1, generator=g).requires_grad_(True)
    y = torch.empty(n).uniform_(-1, 1, generator=g).requires_grad_(True)
    t = torch.empty(n).uniform_(0, 1, generator=g).requires_grad_(True)
    return x, y, t


def make_bc(n: int, seed: int):
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.empty(n).uniform_(-1, 1, generator=g)
    y = torch.empty(n).uniform_(-1, 1, generator=g)
    t = torch.zeros(n)
    u = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    v = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
    return x, y, t, u, v


def adaptive_lambda(pde, bc, params, current, alpha=0.9, lambda_max=100.0):
    """arXiv:2001.04536 Algorithm 1, same formula our own scripts/train.py uses."""
    g_pde = torch.autograd.grad(pde, params, retain_graph=True, allow_unused=True)
    g_bc = torch.autograd.grad(bc, params, retain_graph=True, allow_unused=True)
    max_pde = max(g.abs().max() for g in g_pde if g is not None)
    mean_bc = torch.cat([g.flatten() for g in g_bc if g is not None]).abs().mean()
    lam_hat = min((max_pde / (mean_bc + 1e-8)).item(), lambda_max)
    return (1 - alpha) * current + alpha * lam_hat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="store_true", help="use ST-MFF split-axis embedding, else joint baseline")
    parser.add_argument("--adaptive", action="store_true", help="enable adaptive_lambda (arXiv:2001.04536)")
    parser.add_argument("--lambda_max", type=float, default=100.0,
                         help="cap on adaptive lambda; default matches train.py's generic default, "
                              "override to bracket the problem's known-good fixed-lambda range")
    parser.add_argument("--sigma_xy", type=float, default=0.1, help="spatial sigma (or joint sigma if not --split)")
    parser.add_argument("--sigma_t", type=float, default=1.0, help="temporal sigma, only used with --split")
    parser.add_argument("--steps", type=int, default=4000, help="reduced from the full 18000 for a cheap falsification test")
    parser.add_argument("--n_colloc", type=int, default=4096)
    parser.add_argument("--n_bc", type=int, default=4096)
    parser.add_argument("--lambda_bc", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout_n", type=int, default=4096)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = SplitAxisPINN(args.sigma_xy, args.sigma_t) if args.split else JointPINN(args.sigma_xy)
    gen = DirectGenerator()
    params = list(gen.parameters())
    n_params = sum(p.numel() for p in params)
    assert n_params == TOTAL_WEIGHTS, f"expected {TOTAL_WEIGHTS} params, got {n_params}"

    x, y, t = make_colloc(args.n_colloc, args.seed)
    xb, yb, tb, ub, vb = make_bc(args.n_bc, args.seed)

    opt = torch.optim.Adam(params, lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)

    label = f"{'split' if args.split else 'joint'}{'+adaptive' if args.adaptive else ''}"
    print(f"[{label}] sigma_xy={args.sigma_xy} sigma_t={args.sigma_t if args.split else 'n/a'} "
          f"n_params={n_params} steps={args.steps}")

    lam = float(args.lambda_bc)
    for step in range(args.steps):
        opt.zero_grad()
        w = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
        if args.adaptive and step >= 200 and step % 1 == 0:
            lam = adaptive_lambda(pde, bc, params, lam, lambda_max=args.lambda_max)
        (pde + lam * bc).backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"{step:6d}  pde={pde.item():.6f}  bc={bc.item():.6f}  lam={lam:.4f}")

    pde_f, bc_f = pde.item(), bc.item()
    print(f"Final    pde={pde_f:.7f}  bc={bc_f:.7f}  total={pde_f+bc_f:.7f}  lambda_final={lam:.4f}")

    xh, yh, th = make_colloc(args.holdout_n, args.seed + 90000)
    xhb, yhb, thb, uhb, vhb = make_bc(args.holdout_n, args.seed + 90000)
    pde_h, bc_h = compute_burgers_loss(model, xh, yh, th, xhb, yhb, thb, uhb, vhb, gen())
    print(f"Holdout  pde={pde_h.item():.7f}  bc={bc_h.item():.7f}  total={pde_h.item()+bc_h.item():.7f}")

    result = {
        "label": label, "split": args.split, "adaptive": args.adaptive,
        "sigma_xy": args.sigma_xy, "sigma_t": args.sigma_t if args.split else None,
        "lambda_max": args.lambda_max if args.adaptive else None,
        "n_params": n_params, "steps": args.steps, "lambda_final": round(lam, 4),
        "pde_loss": round(pde_f, 8), "bc_loss": round(bc_f, 8), "total": round(pde_f + bc_f, 8),
        "holdout_pde_loss": round(pde_h.item(), 8), "holdout_bc_loss": round(bc_h.item(), 8),
        "holdout_total": round(pde_h.item() + bc_h.item(), 8),
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    lm_suffix = f"_lmax{args.lambda_max:.0f}" if args.adaptive else ""
    out_path = out_dir / f"{label}_seed{args.seed}{lm_suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
