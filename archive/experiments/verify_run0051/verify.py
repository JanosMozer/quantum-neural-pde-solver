import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch
from qt_pinn.pinn_target import TargetPINN
from qt_pinn.learned_proj.qnn_generator import QuantumWeightGeneratorLP
from pdes.burgers2d.physics_loss import compute_burgers_loss


def make_colloc(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n).uniform_(-1, 1, generator=g).requires_grad_(True)
    y = torch.empty(n).uniform_(-1, 1, generator=g).requires_grad_(True)
    t = torch.empty(n).uniform_(0, 1, generator=g).requires_grad_(True)
    return x, y, t


def make_bc(n, seed):
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.empty(n).uniform_(-1, 1, generator=g)
    y = torch.empty(n).uniform_(-1, 1, generator=g)
    t = torch.zeros(n)
    u = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    v = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
    return x, y, t, u, v


N_COLLOC, N_BC = 256, 64
LAMBDA_BC = 7.0
WEIGHT_DECAY = 0.001
STEPS = 18000
SEED = 0

torch.manual_seed(SEED)
model = TargetPINN()
gen = QuantumWeightGeneratorLP()
params = list(gen.parameters())
n_params = sum(p.numel() for p in params)
print(f"n_params={n_params}  (expect 60083, matching run_0051's reported count)")

x, y, t = make_colloc(N_COLLOC, SEED)
xb, yb, tb, ub, vb = make_bc(N_BC, SEED)

opt = torch.optim.Adam(params, lr=0.01, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-5)

t0 = time.perf_counter()
for step in range(STEPS):
    opt.zero_grad()
    w = gen()
    pde, bc = compute_burgers_loss(model, x, y, t, xb, yb, tb, ub, vb, w)
    (pde + LAMBDA_BC * bc).backward()
    opt.step()
    sched.step()
    if step % 2000 == 0:
        print(f"{step:6d}  pde={pde.item():.6f}  bc={bc.item():.6f}  total={pde.item()+bc.item():.6f}")
elapsed = time.perf_counter() - t0

print(f"\nTrain (matches run_0051's own set)  pde={pde.item():.7f}  bc={bc.item():.7f}  "
      f"sum={pde.item()+bc.item():.7f}  ({elapsed:.1f}s)")

xh, yh, th = make_colloc(4096, SEED + 90000)
xhb, yhb, thb, uhb, vhb = make_bc(4096, SEED + 90000)
pde_h, bc_h = compute_burgers_loss(model, xh, yh, th, xhb, yhb, thb, uhb, vhb, gen())
print(f"Held-out (n=4096, never trained on)  pde={pde_h.item():.7f}  bc={bc_h.item():.7f}  "
      f"sum={pde_h.item()+bc_h.item():.7f}")

result = {
    "n_params": n_params, "n_colloc": N_COLLOC, "n_bc": N_BC, "lambda_bc": LAMBDA_BC,
    "weight_decay": WEIGHT_DECAY, "steps": STEPS, "seed": SEED,
    "train_pde": round(pde.item(), 8), "train_bc": round(bc.item(), 8),
    "train_total": round(pde.item() + bc.item(), 8),
    "holdout_pde": round(pde_h.item(), 8), "holdout_bc": round(bc_h.item(), 8),
    "holdout_total": round(pde_h.item() + bc_h.item(), 8),
    "pde_ratio": round(pde_h.item() / pde.item(), 2) if pde.item() > 0 else None,
    "elapsed_s": round(elapsed, 1),
}
out = Path(__file__).resolve().parent / "result.json"
out.write_text(json.dumps(result, indent=2))
print(f"Saved {out}")
