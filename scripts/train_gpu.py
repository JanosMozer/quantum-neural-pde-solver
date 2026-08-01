import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json, torch, numpy as np
from qt_pinn.learned_proj.qnn_generator import QuantumWeightGeneratorLP
from qt_pinn.qnn_generator import _circuit, N_QUBITS, W1_SIZE, W2_SIZE, W3_SIZE
from qt_pinn.config_loader import load as _load_cfg
from qt_pinn.pinn_target import TargetPINN
from pdes.burgers2d.physics_loss import compute_burgers_loss

from scripts.train import (
    SEED, N_COLLOC, N_BC, LAMBDA_BC, ADAPTIVE_LAMBDA, ALPHA,
    ADAPT_EVERY, ADAPT_WARMUP, LOG_EVERY,
    ADAM_LR, ADAM_STEPS, LBFGS_LR, LBFGS_STEPS, LBFGS_MAX_ITER,
    COSINE_ANNEAL, COSINE_ETA_MIN, WARMUP_STEPS,
    _next_run_dir, forward_losses, adaptive_lambda,
)

_t = _load_cfg()["training"]
WEIGHT_DECAY   = _t["adam"].get("weight_decay", 0.0)
RESAMPLE_EVERY = _t.get("resample_every", 0)
STRUCTURED_BC  = _t.get("structured_bc", False)
WEIGHT_REG     = _t.get("weight_reg", 0.0)
GRAD_CLIP_NORM = _t.get("grad_clip_norm", 1.0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPUWeightGeneratorLP(QuantumWeightGeneratorLP):
    """QuantumWeightGeneratorLP with the proj head moved to `device`.

    q_weights (the circuit's rotation angles) stays on CPU: PennyLane's
    default.qubit device initializes its statevector on CPU and errors if
    any input tensor lives elsewhere.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.proj.to(device)
        self._device = device

    def forward(self, inputs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if inputs is None:
            inputs = torch.zeros(N_QUBITS)
        probs = _circuit(inputs, self.q_weights).float().to(self._device)
        flat = self.proj(probs)
        return {
            "W1": flat[:W1_SIZE],
            "W2": flat[W1_SIZE: W1_SIZE + W2_SIZE],
            "W3": flat[W1_SIZE + W2_SIZE:],
        }


def make_colloc(n: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    x = torch.empty(n, device=device).uniform_(-1, 1).requires_grad_(True)
    y = torch.empty(n, device=device).uniform_(-1, 1).requires_grad_(True)
    t = torch.empty(n, device=device).uniform_(0, 1).requires_grad_(True)
    return x, y, t


def make_bc(n: int, device: torch.device, structured: bool = False) -> tuple[torch.Tensor, ...]:
    if structured:
        side = max(1, int(n ** 0.5))
        xs = torch.linspace(-1, 1, side, device=device)
        ys = torch.linspace(-1, 1, side, device=device)
        xg, yg = torch.meshgrid(xs, ys, indexing="ij")
        x = xg.flatten()[:n]
        y = yg.flatten()[:n]
    else:
        x = torch.empty(n, device=device).uniform_(-1, 1)
        y = torch.empty(n, device=device).uniform_(-1, 1)
    t = torch.zeros(len(x), device=device)
    u = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    v = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
    return x, y, t, u, v


def main_gpu() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_id, run_dir = _next_run_dir()
    run_dir.mkdir(parents=True)
    print(f"Run (GPU/LP): {run_id}  →  {run_dir}/  device={DEVICE}")

    model  = TargetPINN().to(DEVICE)
    gen    = GPUWeightGeneratorLP(DEVICE)
    params = list(gen.parameters())

    n_total = sum(p.numel() for p in gen.parameters())
    n_q     = gen.q_weights.numel()
    print(f"Params: {n_total:,}  (circuit={n_q}, proj={n_total - n_q})")

    x, y, t                      = make_colloc(N_COLLOC, DEVICE)
    x_bc, y_bc, t_bc, u_bc, v_bc  = make_bc(N_BC, DEVICE, structured=STRUCTURED_BC)

    lam = float(LAMBDA_BC)

    def _step(opt, step):
        nonlocal lam
        opt.zero_grad()
        weights = gen()
        pde, bc = compute_burgers_loss(model, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc, weights)
        if ADAPTIVE_LAMBDA and step >= ADAPT_WARMUP and step % ADAPT_EVERY == 0:
            lam = adaptive_lambda(pde, bc, params, lam, ALPHA)
        reg = sum(w.pow(2).mean() for w in weights.values()) if WEIGHT_REG > 0 else 0.0
        loss = pde + lam * bc + WEIGHT_REG * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
        opt.step()
        return loss.item(), pde.item(), bc.item()

    opt = torch.optim.Adam(params, lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
    if COSINE_ANNEAL:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=max(WARMUP_STEPS, 1))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(ADAM_STEPS - WARMUP_STEPS, 1), eta_min=COSINE_ETA_MIN)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[WARMUP_STEPS])
    else:
        sched = None
    lr_end = COSINE_ETA_MIN if COSINE_ANNEAL else ADAM_LR
    print(f"\nAdam  lr=0→{ADAM_LR}→{lr_end}  warmup={WARMUP_STEPS}  steps={ADAM_STEPS}  cosine={COSINE_ANNEAL}")
    print(f"{'step':>6}  {'total':>12}  {'pde':>12}  {'bc':>12}  {'λ':>8}  {'lr':>10}")
    for step in range(ADAM_STEPS):
        if RESAMPLE_EVERY > 0 and step % RESAMPLE_EVERY == 0 and step > 0:
            x, y, t = make_colloc(N_COLLOC, DEVICE)
            x_bc, y_bc, t_bc, u_bc, v_bc = make_bc(N_BC, DEVICE, structured=STRUCTURED_BC)
        total, pde, bc = _step(opt, step)
        if sched: sched.step()
        if step % LOG_EVERY == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"{step:6d}  {total:12.7f}  {pde:12.7f}  {bc:12.7f}  {lam:8.4f}  {lr_now:.2e}")

    # ── L-BFGS ───────────────────────────────────────────────────────────────
    # NOTE: torch.optim.LBFGS flattens all params' grads into one tensor for its
    # line search, which errors on mixed CPU/GPU params (q_weights is CPU).
    # Fine while lbfgs.steps=0 (current config); if you turn it on, this needs
    # a real fix (not just device placement) — flag it, don't silently hack it.
    if LBFGS_STEPS > 0:
        opt3    = torch.optim.LBFGS(params, lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
                                     history_size=10, line_search_fn="strong_wolfe")
        counter = [0]
        print(f"\nL-BFGS  lr={LBFGS_LR}  steps={LBFGS_STEPS}")
        print(f"{'closure':>7}  {'total':>12}  {'pde':>12}  {'bc':>12}")

        def closure():
            nonlocal lam
            opt3.zero_grad()
            pde, bc = forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc)
            loss = pde + lam * bc
            loss.backward()
            if counter[0] % (LOG_EVERY * 2) == 0:
                print(f"{counter[0]:7d}  {loss.item():12.7f}  {pde.item():12.7f}  {bc.item():12.7f}")
            counter[0] += 1
            return loss

        for _ in range(LBFGS_STEPS):
            opt3.step(closure)

    # ── Final eval ───────────────────────────────────────────────────────────
    pde_f, bc_f = forward_losses(model, gen, x, y, t, x_bc, y_bc, t_bc, u_bc, v_bc)
    pde_final = pde_f.item(); bc_final = bc_f.item()
    print(f"\nFinal  pde={pde_final:.7f}  bc={bc_final:.7f}  sum={pde_final+bc_final:.7f}")

    # ── Holdout eval: fresh points, seed offset +90000 (never seen in training) ─
    torch.manual_seed(SEED + 90000)
    xh, yh, th = make_colloc(N_COLLOC, DEVICE)
    xh_bc, yh_bc, th_bc, uh_bc, vh_bc = make_bc(N_BC, DEVICE, structured=STRUCTURED_BC)
    pde_h, bc_h = forward_losses(model, gen, xh, yh, th, xh_bc, yh_bc, th_bc, uh_bc, vh_bc)
    pde_hold = pde_h.item(); bc_hold = bc_h.item()
    print(f"Holdout pde={pde_hold:.7f}  bc={bc_hold:.7f}  sum={pde_hold+bc_hold:.7f}  "
          f"(pde ratio={pde_hold/max(pde_final,1e-12):.2f}x)")

    # ── Save ─────────────────────────────────────────────────────────────────
    torch.save(gen.state_dict(), run_dir / "q_weights.pt")
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "generator": "learned_proj_gpu", "device": str(DEVICE),
        "seed": SEED, "n_colloc": N_COLLOC, "n_bc": N_BC,
        "adam_lr": ADAM_LR, "adam_steps": ADAM_STEPS,
        "lambda_bc_init": LAMBDA_BC, "lambda_bc_final": round(lam, 4),
        "cosine": COSINE_ANNEAL, "weight_decay": WEIGHT_DECAY,
        "bottleneck_width": gen.proj[0].out_features,
    }, indent=2))
    (run_dir / "results.json").write_text(json.dumps({
        "run_id": run_id,
        "pde_loss": round(pde_final, 8),
        "bc_loss":  round(bc_final, 8),
        "total":    round(pde_final + bc_final, 8),
        "holdout_pde_loss": round(pde_hold, 8),
        "holdout_bc_loss":  round(bc_hold, 8),
        "holdout_total":    round(pde_hold + bc_hold, 8),
        "pde_ratio": round(pde_hold / max(pde_final, 1e-12), 4),
        "lambda_final": round(lam, 6),
        "n_params": sum(p.numel() for p in gen.parameters()),
    }, indent=2))
    print(f"Saved  {run_dir}/q_weights.pt")
    print(f"       {run_dir}/results.json")


if __name__ == "__main__":
    main_gpu()
