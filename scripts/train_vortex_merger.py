"""Vortex-merger demo: DNS reference + classical / QT PINNs (max GPU).

Four same-sign Gaussians → one central swirl (no closed form).

  .venv/bin/python scripts/train_vortex_merger.py --stage dns
  .venv/bin/python scripts/train_vortex_merger.py --stage train --model classical
  .venv/bin/python scripts/train_vortex_merger.py --stage train --model quantum
  .venv/bin/python scripts/train_vortex_merger.py --stage all
  .venv/bin/python scripts/train_vortex_merger.py --stage plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pdes.ns2d.physics_loss import _grad, relative_l2, relative_l2_gauge
from qt_pinn.fourier import FourierFeatureMapTGV, FourierFeatureMapWide
from qt_pinn.ns2d_spectral import (
    TWO_PI,
    ic_values_from_dns,
    sample_dns,
    simulate,
)
from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.qnn_generator_cond import ConditionedQuantumGeneratorV2
from qt_pinn.tgv_demo import resolve_device, write_run

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "blog" / "checkpoint" / "v3"


class MergerMLP(nn.Module):
    """Classical PINN with wide RFF (soft IC; DNS data carries the IC)."""

    def __init__(self, hidden=(128, 128), t_max: float = 40.0, n_freqs: int = 48):
        super().__init__()
        self.fourier = FourierFeatureMapWide(n_freqs=n_freqs, sigma=1.2, t_max=t_max, seed=0)
        h1, h2 = hidden
        d = self.fourier.out_dim
        self.net = nn.Sequential(
            nn.Linear(d, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, 3),
        )

    def forward(self, x, y, t):
        return self.net(self.fourier(torch.stack([x, y, t], dim=-1)))



def pde_loss(uvp, x, y, t, nu: float):
    u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]
    u_t = _grad(u, t); u_x = _grad(u, x); u_y = _grad(u, y)
    v_t = _grad(v, t); v_x = _grad(v, x); v_y = _grad(v, y)
    p_x = _grad(p, x); p_y = _grad(p, y)
    u_xx = _grad(u_x, x); u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x); v_yy = _grad(v_y, y)
    f_c = u_x + v_y
    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return f_c.pow(2).mean() + f_u.pow(2).mean() + f_v.pow(2).mean()


def data_loss(uvp, tgt):
    return (uvp - tgt).pow(2).mean()


@torch.no_grad()
def eval_vs_dns(predict_fn, dns, device, n_grid: int = 96, n_times: int = 5) -> dict:
    xs = dns["x"].to(device)
    n = xs.numel()
    # subsample grid
    step = max(1, n // n_grid)
    xs_s = xs[::step]
    xg, yg = torch.meshgrid(xs_s, xs_s, indexing="ij")
    x, y = xg.flatten(), yg.flatten()
    t_idx = torch.linspace(0, len(dns["t"]) - 1, n_times).long()
    out = {}
    vel_max = 0.0
    for ti in t_idx.tolist():
        tv = float(dns["t"][ti])
        t = torch.full_like(x, tv)
        pred = predict_fn(x, y, t)
        u_ex = dns["u"][ti].to(device)[::step, ::step].reshape(-1)
        v_ex = dns["v"][ti].to(device)[::step, ::step].reshape(-1)
        p_ex = dns["p"][ti].to(device)[::step, ::step].reshape(-1)
        u_e = relative_l2(pred[:, 0], u_ex)
        v_e = relative_l2(pred[:, 1], v_ex)
        vel = 0.5 * (u_e + v_e)
        vel_max = max(vel_max, vel)
        out[f"t{tv:.1f}"] = {
            "t": tv, "u": u_e, "v": v_e,
            "p": relative_l2_gauge(pred[:, 2], p_ex), "vel": vel,
        }
    out["vel_rel_l2_max"] = vel_max
    return out


def run_dns(args) -> Path:
    V3.mkdir(parents=True, exist_ok=True)
    out = V3 / "dns"
    out.mkdir(exist_ok=True)
    print(f"DNS on {args.device}  n={args.dns_n}  T={args.t_max}  ν={args.nu}")
    t0 = time.time()
    dns = simulate(
        n=args.dns_n, nu=args.nu, t_max=args.t_max, n_save=args.n_save,
        gamma=args.gamma, delta=args.delta, pull_in=args.pull_in,
        device=str(args.device), cfl=0.4,
    )
    path = out / "reference.pt"
    torch.save(dns, path)
    meta = {
        "nu": args.nu, "t_max": args.t_max, "n": args.dns_n,
        "n_save": args.n_save, "gamma": args.gamma, "delta": args.delta,
        "pull_in": args.pull_in, "elapsed_s": round(time.time() - t0, 1),
        "centers": dns["centers"],
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"DNS done in {time.time()-t0:.1f}s → {path}")
    return path


def train_one(args, model_kind: str, dns: dict) -> Path:
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Keep DNS grids on GPU for fast sampling (big win on 5090)
    dns_g = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()
    }

    run_dir = V3 / model_kind
    if run_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    nu = float(dns["nu"])
    t_max = float(dns["t_max"])
    n_colloc = args.n_colloc if model_kind == "classical" else min(args.n_colloc, args.qt_colloc)

    if model_kind == "classical":
        hidden = tuple(args.classical_hidden)
        model = MergerMLP(hidden=hidden, t_max=t_max, n_freqs=args.n_freqs).to(device)
        gen = None
        params = list(model.parameters())
        n_deployed = sum(p.numel() for p in params)

        def predict(x, y, t):
            return model(x, y, t)
    else:
        hidden = tuple(args.qt_hidden)
        target = TargetPINNNS(
            fourier="tgv", hard_ic=False, hidden=hidden, t_max=t_max,
        ).to(device)
        gen = ConditionedQuantumGeneratorV2(
            in_dim=target.in_dim, h1=hidden[0], h2=hidden[1],
            n_qubits=args.n_qubits, n_layers=args.n_layers,
            bottleneck_width=args.bottleneck, nu_range=(0.001, 0.05),
            nu_encode="log",
        ).to(device)
        params = list(gen.parameters())
        n_deployed = int(gen.total_weights)
        model = target

        def predict(x, y, t):
            w = gen(torch.tensor([nu], device=device, dtype=torch.float32))
            w = {k: v[0] for k, v in w.items()}
            return target(x, y, t, w)

    n_train = sum(p.numel() for p in params)
    print(f"\n=== train {model_kind}  deployed={n_deployed:,}  train_params={n_train:,}  "
          f"colloc={n_colloc}  steps={args.adam_steps} ===")

    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.adam_steps, eta_min=1e-5)

    if model_kind == "classical" and args.compile and args.lambda_pde <= 0:
        try:
            model.net = torch.compile(model.net)
            print("torch.compile enabled on classical MLP")
        except Exception as e:
            print(f"compile skipped: {e}")
    elif model_kind == "classical" and args.compile:
        print("torch.compile skipped (PDE needs double backward)")

    history = []
    best = float("inf")
    t0 = time.time()
    n_ic = max(1024, n_colloc // 8)
    for step in range(args.adam_steps):
        if args.budget_s and (time.time() - t0) > args.budget_s:
            print(f"budget stop at step {step}")
            break
        x, y, t, tgt, _omega = sample_dns(dns_g, n_colloc, device, t_sample=args.t_sample)
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)

        x_ic = torch.empty(n_ic, device=device).uniform_(0, TWO_PI)
        y_ic = torch.empty(n_ic, device=device).uniform_(0, TWO_PI)
        t_ic = torch.zeros(n_ic, device=device)
        u0, v0, p0 = ic_values_from_dns(dns_g, x_ic, y_ic)
        tgt_ic = torch.stack([u0, v0, p0], dim=-1)

        opt.zero_grad(set_to_none=True)
        uvp = predict(x, y, t)
        dat = data_loss(uvp, tgt)
        ic = data_loss(predict(x_ic, y_ic, t_ic), tgt_ic)
        pde = pde_loss(uvp, x, y, t, nu) if args.lambda_pde > 0 else torch.zeros((), device=device)
        loss = args.lambda_data * dat + args.lambda_pde * pde + args.lambda_ic * ic
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == args.adam_steps - 1:
            with torch.no_grad():
                errs = eval_vs_dns(predict, dns_g, device, n_grid=64, n_times=5)
            vel = errs["vel_rel_l2_max"]
            print(f"{step:6d}  loss={loss.item():.5f}  data={dat.item():.5f}  "
                  f"ic={ic.item():.5f}  pde={pde.item():.5f}  "
                  f"vel%={100*vel:.3f}  [{time.time()-t0:.0f}s]")
            history.append({
                "step": step, "loss": loss.item(), "data": dat.item(),
                "ic": float(ic.item()), "pde": float(pde.item()), "vel_max": vel,
                "elapsed_s": round(time.time() - t0, 1),
            })
            if vel < best:
                best = vel
                if gen is None:
                    torch.save(model.state_dict(), run_dir / "model.pt")
                else:
                    torch.save(gen.state_dict(), run_dir / "generator.pt")
                    with torch.no_grad():
                        w = gen(torch.tensor([nu], device=device))
                        torch.save({k: v[0].cpu() for k, v in w.items()},
                                   run_dir / "deployed_weights.pt")

    elapsed = time.time() - t0
    if gen is None and (run_dir / "model.pt").exists():
        try:
            model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
        except Exception:
            pass
    elif gen is not None and (run_dir / "generator.pt").exists():
        gen.load_state_dict(torch.load(run_dir / "generator.pt", map_location=device, weights_only=True))

    with torch.no_grad():
        errs = eval_vs_dns(predict, dns_g, device, n_grid=128, n_times=9)
    vmax = errs["vel_rel_l2_max"]
    print(f"BEST vel_max={100*vmax:.3f}%  elapsed={elapsed:.0f}s")

    cfg = {
        "model": model_kind, "nu": nu, "t_max": t_max,
        "classical_hidden": list(args.classical_hidden),
        "qt_hidden": list(args.qt_hidden),
        "n_freqs": args.n_freqs,
        "n_qubits": args.n_qubits, "n_layers": args.n_layers,
        "bottleneck_width": args.bottleneck,
        "adam_steps": args.adam_steps, "n_colloc": args.n_colloc,
        "lambda_data": args.lambda_data, "lambda_pde": args.lambda_pde,
        "lambda_ic": args.lambda_ic,
        "lr": args.lr, "t_sample": args.t_sample, "hard_ic": False,
        "n_deployed_params": n_deployed, "n_train_params": n_train,
        "dns": "blog/checkpoint/v3/dns/reference.pt",
    }
    results = {
        "model": model_kind,
        "vel_rel_l2_max": vmax,
        "gate_pass": vmax <= 0.05,
        "elapsed_s": round(elapsed, 1),
        "n_deployed_params": n_deployed,
        "exact_l2": {k: v for k, v in errs.items() if k != "vel_rel_l2_max"},
        "history": history,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved → {run_dir}")
    return run_dir


def make_plots(dns: dict, device: torch.device) -> None:
    media = V3 / "media"
    media.mkdir(parents=True, exist_ok=True)

    # load models
    cfg_c = json.loads((V3 / "classical" / "config.json").read_text())
    model_c = MergerMLP(
        hidden=tuple(cfg_c["classical_hidden"]),
        t_max=dns["t_max"],
        n_freqs=cfg_c.get("n_freqs", 48),
    ).to(device)
    sd = torch.load(V3 / "classical" / "model.pt", map_location=device, weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model_c.load_state_dict(sd, strict=False)
    model_c.eval()

    cfg_q = json.loads((V3 / "quantum" / "config.json").read_text())
    h = tuple(cfg_q["qt_hidden"])
    target = TargetPINNNS(
        fourier="tgv", hard_ic=False, hidden=h, t_max=dns["t_max"],
    ).to(device)
    w = torch.load(V3 / "quantum" / "deployed_weights.pt", map_location=device, weights_only=True)

    def pred_c(x, y, t):
        return model_c(x, y, t)

    def pred_q(x, y, t):
        return target(x, y, t, w)

    xs = dns["x"].cpu().numpy()
    n = len(xs)
    step = max(1, n // 128)
    xs_s = xs[::step]
    t_idx = [0, len(dns["t"]) // 4, len(dns["t"]) // 2, 3 * len(dns["t"]) // 4, len(dns["t"]) - 1]

    # snapshots
    fig, axes = plt.subplots(len(t_idx), 3, figsize=(10, 12), constrained_layout=True)
    for row, ti in enumerate(t_idx):
        omega_ex = dns["omega"][ti].numpy()[::step, ::step]
        tv = float(dns["t"][ti])
        xg, yg = np.meshgrid(xs_s, xs_s, indexing="ij")
        xt = torch.tensor(xg.ravel(), device=device, dtype=torch.float32)
        yt = torch.tensor(yg.ravel(), device=device, dtype=torch.float32)
        tt = torch.full_like(xt, tv)
        with torch.no_grad():
            pc = pred_c(xt, yt, tt).cpu().numpy().reshape(*xg.shape, 3)
            pq = pred_q(xt, yt, tt).cpu().numpy().reshape(*xg.shape, 3)
        # vorticity FD
        dx = TWO_PI / xg.shape[0]

        def vort(u, v):
            dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx)
            dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
            return dvdx - dudy

        panels = [omega_ex, vort(pc[..., 0], pc[..., 1]), vort(pq[..., 0], pq[..., 1])]
        vmax = max(abs(omega_ex).max(), 1e-6)
        for col, (arr, title) in enumerate(zip(panels, ["DNS", "Classical", "Quantum"])):
            ax = axes[row, col]
            im = ax.pcolormesh(xs_s, xs_s, arr.T, cmap="RdBu_r", shading="auto",
                               vmin=-vmax, vmax=vmax)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"t={tv:.1f}")
        fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.046, pad=0.02)
    fig.suptitle("Vortex merger: 4 same-sign → 1 swirl", fontsize=13)
    fig.savefig(media / "merger_triplet_snapshots.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {media/'merger_triplet_snapshots.png'}")

    # Gif via no-tracer campaign helper when checkpoints exist
    try:
        from scripts.exp_merger_omega_v2 import regenerate_media_flexible
        regenerate_media_flexible(dns, device, "uvpw")
        print(f"Wrote {media/'merger_triplet.gif'} (no tracers)")
    except Exception as exc:
        print(f"Skipping gif (need promoted v3 checkpoints): {exc}")
    return


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["dns", "train", "plot", "all"], default="all")
    p.add_argument("--model", choices=["classical", "quantum", "both"], default="both")
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    # DNS
    p.add_argument("--dns-n", type=int, default=256)
    p.add_argument("--nu", type=float, default=0.005)
    p.add_argument("--t-max", type=float, default=40.0)
    p.add_argument("--n-save", type=int, default=81)
    p.add_argument("--gamma", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=0.65)
    p.add_argument("--pull-in", type=float, default=0.95)
    # train — sized for RTX 5090
    p.add_argument("--adam-steps", type=int, default=20000)
    p.add_argument("--n-colloc", type=int, default=24576)
    p.add_argument("--qt-colloc", type=int, default=8192,
                   help="collocation points for quantum (circuit once/step; PDE graph is heavy)")
    p.add_argument("--classical-hidden", type=int, nargs=2, default=[128, 128])
    p.add_argument("--qt-hidden", type=int, nargs=2, default=[32, 32])
    p.add_argument("--n-freqs", type=int, default=48)
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--lambda-data", type=float, default=50.0)
    p.add_argument("--lambda-pde", type=float, default=0.2)
    p.add_argument("--lambda-ic", type=float, default=20.0)
    p.add_argument("--t-sample", choices=["uniform", "tail"], default="tail")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--budget-s", type=float, default=0.0)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    print(f"device={args.device}  cuda_avail={torch.cuda.is_available()}")
    if args.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    dns_path = V3 / "dns" / "reference.pt"

    if args.stage in ("dns", "all"):
        run_dns(args)

    if args.stage in ("train", "all"):
        if not dns_path.exists():
            raise FileNotFoundError(f"missing {dns_path}; run --stage dns first")
        dns = torch.load(dns_path, map_location="cpu", weights_only=False)
        models = ["classical", "quantum"] if args.model == "both" else [args.model]
        for m in models:
            train_one(args, m, dns)

    if args.stage in ("plot", "all"):
        if not dns_path.exists():
            raise FileNotFoundError(dns_path)
        dns = torch.load(dns_path, map_location="cpu", weights_only=False)
        make_plots(dns, args.device)

    # update root README
    readme = ROOT / "blog" / "checkpoint" / "README.md"
    readme.write_text(
        "# Blog checkpoints\n\n"
        "| Version | Content |\n"
        "|---------|---------|\n"
        "| `v1/` | TGV demo gate-pass (~0.95% / ~1.36%) |\n"
        "| `v2/` | TGV polish (~0.61% / ~0.62%) + `media/` |\n"
        "| `v3/` | Vortex merger DNS + classical/QT + `media/` |\n"
    )
    print("Done.")


if __name__ == "__main__":
    main()
