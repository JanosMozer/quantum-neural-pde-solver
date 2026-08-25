"""Benchmark promoted vortex-merger classical + quantum models.

Reports latency, DNS accuracy (vel / FD-curl ω), data MSE, and PDE residual MSE.

  .venv/bin/python scripts/bench_merger.py
  .venv/bin/python scripts/bench_merger.py --out blog/checkpoint/v4/bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pdes.ns2d.physics_loss import _grad, relative_l2, relative_l2_gauge
from qt_pinn.ns2d_spectral import TWO_PI
from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.tgv_demo import resolve_device
from scripts.exp_merger_omega import GATE_TIMES, HarmMLP, OMEGA_GATE, V3, eval_fields


def load_classical(device):
    cfg = json.loads((V3 / "classical" / "config.json").read_text())
    model = HarmMLP(
        hidden=tuple(cfg.get("hidden", [96, 96])),
        t_max=float(cfg.get("t_max", 40.0)),
        k_max=int(cfg.get("k_max", 6)),
        axis_extra=int(cfg.get("axis_extra", 0)),
    ).to(device)
    model.load_state_dict(torch.load(V3 / "classical" / "model.pt", map_location=device, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, cfg, n_params


def load_quantum(device):
    cfg = json.loads((V3 / "quantum" / "config.json").read_text())
    target = TargetPINNNS(
        fourier=cfg.get("qt_fourier", "harm"),
        hard_ic=False,
        hidden=tuple(cfg.get("qt_hidden", [96, 96])),
        t_max=float(cfg.get("t_max", 40.0)),
        n_freqs=int(cfg.get("n_freqs", 6)),
    ).to(device)
    w = torch.load(V3 / "quantum" / "deployed_weights.pt", map_location=device, weights_only=True)
    n_params = int(cfg.get("n_deployed_params") or sum(v.numel() for v in w.values()))
    return target, w, cfg, n_params


@torch.no_grad()
def latency_ms(predict, device, n_pts=65536, warmup=20, reps=100) -> dict:
    x = torch.rand(n_pts, device=device) * TWO_PI
    y = torch.rand(n_pts, device=device) * TWO_PI
    t = torch.rand(n_pts, device=device) * 15.0
    for _ in range(warmup):
        predict(x, y, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        predict(x, y, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_batch = 1000.0 * elapsed / reps
    return {
        "n_pts": n_pts,
        "reps": reps,
        "ms_per_batch": ms_batch,
        "us_per_point": 1000.0 * ms_batch / n_pts,
        "points_per_s": n_pts / (elapsed / reps),
    }


def data_mse(predict, dns_g, device, times=GATE_TIMES, step=2) -> dict:
    xs = dns_g["x"]
    xs_s = xs[::step]
    xg, yg = torch.meshgrid(xs_s, xs_s, indexing="ij")
    x, y = xg.reshape(-1), yg.reshape(-1)
    ts = dns_g["t"]
    per = {}
    mses = []
    for tv in times:
        ti = int((ts - tv).abs().argmin().item())
        t = torch.full_like(x, float(ts[ti]))
        pred = predict(x, y, t)
        tgt = torch.stack(
            [
                dns_g["u"][ti][::step, ::step].reshape(-1),
                dns_g["v"][ti][::step, ::step].reshape(-1),
                dns_g["p"][ti][::step, ::step].reshape(-1),
            ],
            dim=-1,
        )
        mse = (pred - tgt).pow(2).mean().item()
        mses.append(mse)
        per[f"{float(ts[ti]):.1f}"] = mse
    return {"mse_mean": sum(mses) / len(mses), "mse_max": max(mses), "per_time": per}


def pde_residual_mse(predict, dns_g, device, n_pts=4096, seed=0) -> dict:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    nu = float(dns_g["nu"])
    x = (torch.rand(n_pts, generator=g, device=device) * TWO_PI).requires_grad_(True)
    y = (torch.rand(n_pts, generator=g, device=device) * TWO_PI).requires_grad_(True)
    t = (torch.rand(n_pts, generator=g, device=device) * 15.0).requires_grad_(True)
    uvp = predict(x, y, t)
    u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]
    u_t, u_x, u_y = _grad(u, t), _grad(u, x), _grad(u, y)
    v_t, v_x, v_y = _grad(v, t), _grad(v, x), _grad(v, y)
    p_x, p_y = _grad(p, x), _grad(p, y)
    u_xx, u_yy = _grad(u_x, x), _grad(u_y, y)
    v_xx, v_yy = _grad(v_x, x), _grad(v_y, y)
    f_c = u_x + v_y
    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return {
        "n_pts": n_pts,
        "continuity_mse": f_c.pow(2).mean().item(),
        "momentum_x_mse": f_u.pow(2).mean().item(),
        "momentum_y_mse": f_v.pow(2).mean().item(),
        "pde_mse": (f_c.pow(2) + f_u.pow(2) + f_v.pow(2)).mean().item(),
    }


def bench_one(name, predict, dns_g, device, n_params, cfg) -> dict:
    fields = eval_fields(lambda a, b, c: predict(a, b, c).detach(), dns_g, device,
                         times=GATE_TIMES, step=1)
    return {
        "model": name,
        "n_deployed_params": n_params,
        "config": {k: cfg[k] for k in cfg if k in (
            "arch", "hidden", "k_max", "qt_hidden", "qt_fourier", "n_freqs",
            "n_qubits", "n_layers", "bottleneck_width", "nu", "t_max",
        )},
        "gate": {
            "omega_rel_l2_max": fields["omega_max"],
            "vel_rel_l2_max": fields["vel_max"],
            "omega_gate_2pct": fields["omega_max"] <= OMEGA_GATE,
            "per_time": fields["times"],
        },
        "data_loss": data_mse(predict, dns_g, device),
        "pde_residual": pde_residual_mse(predict, dns_g, device),
        "latency": latency_ms(predict, device),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="blog/checkpoint/v4/bench.json")
    args = p.parse_args()
    device = resolve_device(args.device)
    dns = torch.load(V3 / "dns" / "reference.pt", map_location="cpu", weights_only=False)
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}

    classical, cfg_c, n_c = load_classical(device)
    target, w_q, cfg_q, n_q = load_quantum(device)

    def pred_c(x, y, t):
        return classical(x, y, t)

    def pred_q(x, y, t):
        return target(x, y, t, w_q)

    report = {
        "dns": {"nu": float(dns["nu"]), "t_max": float(dns["t_max"]), "grid": int(dns["x"].numel())},
        "device": str(device),
        "classical": bench_one("classical", pred_c, dns_g, device, n_c, cfg_c),
        "quantum": bench_one("quantum", pred_q, dns_g, device, n_q, cfg_q),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    for key in ("classical", "quantum"):
        b = report[key]
        g, lat, data, pde = b["gate"], b["latency"], b["data_loss"], b["pde_residual"]
        print(f"\n=== {key} ({b['n_deployed_params']} params) ===")
        print(f"  ω_max={100*g['omega_rel_l2_max']:.3f}%  vel_max={100*g['vel_rel_l2_max']:.3f}%  "
              f"gate={g['omega_gate_2pct']}")
        print(f"  data MSE mean={data['mse_mean']:.3e}  max={data['mse_max']:.3e}")
        print(f"  PDE MSE={pde['pde_mse']:.3e}  (cont={pde['continuity_mse']:.3e}  "
              f"mx={pde['momentum_x_mse']:.3e}  my={pde['momentum_y_mse']:.3e})")
        print(f"  latency={lat['ms_per_batch']:.3f} ms / {lat['n_pts']} pts  "
              f"({lat['points_per_s']/1e6:.2f} Mpts/s)")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
