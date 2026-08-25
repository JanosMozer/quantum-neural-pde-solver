"""v2 campaign: streamfunction PINN + grid ω loss → target ω_max ≤ 2%.

Conclusion from E1–E2: pointwise (u,v)+curl loss plateaus ~7–9% ω because
filamentary structure is under-resolved. v2 uses:
  - streamfunction ψ ⇒ u=ψ_y, v=-ψ_x, ω=-(ψ_xx+ψ_yy) (div-free by construction)
  - mixed random + dense grid batches for ω/uv matching
  - relative ω loss

  .venv/bin/python scripts/exp_merger_omega_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from pdes.ns2d.physics_loss import _grad, relative_l2
from qt_pinn.fourier import FourierFeatureMapWide
from qt_pinn.ns2d_spectral import TWO_PI, ic_values_from_dns, sample_dns
from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.qnn_generator_cond import ConditionedQuantumGeneratorV2
from qt_pinn.tgv_demo import resolve_device

# reuse eval + media from v1 script
from scripts.exp_merger_omega import (
    OMEGA_GATE,
    V3,
    EXP,
    eval_fields,
    regenerate_media,
)

ROOT = Path(__file__).resolve().parents[1]


class StreamFnMLP(nn.Module):
    """Outputs (ψ, p). Velocity/vorticity from autograd of ψ."""

    def __init__(self, hidden=(512, 512, 512), t_max=40.0, n_freqs=192):
        super().__init__()
        self.fourier = FourierFeatureMapWide(n_freqs=n_freqs, sigma=1.5, t_max=t_max, seed=0)
        d = self.fourier.out_dim
        layers: list[nn.Module] = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 2))  # ψ, p
        self.net = nn.Sequential(*layers)

    def forward_raw(self, x, y, t):
        return self.net(self.fourier(torch.stack([x, y, t], dim=-1)))

    def forward_uvp(self, x, y, t):
        raw = self.forward_raw(x, y, t)
        psi, p = raw[:, 0], raw[:, 1]
        # u = ∂ψ/∂y, v = −∂ψ/∂x
        u = _grad(psi, y)
        v = -_grad(psi, x)
        return torch.stack([u, v, p], dim=-1), psi


class UVPOmegaMLP(nn.Module):
    """Direct (u,v,p,ω) with soft curl consistency."""

    def __init__(self, hidden=(512, 512, 512), t_max=40.0, n_freqs=192):
        super().__init__()
        self.fourier = FourierFeatureMapWide(n_freqs=n_freqs, sigma=1.5, t_max=t_max, seed=1)
        d = self.fourier.out_dim
        layers: list[nn.Module] = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        return self.net(self.fourier(torch.stack([x, y, t], dim=-1)))


def sample_grid_batch(dns, device, n_side=64, t_sample="early"):
    """Dense spatial grid at one random time — strong ω supervision."""
    ts = dns["t"].to(device)
    n_t = ts.numel()
    if t_sample == "early":
        ti = int((torch.rand(1, device=device).pow(2) * (0.45 * (n_t - 1))).item())
    else:
        ti = int(torch.randint(0, n_t, (1,), device=device).item())
    # subsample spatial
    xs = dns["x"].to(device)
    n = xs.numel()
    step = max(1, n // n_side)
    xs_s = xs[::step][:n_side]
    xg, yg = torch.meshgrid(xs_s, xs_s, indexing="ij")
    x = xg.reshape(-1).requires_grad_(True)
    y = yg.reshape(-1).requires_grad_(True)
    t = torch.full_like(x, float(ts[ti])).requires_grad_(True)
    u = dns["u"][ti].to(device)[::step, ::step][:n_side, :n_side].reshape(-1)
    v = dns["v"][ti].to(device)[::step, ::step][:n_side, :n_side].reshape(-1)
    p = dns["p"][ti].to(device)[::step, ::step][:n_side, :n_side].reshape(-1)
    w = dns["omega"][ti].to(device)[::step, ::step][:n_side, :n_side].reshape(-1)
    return x, y, t, torch.stack([u, v, p], -1), w


def rel_mse(a, b, eps=1e-3):
    return ((a - b).pow(2) / (b.pow(2) + eps)).mean()


def train_stream_classical(dns, device, steps=40000, name="E7_stream_classical"):
    print(f"\n======== {name} ========")
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
    model = StreamFnMLP(hidden=(512, 512, 512), t_max=float(dns["t_max"]), n_freqs=192).to(device)
    n_dep = sum(p.numel() for p in model.parameters())
    print(f"params={n_dep:,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    best_w, best_sd = 1e9, None
    t0 = time.time()
    history = []

    for step in range(steps):
        # random point batch
        x, y, t, tgt, w_dns = sample_dns(dns_g, 16384, device, t_sample="early")
        x, y, t = x.requires_grad_(True), y.requires_grad_(True), t.requires_grad_(True)
        uvp, psi = model.forward_uvp(x, y, t)
        # ω = −∇²ψ
        psi_x = _grad(psi, x)
        psi_y = _grad(psi, y)
        w_pred = -(_grad(psi_x, x) + _grad(psi_y, y))

        loss_uvp = (uvp - tgt).pow(2).mean()
        loss_w = (w_pred - w_dns).pow(2).mean()

        # dense grid batch every step (smaller)
        xg, yg, tg, tgt_g, wg = sample_grid_batch(dns_g, device, n_side=48, t_sample="early")
        uvp_g, psi_g = model.forward_uvp(xg, yg, tg)
        psi_gx = _grad(psi_g, xg)
        psi_gy = _grad(psi_g, yg)
        w_g = -(_grad(psi_gx, xg) + _grad(psi_gy, yg))
        loss_grid = (uvp_g - tgt_g).pow(2).mean() + 2.0 * (w_g - wg).pow(2).mean()

        # IC
        n_ic = 2048
        x_ic = torch.empty(n_ic, device=device).uniform_(0, TWO_PI).requires_grad_(True)
        y_ic = torch.empty(n_ic, device=device).uniform_(0, TWO_PI).requires_grad_(True)
        t_ic = torch.zeros(n_ic, device=device).requires_grad_(True)
        u0, v0, p0 = ic_values_from_dns(dns_g, x_ic, y_ic)
        uvp_ic, _ = model.forward_uvp(x_ic, y_ic, t_ic)
        loss_ic = (uvp_ic - torch.stack([u0, v0, p0], -1)).pow(2).mean()

        loss = 50.0 * loss_uvp + 20.0 * loss_w + 10.0 * loss_grid + 40.0 * loss_ic
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps - 1:
            errs = eval_stream_model(model, dns_g, device)
            print(
                f"{step:6d}  loss={loss.item():.4f}  ω%={100*errs['omega_max']:.2f}  "
                f"vel%={100*errs['vel_max']:.2f}  [{time.time()-t0:.0f}s]"
            )
            history.append({"step": step, "omega_max": errs["omega_max"],
                            "vel_max": errs["vel_max"], "loss": loss.item()})
            if errs["omega_max"] < best_w:
                best_w = errs["omega_max"]
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if best_w <= OMEGA_GATE:
                    print(f"EARLY PASS ω={100*best_w:.3f}%")
                    break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    final = eval_stream_model(model, dns_g, device, step=1)
    print(f"RESULT {name}: ω_max={100*final['omega_max']:.3f}% vel={100*final['vel_max']:.3f}% "
          f"pass={final['omega_max']<=OMEGA_GATE}")
    exp_dir = EXP / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_sd or model.state_dict(), exp_dir / "model.pt")
    # save as uvp predictor wrapper state is stream fn — for media we need special load
    cfg = {
        "arch": "streamfunction",
        "hidden": [512, 512, 512],
        "n_freqs": 192,
        "t_max": float(dns["t_max"]),
        "nu": float(dns["nu"]),
        "n_deployed_params": n_dep,
    }
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (exp_dir / "results.json").write_text(json.dumps({
        "final": final, "history": history, "passed": final["omega_max"] <= OMEGA_GATE,
        "n_deployed_params": n_dep,
    }, indent=2) + "\n")
    return final, best_sd, cfg, model


def eval_stream_model(model, dns, device, times=(0.0, 2.0, 5.0, 8.0, 12.0, 15.0), step=2):
    """Eval streamfunction model (needs grad for u,v from ψ)."""
    xs = dns["x"].to(device)
    xs_s = xs[::step]
    ns = xs_s.numel()
    xg, yg = torch.meshgrid(xs_s, xs_s, indexing="ij")
    out = {"times": {}, "vel_max": 0.0, "omega_max": 0.0}
    ts = dns["t"]
    dx = float(TWO_PI / ns)
    for tv in times:
        ti = int((ts - tv).abs().argmin().item())
        t_real = float(ts[ti])
        x = xg.reshape(-1).detach().requires_grad_(True)
        y = yg.reshape(-1).detach().requires_grad_(True)
        t = torch.full_like(x, t_real).requires_grad_(True)
        uvp, psi = model.forward_uvp(x, y, t)
        # omega from −∇²ψ
        px = _grad(psi, x)
        py = _grad(psi, y)
        w_pred = -(_grad(px, x) + _grad(py, y)).detach()
        uvp = uvp.detach()
        u_ex = dns["u"][ti].to(device)[::step, ::step].reshape(-1)
        v_ex = dns["v"][ti].to(device)[::step, ::step].reshape(-1)
        w_ex = dns["omega"][ti].to(device)[::step, ::step].reshape(-1)
        vel = 0.5 * (relative_l2(uvp[:, 0], u_ex) + relative_l2(uvp[:, 1], v_ex))
        w_e = relative_l2(w_pred, w_ex)
        out["times"][f"{t_real:.1f}"] = {"vel": vel, "omega": w_e}
        out["vel_max"] = max(out["vel_max"], vel)
        out["omega_max"] = max(out["omega_max"], w_e)
    out["omega_gate"] = out["omega_max"] <= OMEGA_GATE
    return out


def train_uvpw_classical(dns, device, steps=40000, name="E8_uvpw_classical"):
    print(f"\n======== {name} ========")
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
    model = UVPOmegaMLP(hidden=(512, 512, 512), t_max=float(dns["t_max"]), n_freqs=192).to(device)
    n_dep = sum(p.numel() for p in model.parameters())
    print(f"params={n_dep:,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    best_w, best_sd = 1e9, None
    t0 = time.time()
    history = []

    for step in range(steps):
        x, y, t, tgt, w_dns = sample_dns(dns_g, 16384, device, t_sample="early")
        x, y, t = x.requires_grad_(True), y.requires_grad_(True), t.requires_grad_(True)
        out = model(x, y, t)
        uvp, w_head = out[:, :3], out[:, 3]
        curl = _grad(uvp[:, 1], x) - _grad(uvp[:, 0], y)
        loss = (
            30.0 * (uvp - tgt).pow(2).mean()
            + 100.0 * ((w_head - w_dns).pow(2).mean() + rel_mse(w_head, w_dns))
            + 50.0 * (w_head - curl).pow(2).mean()
            + 20.0 * ((curl - w_dns).pow(2).mean() + rel_mse(curl, w_dns))
        )
        xg, yg, tg, tgt_g, wg = sample_grid_batch(dns_g, device, n_side=48)
        out_g = model(xg, yg, tg)
        loss = loss + 20.0 * (out_g[:, :3] - tgt_g).pow(2).mean() + 80.0 * (
            (out_g[:, 3] - wg).pow(2).mean() + rel_mse(out_g[:, 3], wg)
        )

        n_ic = 2048
        x_ic = torch.empty(n_ic, device=device).uniform_(0, TWO_PI)
        y_ic = torch.empty(n_ic, device=device).uniform_(0, TWO_PI)
        t_ic = torch.zeros(n_ic, device=device)
        u0, v0, p0 = ic_values_from_dns(dns_g, x_ic, y_ic)
        out_ic = model(x_ic, y_ic, t_ic)
        loss = loss + 30.0 * (out_ic[:, :3] - torch.stack([u0, v0, p0], -1)).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps - 1:
            def pred_eval(xx, yy, tt):
                return model(xx, yy, tt)[:, :3]

            # For ω eval use head if available — monkey via FD still in eval_fields
            # Better: custom eval using ω head
            errs = eval_omega_head(model, dns_g, device)
            print(
                f"{step:6d}  loss={loss.item():.4f}  ω%={100*errs['omega_max']:.2f}  "
                f"vel%={100*errs['vel_max']:.2f}  [{time.time()-t0:.0f}s]"
            )
            history.append({"step": step, **{k: errs[k] for k in ("omega_max", "vel_max")}})
            if errs["omega_max"] < best_w:
                best_w = errs["omega_max"]
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if best_w <= OMEGA_GATE:
                    print(f"EARLY PASS ω={100*best_w:.3f}%")
                    break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    final = eval_omega_head(model, dns_g, device, step=1)
    print(f"RESULT {name}: ω_max={100*final['omega_max']:.3f}% vel={100*final['vel_max']:.3f}% "
          f"pass={final['omega_max']<=OMEGA_GATE}")
    exp_dir = EXP / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_sd or model.state_dict(), exp_dir / "model.pt")
    cfg = {"arch": "uvp_omega", "hidden": [512, 512, 512], "n_freqs": 192,
           "t_max": float(dns["t_max"]), "nu": float(dns["nu"]), "n_deployed_params": n_dep}
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (exp_dir / "results.json").write_text(json.dumps({
        "final": final, "history": history, "passed": final["omega_max"] <= OMEGA_GATE,
        "n_deployed_params": n_dep,
    }, indent=2) + "\n")
    return final, best_sd, cfg, model


@torch.no_grad()
def eval_omega_head(model, dns, device, times=(0.0, 2.0, 5.0, 8.0, 12.0, 15.0), step=2):
    xs = dns["x"].to(device)
    xs_s = xs[::step]
    xg, yg = torch.meshgrid(xs_s, xs_s, indexing="ij")
    x, y = xg.reshape(-1), yg.reshape(-1)
    out = {"times": {}, "vel_max": 0.0, "omega_max": 0.0}
    ts = dns["t"]
    for tv in times:
        ti = int((ts - tv).abs().argmin().item())
        t_real = float(ts[ti])
        t = torch.full_like(x, t_real)
        pred = model(x, y, t)
        u_ex = dns["u"][ti].to(device)[::step, ::step].reshape(-1)
        v_ex = dns["v"][ti].to(device)[::step, ::step].reshape(-1)
        w_ex = dns["omega"][ti].to(device)[::step, ::step].reshape(-1)
        vel = 0.5 * (relative_l2(pred[:, 0], u_ex) + relative_l2(pred[:, 1], v_ex))
        w_e = relative_l2(pred[:, 3], w_ex)
        out["times"][f"{t_real:.1f}"] = {"vel": vel, "omega": w_e}
        out["vel_max"] = max(out["vel_max"], vel)
        out["omega_max"] = max(out["omega_max"], w_e)
    out["omega_gate"] = out["omega_max"] <= OMEGA_GATE
    return out


def train_quantum_uvpw(dns, device, steps=20000, name="E9_quantum_uvpw"):
    """QT generates weights for MLP with 4 outputs (u,v,p,ω)."""
    print(f"\n======== {name} ========")
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
    t_max = float(dns["t_max"])
    nu = float(dns["nu"])
    # Use TargetPINNNS-like but 4 outs — build custom via generator h1,h2,out=4
    # ConditionedQuantumGeneratorV2 has out_dim=3 fixed in unpacking of target.
    # So keep 3-output target but add a small classical ω head? That breaks QT story.
    # Instead: wide Fourier target 3-out + heavy omega curl loss + more capacity.
    hidden = (96, 96)
    target = TargetPINNNS(
        fourier="wide", hard_ic=False, hidden=hidden, t_max=t_max,
        n_freqs=48, fourier_seed=0,
    ).to(device)
    gen = ConditionedQuantumGeneratorV2(
        in_dim=target.in_dim, h1=hidden[0], h2=hidden[1], out_dim=3,
        n_qubits=8, n_layers=12, bottleneck_width=128,
        nu_range=(0.001, 0.05), nu_encode="log",
    ).to(device)
    print(f"deployed={gen.total_weights:,}  gen_params={sum(p.numel() for p in gen.parameters()):,}")
    opt = torch.optim.Adam(gen.parameters(), lr=1.5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    best_w, best = 1e9, None
    t0 = time.time()
    history = []

    def predict(x, y, t):
        w = gen(torch.tensor([nu], device=device, dtype=torch.float32))
        w = {k: v[0] for k, v in w.items()}
        return target(x, y, t, w)

    for step in range(steps):
        x, y, t, tgt, w_dns = sample_dns(dns_g, 6144, device, t_sample="early")
        x, y, t = x.requires_grad_(True), y.requires_grad_(True), t.requires_grad_(True)
        uvp = predict(x, y, t)
        curl = _grad(uvp[:, 1], x) - _grad(uvp[:, 0], y)
        loss = (
            30.0 * (uvp - tgt).pow(2).mean()
            + 200.0 * ((curl - w_dns).pow(2).mean() + rel_mse(curl, w_dns))
        )
        xg, yg, tg, tgt_g, wg = sample_grid_batch(dns_g, device, n_side=32)
        uvp_g = predict(xg, yg, tg)
        curl_g = _grad(uvp_g[:, 1], xg) - _grad(uvp_g[:, 0], yg)
        loss = loss + 15.0 * (uvp_g - tgt_g).pow(2).mean() + 150.0 * (
            (curl_g - wg).pow(2).mean() + rel_mse(curl_g, wg)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps - 1:
            errs = eval_fields(lambda a, b, c: predict(a, b, c).detach(), dns_g, device)
            print(
                f"{step:6d}  loss={loss.item():.4f}  ω%={100*errs['omega_max']:.2f}  "
                f"vel%={100*errs['vel_max']:.2f}  [{time.time()-t0:.0f}s]"
            )
            history.append({"step": step, "omega_max": errs["omega_max"], "vel_max": errs["vel_max"]})
            if errs["omega_max"] < best_w:
                best_w = errs["omega_max"]
                with torch.no_grad():
                    packed = gen(torch.tensor([nu], device=device))
                    best = {
                        "generator": {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()},
                        "deployed": {k: v[0].detach().cpu().clone() for k, v in packed.items()},
                    }
                if best_w <= OMEGA_GATE:
                    print(f"EARLY PASS ω={100*best_w:.3f}%")
                    break

    if best is not None:
        gen.load_state_dict(best["generator"])
    final = eval_fields(lambda a, b, c: predict(a, b, c).detach(), dns_g, device, step=1)
    print(f"RESULT {name}: ω_max={100*final['omega_max']:.3f}% vel={100*final['vel_max']:.3f}% "
          f"pass={final['omega_max']<=OMEGA_GATE}")
    exp_dir = EXP / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    if best:
        torch.save(best["generator"], exp_dir / "generator.pt")
        torch.save(best["deployed"], exp_dir / "deployed_weights.pt")
    cfg = {
        "arch": "qt_wide_uvp", "qt_hidden": list(hidden), "qt_fourier": "wide",
        "n_freqs": 48, "n_qubits": 8, "n_layers": 12, "bottleneck_width": 128,
        "t_max": t_max, "nu": nu, "n_deployed_params": int(gen.total_weights),
    }
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (exp_dir / "results.json").write_text(json.dumps({
        "final": final, "history": history, "passed": final["omega_max"] <= OMEGA_GATE,
        "n_deployed_params": int(gen.total_weights),
    }, indent=2) + "\n")
    return final, best, cfg


def promote_stream_or_uvpw(arch, cfg, state, model_kind="classical"):
    dest = V3 / model_kind
    dest.mkdir(parents=True, exist_ok=True)
    if model_kind == "classical":
        torch.save(state, dest / "model.pt")
        out = {**cfg, "model": "classical", "exp_arch": arch}
        (dest / "config.json").write_text(json.dumps(out, indent=2) + "\n")
    else:
        torch.save(state["generator"], dest / "generator.pt")
        torch.save(state["deployed"], dest / "deployed_weights.pt")
        out = {**cfg, "model": "quantum"}
        (dest / "config.json").write_text(json.dumps(out, indent=2) + "\n")


def main():
    device = resolve_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    EXP.mkdir(parents=True, exist_ok=True)
    dns = torch.load(V3 / "dns" / "reference.pt", map_location="cpu", weights_only=False)
    log = []

    # E8 first: direct ω head (most direct path to ω≤2%)
    final8, sd8, cfg8, _ = train_uvpw_classical(dns, device, steps=50000, name="E8_uvpw_classical")
    log.append({"exp": "E8", "omega_max": final8["omega_max"], "vel_max": final8["vel_max"],
                "passed": final8["omega_max"] <= OMEGA_GATE})
    best_c = ("E8", final8["omega_max"], sd8, cfg8, "uvpw")

    # E7 streamfunction as backup classical
    if final8["omega_max"] > OMEGA_GATE:
        final7, sd7, cfg7, _ = train_stream_classical(dns, device, steps=30000, name="E7_stream_classical")
        log.append({"exp": "E7", "omega_max": final7["omega_max"], "vel_max": final7["vel_max"],
                    "passed": final7["omega_max"] <= OMEGA_GATE})
        if final7["omega_max"] < best_c[1]:
            best_c = ("E7", final7["omega_max"], sd7, cfg7, "stream")

    promote_stream_or_uvpw(best_c[4], best_c[3], best_c[2], "classical")
    (V3 / "classical" / "results.json").write_text(json.dumps({
        "omega_rel_l2_max": best_c[1],
        "vel_rel_l2_max": final8.get("vel_max", best_c[1]),
        "gate_pass_omega_2pct": best_c[1] <= OMEGA_GATE,
        "exp": best_c[0],
        "arch": best_c[4],
    }, indent=2) + "\n")

    # E9 quantum
    final9, best9, cfg9 = train_quantum_uvpw(dns, device, steps=25000, name="E9_quantum_wide")
    log.append({"exp": "E9", "omega_max": final9["omega_max"], "vel_max": final9["vel_max"],
                "passed": final9["omega_max"] <= OMEGA_GATE})
    if best9:
        promote_stream_or_uvpw("qt_wide", cfg9, best9, "quantum")
        (V3 / "quantum" / "results.json").write_text(json.dumps({
            "omega_rel_l2_max": final9["omega_max"],
            "vel_rel_l2_max": final9["vel_max"],
            "gate_pass_omega_2pct": final9["omega_max"] <= OMEGA_GATE,
            "exp": "E9",
        }, indent=2) + "\n")

    (EXP / "log_v2.json").write_text(json.dumps(log, indent=2) + "\n")

    try:
        regenerate_media_flexible(dns, device, best_c[4])
    except Exception as e:
        print(f"media regen failed: {e}")

    print("\n=== V2 SUMMARY ===")
    for e in log:
        print(f"  {e['exp']}: ω={100*e['omega_max']:.3f}%  pass={e['passed']}")
    c_ok = best_c[1] <= OMEGA_GATE
    q_ok = final9["omega_max"] <= OMEGA_GATE
    # write conclusions
    (EXP / "CONCLUSIONS.md").write_text(
        "# ω ≤ 2% campaign\n\n"
        + "\n".join(f"- {e}" for e in log)
        + f"\n\nClassical best: {100*best_c[1]:.3f}% ({best_c[0]})\n"
        + f"Quantum: {100*final9['omega_max']:.3f}%\n"
        + f"Tracers removed from media.\n"
    )
    if c_ok and q_ok:
        print("GOAL MET")
        sys.exit(0)
    print("GOAL NOT MET YET")
    sys.exit(2)


def regenerate_media_flexible(
    dns, device, classical_arch: str, ckpt_root: Path | None = None,
):
    """Media without tracers; supports stream / uvpw / merger mlp classical."""
    ckpt_root = ckpt_root or V3
    import math
    import subprocess
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scripts.exp_merger_omega import CMAP, MergerMLP

    media = ckpt_root / "media"
    media.mkdir(exist_ok=True)
    cfg_c = json.loads((ckpt_root / "classical" / "config.json").read_text())
    arch = cfg_c.get("arch") or cfg_c.get("exp_arch") or classical_arch

    if arch == "streamfunction" or arch == "stream":
        model_c = StreamFnMLP(
            hidden=tuple(cfg_c.get("hidden", [512, 512, 512])),
            t_max=dns["t_max"], n_freqs=cfg_c.get("n_freqs", 192),
        ).to(device)
        model_c.load_state_dict(torch.load(ckpt_root/"classical/model.pt", map_location=device, weights_only=True))
        model_c.eval()

        def pred_c(x, y, t):
            x = x.requires_grad_(True); y = y.requires_grad_(True); t = t.requires_grad_(True)
            uvp, _ = model_c.forward_uvp(x, y, t)
            return uvp.detach()

        def omega_c(x, y, t, shape):
            x = x.requires_grad_(True); y = y.requires_grad_(True); t = t.requires_grad_(True)
            raw = model_c.forward_raw(x, y, t)
            psi = raw[:, 0]
            px = _grad(psi, x); py = _grad(psi, y)
            w = -(_grad(px, x) + _grad(py, y))
            return w.detach().cpu().numpy().reshape(shape)
    elif arch == "uvp_omega" or arch == "uvpw":
        model_c = UVPOmegaMLP(
            hidden=tuple(cfg_c.get("hidden", [512, 512, 512])),
            t_max=dns["t_max"], n_freqs=cfg_c.get("n_freqs", 192),
        ).to(device)
        model_c.load_state_dict(torch.load(ckpt_root/"classical/model.pt", map_location=device, weights_only=True))
        model_c.eval()

        def pred_c(x, y, t):
            return model_c(x, y, t)[:, :3]

        def omega_c(x, y, t, shape):
            uvp = model_c(x, y, t)[:, :3].detach().cpu().numpy().reshape(*shape, 3)
            dx = TWO_PI / shape[0]
            u, v = uvp[..., 0], uvp[..., 1]
            dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx)
            dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
            return dvdx - dudy
    elif arch == "wide_uvp":
        from scripts.exp_merger_omega_v5 import WideUVP
        model_c = WideUVP(
            hidden=tuple(cfg_c.get("hidden", [768, 768, 768])),
            n_freqs=cfg_c.get("n_freqs", 384), t_max=float(dns["t_max"]),
        ).to(device)
        model_c.load_state_dict(torch.load(ckpt_root/"classical/model.pt", map_location=device, weights_only=True))
        model_c.eval()

        def pred_c(x, y, t):
            return model_c(x, y, t)

        def omega_c(x, y, t, shape):
            uvp = model_c(x, y, t).detach().cpu().numpy().reshape(*shape, 3)
            dx = TWO_PI / shape[0]
            u, v = uvp[..., 0], uvp[..., 1]
            dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx)
            dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
            return dvdx - dudy
    elif arch == "deep_uvp":
        from scripts.exp_merger_omega_v21 import DeepUVP
        model_c = DeepUVP(
            hidden=tuple(cfg_c.get("hidden", [384, 384, 384])),
            n_freqs=cfg_c.get("n_freqs", 192), t_max=float(dns["t_max"]),
            sigma=cfg_c.get("sigma", 1.5),
        ).to(device)
        model_c.load_state_dict(torch.load(ckpt_root/"classical/model.pt", map_location=device, weights_only=True))
        model_c.eval()

        def pred_c(x, y, t):
            return model_c(x, y, t)

        def omega_c(x, y, t, shape):
            uvp = model_c(x, y, t).detach().cpu().numpy().reshape(*shape, 3)
            dx = TWO_PI / shape[0]
            u, v = uvp[..., 0], uvp[..., 1]
            dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx)
            dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
            return dvdx - dudy
    elif arch == "harm_mlp":
        from scripts.exp_merger_omega import HarmMLP
        model_c = HarmMLP(
            hidden=tuple(cfg_c.get("hidden", [96, 96])),
            t_max=float(dns["t_max"]),
            k_max=int(cfg_c.get("k_max", 6)),
            axis_extra=int(cfg_c.get("axis_extra", 0)),
        ).to(device)
        model_c.load_state_dict(torch.load(ckpt_root/"classical/model.pt", map_location=device, weights_only=True))
        model_c.eval()

        def pred_c(x, y, t):
            return model_c(x, y, t)

        def omega_c(x, y, t, shape):
            uvp = model_c(x, y, t).detach().cpu().numpy().reshape(*shape, 3)
            dx = TWO_PI / shape[0]
            u, v = uvp[..., 0], uvp[..., 1]
            dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx)
            dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
            return dvdx - dudy
    else:
        model_c = MergerMLP(
            hidden=tuple(cfg_c["classical_hidden"]), t_max=dns["t_max"],
            n_freqs=cfg_c.get("n_freqs", 96),
        ).to(device)
        model_c.load_state_dict(torch.load(ckpt_root/"classical/model.pt", map_location=device, weights_only=True), strict=False)
        model_c.eval()

        def pred_c(x, y, t):
            return model_c(x, y, t)

        def omega_c(x, y, t, shape):
            uvp = model_c(x, y, t).detach().cpu().numpy().reshape(*shape, 3)
            dx = TWO_PI / shape[0]
            u, v = uvp[..., 0], uvp[..., 1]
            dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx)
            dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
            return dvdx - dudy

    cfg_q = json.loads((ckpt_root / "quantum" / "config.json").read_text())
    q_fourier = cfg_q.get("qt_fourier", "tgv")
    target = TargetPINNNS(
        fourier=q_fourier, hard_ic=False,
        hidden=tuple(cfg_q.get("qt_hidden", [96, 96])),
        t_max=dns["t_max"], n_freqs=cfg_q.get("n_freqs", 48), fourier_seed=0,
    ).to(device)
    wq = torch.load(ckpt_root / "quantum" / "deployed_weights.pt", map_location=device, weights_only=True)

    xs = dns["x"].numpy()
    step = max(1, len(xs) // 128)
    xs_s = xs[::step]
    vmax = float(dns["omega"][0].max()) * 1.05
    ts = dns["t"].numpy()

    def lerp(name, t):
        i = int(np.argmin(np.abs(ts - t)))
        # linear
        if t <= ts[0]:
            return dns[name][0].numpy()
        if t >= ts[-1]:
            return dns[name][-1].numpy()
        j = int(np.searchsorted(ts, t) - 1)
        j = max(0, min(j, len(ts) - 2))
        a = (t - ts[j]) / (ts[j + 1] - ts[j])
        return ((1 - a) * dns[name][j] + a * dns[name][j + 1]).numpy()

    def fd_w(u, v, dx):
        dvdx = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx)
        dudy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
        return dvdx - dudy

    def panels_at(tv):
        o_ex = lerp("omega", tv)[::step, ::step]
        xg, yg = np.meshgrid(xs_s, xs_s, indexing="ij")
        xt = torch.tensor(xg.ravel(), device=device, dtype=torch.float32)
        yt = torch.tensor(yg.ravel(), device=device, dtype=torch.float32)
        tt = torch.full_like(xt, float(tv))
        oc = omega_c(xt, yt, tt, xg.shape)
        with torch.no_grad():
            pq = target(xt, yt, tt, wq).cpu().numpy().reshape(*xg.shape, 3)
        dx = TWO_PI / xg.shape[0]
        oq = fd_w(pq[..., 0], pq[..., 1], dx)
        return o_ex, oc, oq

    def phase(t):
        if t < 4:
            return "1. Four co-rotating vortices"
        if t < 10:
            return "2. Orbiting & stretching toward center"
        return "3. Merging into one core"

    t_want = [0.0, 2.0, 5.0, 8.0, 12.0, 15.0]
    fig, axes = plt.subplots(len(t_want), 3, figsize=(11, 13), constrained_layout=True)
    for row, tv in enumerate(t_want):
        for col, (arr, title) in enumerate(zip(panels_at(tv), ["DNS", "Classical", "Quantum"])):
            ax = axes[row, col]
            im = ax.pcolormesh(xs_s, xs_s, arr.T, cmap=CMAP, shading="auto", vmin=0, vmax=vmax)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"t={tv:.1f}\n{phase(tv)}", fontsize=9)
        fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.046, pad=0.02, label="ω")
    fig.suptitle("Vortex merger ω (no tracers)")
    fig.savefig(media / "merger_triplet_snapshots.png", dpi=150)
    plt.close(fig)

    t_frames = np.arange(0.0, 15.0 + 1e-9, 0.1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for fi, tv in enumerate(t_frames):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), constrained_layout=True)
            for ax, arr, title in zip(axes, panels_at(tv), ["DNS", "Classical", "Quantum"]):
                im = ax.pcolormesh(xs_s, xs_s, arr.T, cmap=CMAP, shading="auto", vmin=0, vmax=vmax)
                ax.set_xlim(0, TWO_PI); ax.set_ylim(0, TWO_PI)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(title)
            fig.colorbar(im, ax=axes.tolist(), fraction=0.046, pad=0.02, label="ω")
            fig.suptitle(f"{phase(tv)}    t = {tv:.1f} / 15")
            fig.savefig(tmp / f"frame_{fi:04d}.png", dpi=100)
            plt.close(fig)
        out = media / "merger_triplet.gif"
        palette = tmp / "palette.png"
        pattern = str(tmp / "frame_%04d.png")
        subprocess.check_call(
            ["ffmpeg", "-y", "-framerate", "10", "-i", pattern, "-vf", "palettegen", str(palette)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["ffmpeg", "-y", "-framerate", "10", "-i", pattern, "-i", str(palette),
             "-lavfi", "paletteuse", str(out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    (media / "CAPTION.md").write_text(
        "# Vortex merger\n\nDeep blue→yellow vorticity. No tracer dots / streamlines.\n"
        "Gif: t=0..15, Δt=0.1, 10 fps (~15s).\n"
    )
    print("Wrote media (no tracers)")


if __name__ == "__main__":
    main()
