"""TGV demo trainer: classical direct PINN or single-ν Quantum-Train.

Architecture story
------------------
  Classical: DirectNSMLP [64,64] (5,059 params) — bigger, trains in minutes
  Quantum:   TargetPINNNS [24–32] (≤1,507 params) deployed; circuit generates
             weights once per training step (not once per collocation point)

  .venv/bin/python scripts/train_tgv_demo.py --model classical --preset polish
  .venv/bin/python scripts/train_tgv_demo.py --model quantum   --preset polish
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import yaml

from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.qnn_generator_cond import (
    ConditionedQuantumGenerator,
    ConditionedQuantumGeneratorV2,
)
from qt_pinn.tgv_demo import (
    DEMO_NU,
    DEMO_T_MAX,
    PRESETS,
    DirectNSMLP,
    data_loss,
    eval_rel_l2,
    gate_pass,
    make_bc,
    make_colloc,
    pde_loss,
    predict,
    prepare_run_dir,
    resolve_device,
    vel_max,
    write_run,
)


def load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TGV demo trainer")
    p.add_argument("--model", choices=["classical", "quantum"], required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="scout")
    p.add_argument("--config", default="")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-id", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nu", type=float, default=DEMO_NU)
    p.add_argument("--t-max", type=float, default=DEMO_T_MAX)
    p.add_argument("--hard-ic", dest="hard_ic", action="store_true", default=True)
    p.add_argument("--soft-ic", dest="hard_ic", action="store_false")
    p.add_argument("--adam-steps", type=int, default=None)
    p.add_argument("--n-colloc", type=int, default=None)
    p.add_argument("--n-bc", type=int, default=None)
    p.add_argument("--classical-hidden", type=int, nargs=2, default=None)
    p.add_argument("--qt-hidden", type=int, nargs=2, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda-data", type=float, default=None)
    p.add_argument("--lambda-pde", type=float, default=None)
    p.add_argument("--lambda-bc", type=float, default=None)
    p.add_argument("--t-weight", dest="t_weight", action="store_true", default=None)
    p.add_argument("--no-t-weight", dest="t_weight", action="store_false")
    p.add_argument("--t-sample", choices=["uniform", "tail"], default=None)
    p.add_argument("--n-qubits", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--bottleneck-width", type=int, default=None)
    p.add_argument("--qc-arch", choices=["expect", "reupload"], default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--resample-every", type=int, default=None)
    p.add_argument("--budget-s", type=float, default=None)
    p.add_argument("--loss-mode", choices=["both", "data", "pde"], default="both")
    return p.parse_args()


def merge_cfg(args: argparse.Namespace) -> dict:
    cfg = dict(PRESETS[args.preset])
    cfg.update({k: v for k, v in load_yaml(args.config).items() if v is not None})
    for k, v in vars(args).items():
        if k in ("config", "preset") or v is None:
            continue
        cfg[k] = v
    cfg["classical_hidden"] = list(cfg.get("classical_hidden") or [64, 64])
    cfg["qt_hidden"] = list(cfg.get("qt_hidden") or [16, 16])
    cfg.setdefault("t_sample", "uniform")
    cfg["preset"] = args.preset
    return cfg


def build_quantum(cfg: dict, device: torch.device):
    qt_h = tuple(cfg["qt_hidden"])
    target = TargetPINNNS(
        fourier="tgv", hard_ic=cfg["hard_ic"],
        hidden=qt_h, t_max=cfg["t_max"],
    ).to(device)
    common = dict(
        in_dim=target.in_dim,
        h1=qt_h[0], h2=qt_h[1],
        n_qubits=cfg["n_qubits"],
        n_layers=cfg["n_layers"],
        bottleneck_width=cfg["bottleneck_width"],
        nu_range=(0.05, 0.2),
        freq_mode="linear",
    )
    if cfg["qc_arch"] == "expect":
        gen = ConditionedQuantumGeneratorV2(nu_encode="log", **common).to(device)
    else:
        gen = ConditionedQuantumGenerator(**common).to(device)
    return target, gen


def weights_for(gen, nu: float, device: torch.device) -> dict[str, torch.Tensor]:
    w = gen(torch.tensor([nu], device=device, dtype=torch.float32))
    return {k: v[0] for k, v in w.items()}


def save_checkpoint(run_dir: Path, gen, target, nu: float, device: torch.device) -> None:
    if gen is None:
        torch.save(target.state_dict(), run_dir / "model.pt")
        return
    torch.save(gen.state_dict(), run_dir / "generator.pt")
    with torch.no_grad():
        w = weights_for(gen, nu, device)
        torch.save({k: v.detach().cpu() for k, v in w.items()}, run_dir / "deployed_weights.pt")


def load_best(run_dir: Path, gen, target, device: torch.device) -> None:
    if gen is None:
        path = run_dir / "model.pt"
        if path.exists():
            target.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    else:
        path = run_dir / "generator.pt"
        if path.exists():
            gen.load_state_dict(torch.load(path, map_location=device, weights_only=True))


def main() -> None:
    args = parse_args()
    cfg = merge_cfg(args)
    device = resolve_device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    tag = "c" if cfg["model"] == "classical" else "q"
    run_id = cfg["run_id"] or f"tgv_demo_{tag}_{cfg['preset']}_s{cfg['seed']}"
    run_dir = prepare_run_dir(run_id, cfg.get("overwrite", False))

    nu, t_max = cfg["nu"], cfg["t_max"]
    t_sample = cfg.get("t_sample", "uniform")

    if cfg["model"] == "classical":
        model = DirectNSMLP(
            hidden=tuple(cfg["classical_hidden"]),
            hard_ic=cfg["hard_ic"], t_max=t_max,
        ).to(device)
        gen = None
        params = list(model.parameters())
        target = model
        n_deployed = sum(p.numel() for p in model.parameters())
    else:
        target, gen = build_quantum(cfg, device)
        params = list(gen.parameters())
        model = target
        n_deployed = int(gen.total_weights)

    n_train_params = sum(p.numel() for p in params)
    print(f"Run: {run_id}  model={cfg['model']}  preset={cfg['preset']}  device={device}")
    print(f"nu={nu}  T={t_max}  decay={math.exp(-2*nu*t_max):.3f}  hard_ic={cfg['hard_ic']}")
    print(f"λ_data={cfg['lambda_data']}  λ_pde={cfg['lambda_pde']}  "
          f"t_weight={cfg.get('t_weight', False)}  t_sample={t_sample}  loss={cfg['loss_mode']}")
    if cfg["model"] == "classical":
        print(f"classical  hidden={cfg['classical_hidden']}  params={n_deployed:,}")
    else:
        print(f"QT: {cfg['qc_arch']}  {cfg['n_qubits']}q×{cfg['n_layers']}L  "
              f"deployed_hidden={cfg['qt_hidden']}  deployed_params={n_deployed:,}  "
              f"generator_params={n_train_params:,}")

    x, y, t = make_colloc(cfg["n_colloc"], device, t_max, t_sample)
    x_bc, y_bc, t_bc = make_bc(cfg["n_bc"], device)
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["adam_steps"], eta_min=1e-5)

    tw = cfg.get("t_weight", False)
    history = []
    best_vel = float("inf")
    t0 = time.time()
    stopped_at = cfg["adam_steps"]
    print(f"{'step':>6}  {'total':>10}  {'pde':>10}  {'data':>10}  {'vel%':>8}")

    for step in range(cfg["adam_steps"]):
        if cfg.get("budget_s") and (time.time() - t0) > cfg["budget_s"]:
            stopped_at = step
            print(f"\nbudget stop at step {step} ({cfg['budget_s']}s)")
            break
        if cfg["resample_every"] and step and step % cfg["resample_every"] == 0:
            x, y, t = make_colloc(cfg["n_colloc"], device, t_max, t_sample)
            x_bc, y_bc, t_bc = make_bc(cfg["n_bc"], device)

        w = None if gen is None else weights_for(gen, nu, device)
        opt.zero_grad()
        uvp = predict(target, x, y, t, w)

        mode = cfg["loss_mode"]
        pde = pde_loss(uvp, x, y, t, nu) if mode in ("both", "pde") else torch.zeros((), device=device)
        dat = data_loss(uvp, x, y, t, nu, tw) if mode in ("both", "data") else torch.zeros((), device=device)

        if cfg["hard_ic"]:
            bc = torch.zeros((), device=device)
        else:
            uvp_bc = predict(target, x_bc, y_bc, t_bc, w)
            bc = data_loss(uvp_bc, x_bc, y_bc, t_bc, nu, False)

        loss = cfg["lambda_pde"] * pde + cfg["lambda_data"] * dat + cfg.get("lambda_bc", 0) * bc
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        do_log = step % cfg["log_every"] == 0
        do_eval = step % cfg["eval_every"] == 0
        vel = None
        if do_eval:
            w_e = None if gen is None else weights_for(gen, nu, device)
            errs = eval_rel_l2(target, device, nu, t_max, w_e)
            vel = vel_max(errs)
            if vel < best_vel:
                best_vel = vel
                save_checkpoint(run_dir, gen, target, nu, device)
        if do_log:
            pde_v = pde.item() if torch.is_tensor(pde) else float(pde)
            dat_v = dat.item() if torch.is_tensor(dat) else float(dat)
            vstr = f"{100*vel:8.3f}" if vel is not None else "       -"
            print(f"{step:6d}  {loss.item():10.5f}  {pde_v:10.5f}  "
                  f"{dat_v:10.5f}  {vstr}  [{time.time()-t0:.0f}s]")
            history.append({
                "step": step, "total": loss.item(),
                "pde": pde_v, "data": dat_v,
                "vel_max": vel, "elapsed_s": round(time.time() - t0, 1),
            })

    # Final eval on *best* checkpoint (reload), then re-save if final is better.
    elapsed = time.time() - t0
    w_now = None if gen is None else weights_for(gen, nu, device)
    errs_now = eval_rel_l2(target, device, nu, t_max, w_now, n_grid=64)
    vel_now = vel_max(errs_now)
    if vel_now < best_vel:
        best_vel = vel_now
        save_checkpoint(run_dir, gen, target, nu, device)
    else:
        load_best(run_dir, gen, target, device)
        if not (run_dir / ("model.pt" if gen is None else "generator.pt")).exists():
            save_checkpoint(run_dir, gen, target, nu, device)

    w_f = None if gen is None else weights_for(gen, nu, device)
    errs = eval_rel_l2(target, device, nu, t_max, w_f, n_grid=64)
    vmax = vel_max(errs)
    ms = errs.get("_ms_per_frame_256sq", 0.0)
    passed = gate_pass(errs, 0.02)
    # ensure deployed weights file exists for quantum
    if gen is not None:
        save_checkpoint(run_dir, gen, target, nu, device)

    print(f"\nBest vel rel-L2 max={100*vmax:.3f}%  gate(≤2%)={'PASS' if passed else 'FAIL'}  "
          f"inference={ms:.3f} ms/frame (256²)  [{elapsed:.0f}s]")
    for k, e in errs.items():
        if k.startswith("_"):
            continue
        print(f"  t={e['t']:4.2f}  u={100*e['u']:.3f}%  v={100*e['v']:.3f}%  "
              f"p={100*e['p']:.3f}%  vel={100*e['vel']:.3f}%")

    write_run(run_dir, {
        "run_id": run_id, "n_train_params": n_train_params,
        "n_deployed_params": n_deployed, **cfg,
    }, {
        "run_id": run_id,
        "model": cfg["model"],
        "preset": cfg["preset"],
        "n_train_params": n_train_params,
        "n_deployed_params": n_deployed,
        "elapsed_s": round(elapsed, 1),
        "stopped_at": stopped_at,
        "vel_rel_l2_max": round(vmax, 6),
        "gate_pass": passed,
        "ms_per_frame_256sq": round(ms, 4),
        "exact_l2": {k: v for k, v in errs.items() if not k.startswith("_")},
        "history": history,
    })
    print(f"Saved -> {run_dir}/")


if __name__ == "__main__":
    main()
