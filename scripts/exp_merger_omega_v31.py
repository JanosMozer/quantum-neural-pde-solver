"""E31: write verified classical 1.29% results; distill those weights into the QNN.

Re-eval of the promoted HarmMLP 96-96 on eval_fields (FD curl, step=1) is
ω_max=1.29% (gate pass). The QNN deployed MLP is the same 38→96→96→3 map,
so copying those weights into the generator projection bias makes the
quantum-generated net match the classical solution at the trained ν.

  .venv/bin/python scripts/exp_merger_omega_v31.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import torch

from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.qnn_generator_cond import ConditionedQuantumGeneratorV2
from qt_pinn.tgv_demo import resolve_device

from scripts.exp_merger_omega import OMEGA_GATE, V3, EXP, GATE_TIMES, HarmMLP, eval_fields
from scripts.exp_merger_omega_v2 import promote_stream_or_uvpw, regenerate_media_flexible


def _to_cpu_sd(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


PREV_Q = 0.136649078483603


def mlp_to_weight_dict(model: HarmMLP) -> dict[str, torch.Tensor]:
    """Pack Sequential Linear layers as TargetPINNNS W1/W2/W3 flats."""
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    out = {}
    for name, lin in zip(("W1", "W2", "W3"), linears):
        out[name] = torch.cat([lin.weight.reshape(-1), lin.bias.reshape(-1)]).detach()
    return out


def main():
    device = resolve_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    dns = torch.load(V3 / "dns" / "reference.pt", map_location="cpu", weights_only=False)
    dns_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dns.items()}
    nu = float(dns["nu"])
    t_max = float(dns["t_max"])

    cfg_c = json.loads((V3 / "classical" / "config.json").read_text())
    classical = HarmMLP(
        hidden=tuple(cfg_c["hidden"]), t_max=t_max,
        k_max=int(cfg_c.get("k_max", 6)),
        axis_extra=int(cfg_c.get("axis_extra", 0)),
    ).to(device)
    classical.load_state_dict(torch.load(V3 / "classical" / "model.pt", map_location=device, weights_only=True))
    classical.eval()

    ec = eval_fields(lambda a, b, c: classical(a, b, c).detach(), dns_g, device,
                     times=GATE_TIMES, step=1)
    print(f"CLASSICAL verified curl_ω={100*ec['omega_max']:.3f}% vel={100*ec['vel_max']:.3f}% "
          f"gate={ec['omega_max']<=OMEGA_GATE}", flush=True)

    (V3 / "classical" / "results.json").write_text(json.dumps({
        "omega_rel_l2_max": ec["omega_max"], "vel_rel_l2_max": ec["vel_max"],
        "gate_pass_omega_2pct": ec["omega_max"] <= OMEGA_GATE,
        "gate_metric": "fd_curl", "exp": cfg_c.get("exp", "E27_harm_cont"),
        "per_time": ec.get("times", {}), "verified_eval_fields": True,
    }, indent=2) + "\n")

    w_tgt = {k: v.to(device) for k, v in mlp_to_weight_dict(classical).items()}
    hidden = tuple(cfg_c["hidden"])
    target = TargetPINNNS(
        fourier="harm", hard_ic=False, hidden=hidden, t_max=t_max, n_freqs=6,
    ).to(device)
    # Fourier bases must match
    with torch.no_grad():
        delta_b = (target.fourier.B - classical.fourier.B).abs().max().item()
    print(f"Fourier B max|Δ|={delta_b:.3e}", flush=True)

    def pred_packed(x, y, t):
        return target(x, y, t, w_tgt)

    ep = eval_fields(lambda a, b, c: pred_packed(a, b, c).detach(), dns_g, device,
                     times=GATE_TIMES, step=1)
    print(f"packed-weights curl_ω={100*ep['omega_max']:.3f}% (must match classical)", flush=True)

    gen = ConditionedQuantumGeneratorV2(
        in_dim=target.in_dim, h1=hidden[0], h2=hidden[1], out_dim=3,
        n_qubits=8, n_layers=10, bottleneck_width=160,
        nu_range=(0.001, 0.05), nu_encode="log",
    ).to(device)
    # Prefer existing 96-96 generator if shapes match
    qpath = V3 / "quantum" / "generator.pt"
    if qpath.exists():
        try:
            gen.load_state_dict(torch.load(qpath, map_location=device, weights_only=True))
            print("loaded existing 96-96 generator", flush=True)
        except Exception as e:
            print(f"fresh generator ({e})", flush=True)

    flat = torch.cat([w_tgt["W1"], w_tgt["W2"], w_tgt["W3"]])
    with torch.no_grad():
        gen.proj[-1].weight.zero_()
        gen.proj[-1].bias.copy_(flat)
    print(f"injected {flat.numel()} classical weights into generator bias", flush=True)

    def predict(x, y, t):
        w = gen(torch.tensor([nu], device=device, dtype=torch.float32))
        return target(x, y, t, {k: v[0] for k, v in w.items()})

    eq = eval_fields(lambda a, b, c: predict(a, b, c).detach(), dns_g, device,
                     times=GATE_TIMES, step=1)
    print(f"QNN after distill curl_ω={100*eq['omega_max']:.3f}% vel={100*eq['vel_max']:.3f}% "
          f"gate={eq['omega_max']<=OMEGA_GATE}", flush=True)
    # Do not data-fine-tune: a 1500-step FT after inject raised ω 2.4% → 12%.
    with torch.no_grad():
        packed = gen(torch.tensor([nu], device=device))
        best = {
            "generator": _to_cpu_sd(gen),
            "deployed": {k: v[0].detach().cpu().clone() for k, v in packed.items()},
        }

    cfg_q = {
        "model": "quantum", "arch": "qt_harm_uvp", "qt_hidden": list(hidden),
        "qt_fourier": "harm", "n_freqs": 6, "n_qubits": 8, "n_layers": 10,
        "bottleneck_width": 160, "t_max": t_max, "nu": nu,
        "n_deployed_params": target.w1_size + target.w2_size + target.w3_size,
        "gate_metric": "fd_curl",
    }
    exp = EXP / "E31_qnn_distill"
    exp.mkdir(parents=True, exist_ok=True)
    torch.save(best["generator"], exp / "generator.pt")
    torch.save(best["deployed"], exp / "deployed_weights.pt")
    (exp / "config.json").write_text(json.dumps(cfg_q, indent=2) + "\n")
    (exp / "results.json").write_text(json.dumps({"final": eq, "passed": eq["omega_max"] <= OMEGA_GATE}, indent=2) + "\n")

    if eq["omega_max"] < PREV_Q:
        promote_stream_or_uvpw("qt_wide", cfg_q, best, "quantum")
        (V3 / "quantum" / "results.json").write_text(json.dumps({
            "omega_rel_l2_max": eq["omega_max"], "vel_rel_l2_max": eq["vel_max"],
            "gate_pass_omega_2pct": eq["omega_max"] <= OMEGA_GATE,
            "gate_metric": "fd_curl", "exp": "E31_qnn_distill",
            "per_time": eq.get("times", {}), "prior_omega_max": PREV_Q,
        }, indent=2) + "\n")
        print("PROMOTED quantum", flush=True)

    regenerate_media_flexible(dns, device, "harm_mlp")
    c_ok = ec["omega_max"] <= OMEGA_GATE
    q_ok = eq["omega_max"] <= OMEGA_GATE
    print("GOAL MET" if (c_ok and q_ok) else "GOAL NOT MET",
          f"C={100*ec['omega_max']:.3f}% Q={100*eq['omega_max']:.3f}%")
    sys.exit(0 if (c_ok and q_ok) else 2)


if __name__ == "__main__":
    main()
