"""Generate analysis plots for NS run: training curves + field comparisons + exact error."""

import sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

from qt_pinn.pinn_target_ns import TargetPINNNS
from qt_pinn.qnn_generator_ns import QuantumWeightGeneratorNS
from pdes.ns2d.physics_loss import exact_solution

RUN_DIR = Path("checkpoints/ns_run_0001")
NU = 0.01
X_LO, X_HI = 0.0, 2.0 * math.pi

# ---- load model ----
gen = QuantumWeightGeneratorNS(bottleneck_width=64)
sd  = torch.load(RUN_DIR / "q_weights.pt", map_location="cpu", weights_only=True)
gen.load_state_dict(sd)
gen.eval()
model = TargetPINNNS(fourier_sigma=1.0)
model.eval()

with torch.no_grad():
    weights = gen()

# ---- generate field on grid ----
N = 64
xs = torch.linspace(X_LO, X_HI, N)
ys = torch.linspace(X_LO, X_HI, N)
xg, yg = torch.meshgrid(xs, ys, indexing="ij")
xf = xg.flatten(); yf = yg.flatten()

T_VALS = [0.0, 0.25, 0.5, 1.0]

# ---- Figure 1: u-field predicted vs exact at 4 times ----
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
vmin_u, vmax_u = -1.1, 1.1
vmin_p, vmax_p = -0.6, 0.6

for row, tv in enumerate(T_VALS):
    t_tensor = torch.full_like(xf, tv)
    with torch.no_grad():
        pred = model(xf, yf, t_tensor, weights)   # (N^2, 3)
    u_pred = pred[:, 0].numpy().reshape(N, N)
    v_pred = pred[:, 1].numpy().reshape(N, N)
    p_pred = pred[:, 2].numpy().reshape(N, N)

    u_ex, v_ex, p_ex = exact_solution(xf, yf, t_tensor)
    u_ex = u_ex.numpy().reshape(N, N)
    v_ex = v_ex.numpy().reshape(N, N)
    p_ex = p_ex.numpy().reshape(N, N)

    xnp = xs.numpy(); ynp = ys.numpy()

    # col 0: u pred, col 1: u exact, col 2: u error, col 3: p error
    im0 = axes[row, 0].pcolormesh(xnp, ynp, u_pred.T, vmin=vmin_u, vmax=vmax_u,
                                   cmap="RdBu_r", shading="auto")
    im1 = axes[row, 1].pcolormesh(xnp, ynp, u_ex.T,   vmin=vmin_u, vmax=vmax_u,
                                   cmap="RdBu_r", shading="auto")
    err_u = np.abs(u_pred - u_ex)
    im2 = axes[row, 2].pcolormesh(xnp, ynp, err_u.T, cmap="hot_r", shading="auto")
    err_p = np.abs(p_pred - p_ex)
    im3 = axes[row, 3].pcolormesh(xnp, ynp, err_p.T, cmap="hot_r", shading="auto")

    plt.colorbar(im0, ax=axes[row, 0])
    plt.colorbar(im1, ax=axes[row, 1])
    plt.colorbar(im2, ax=axes[row, 2])
    plt.colorbar(im3, ax=axes[row, 3])

    axes[row, 0].set_title(f"u pred  t={tv}")
    axes[row, 1].set_title(f"u exact t={tv}")
    axes[row, 2].set_title(f"|u err|  t={tv}")
    axes[row, 3].set_title(f"|p err|  t={tv}")
    for ax in axes[row]:
        ax.set_xlabel("x"); ax.set_ylabel("y")

fig.suptitle("NS Taylor-Green: predicted vs exact (ns_run_0001)", fontsize=13)
plt.tight_layout()
out1 = RUN_DIR / "fields_uvp.png"
plt.savefig(out1, dpi=130)
plt.close()
print(f"Saved {out1}")

# ---- Figure 2: x-slice time evolution (u and v) ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = cm.plasma(np.linspace(0.1, 0.9, len(T_VALS)))
x_slice = torch.linspace(X_LO, X_HI, 200)
y_fixed = torch.full((200,), math.pi)   # y = pi (middle of domain)

for tv, col in zip(T_VALS, colors):
    t_tensor = torch.full((200,), tv)
    with torch.no_grad():
        pred = model(x_slice, y_fixed, t_tensor, weights)
    u_ex, v_ex, p_ex = exact_solution(x_slice, y_fixed, t_tensor)

    axes[0].plot(x_slice.numpy(), pred[:, 0].numpy(), color=col, label=f"pred t={tv}")
    axes[0].plot(x_slice.numpy(), u_ex.numpy(), "--", color=col, alpha=0.6)
    axes[1].plot(x_slice.numpy(), pred[:, 1].numpy(), color=col, label=f"pred t={tv}")
    axes[1].plot(x_slice.numpy(), v_ex.numpy(), "--", color=col, alpha=0.6)
    axes[2].plot(x_slice.numpy(), pred[:, 2].numpy(), color=col, label=f"pred t={tv}")
    axes[2].plot(x_slice.numpy(), p_ex.numpy(), "--", color=col, alpha=0.6)

for ax, field in zip(axes, ["u(x,π,t)", "v(x,π,t)", "p(x,π,t)"]):
    ax.set_xlabel("x"); ax.set_ylabel(field)
    ax.set_title(field + "  solid=pred  dashed=exact")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

fig.suptitle("NS Taylor-Green: x-slice evolution (ns_run_0001)", fontsize=12)
plt.tight_layout()
out2 = RUN_DIR / "slice_evolution.png"
plt.savefig(out2, dpi=130)
plt.close()
print(f"Saved {out2}")

# ---- Figure 3: loss trajectory + rel-L2 bar chart ----
results = json.loads((RUN_DIR / "results.json").read_text())
exact_l2 = results["exact_l2"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# parse training log from terminal output (re-build from known values)
steps_logged = list(range(0, 18000, 500))
pde_vals  = [0.037364, 0.010210, 0.010117, 0.010302, 0.010327, 0.010143,
             0.010232, 0.010223, 0.010374, 0.010228, 0.010294, 0.010304,
             0.010525, 0.010920, 0.010738, 0.010739, 0.010482, 0.011093,
             0.010889, 0.010884, 0.010883, 0.010948, 0.010889, 0.010902,
             0.010893, 0.010900, 0.010905, 0.010911, 0.010907, 0.010909,
             0.010909, 0.010909, 0.010909, 0.010911, 0.010914, 0.010915]
bc_vals   = [0.590634, 0.542879, 0.542992, 0.542648, 0.542593, 0.542946,
             0.542835, 0.542793, 0.542427, 0.542804, 0.542681, 0.542646,
             0.541939, 0.541465, 0.541491, 0.541451, 0.541486, 0.540341,
             0.540342, 0.540306, 0.540289, 0.540444, 0.540251, 0.540235,
             0.540213, 0.540204, 0.540176, 0.540196, 0.540159, 0.540144,
             0.540134, 0.540127, 0.540118, 0.540107, 0.540096, 0.540088]

ax = axes[0]
ax.semilogy(steps_logged, pde_vals,  label="pde_loss",  color="steelblue")
ax.semilogy(steps_logged, bc_vals,   label="bc_loss",   color="coral")
ax.set_xlabel("Adam step"); ax.set_ylabel("loss (log scale)")
ax.set_title("Training loss: NS Taylor-Green (ns_run_0001)")
ax.legend(); ax.grid(True, alpha=0.3)

# Burgers run_0072 reference (from known results)
burgers_pde = 0.01974268
burgers_bc  = 0.02232525
ax.axhline(burgers_pde, color="steelblue", linestyle=":", alpha=0.5, label="Burgers pde (run_0072)")
ax.axhline(burgers_bc,  color="coral",     linestyle=":", alpha=0.5, label="Burgers bc (run_0072)")
ax.legend(fontsize=8)

# Bar chart: rel-L2 per field and time
ax2 = axes[1]
t_labels = [f"t={tv}" for tv in ["0.0", "0.25", "0.5", "1.0"]]
u_errs = [exact_l2[tv]["u"] * 100 for tv in ["0.0", "0.25", "0.5", "1.0"]]
v_errs = [exact_l2[tv]["v"] * 100 for tv in ["0.0", "0.25", "0.5", "1.0"]]
p_errs = [exact_l2[tv]["p"] * 100 for tv in ["0.0", "0.25", "0.5", "1.0"]]

x_pos = np.arange(4)
w = 0.25
ax2.bar(x_pos - w, u_errs, w, label="u", color="steelblue")
ax2.bar(x_pos,     v_errs, w, label="v", color="coral")
ax2.bar(x_pos + w, p_errs, w, label="p", color="seagreen")
ax2.axhline(100, color="k", linestyle="--", lw=0.8, label="100% (random)")
ax2.set_xticks(x_pos); ax2.set_xticklabels(t_labels)
ax2.set_ylabel("Relative L2 error %")
ax2.set_title("Exact solution error vs Taylor-Green (ns_run_0001)")
ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out3 = RUN_DIR / "loss_and_errors.png"
plt.savefig(out3, dpi=130)
plt.close()
print(f"Saved {out3}")

print("\nAll plots saved.")
