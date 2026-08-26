# Unstable Taylor–Green (bottom-left boost)

Start from standard 2D TGV, then **amplify the bottom-left vortex** (center $(\pi/2,\pi/2)$, boost×5.5, mask δ=1.05). That breaks the exact-solution balance so nonlinear advection turns on. Lower viscosity (ν=0.03) keeps the imbalance visible through T=12.0.

- DNS: spectral NS, ν=0.03, T=12.0, Δt_save=0.1
- Classical / Quantum: trained from scratch on this DNS (soft IC; no v2 TGV warm-start)
- `tgv_unstable_triplet.gif` — DNS | Classical | Quantum (121 frames @ 10 fps ≈ 12.1s)
- Red **+** marks the boosted center
