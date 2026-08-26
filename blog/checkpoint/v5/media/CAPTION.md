# Taylor–Green vortex (v5)

Deep blue → yellow colormap. Exact field $|\omega|=2k|\sin(kx)\sin(ky)|e^{-2\nu k^{2}t}$.

## Multi-vortex (screenshot-like density)

- `tgv_dense.gif` — **k=2** → denser lobes (4×4 for k=2), t=0…15 Δt=0.1, 151 frames @ 10 fps.
- `tgv_dense_t0.png`, `tgv_dense_snapshots.png`

## Exact | Classical | Quantum

- `tgv_triplet.gif` — **k=1** (v2 PINN training mode), same time grid. Panels: Exact (analytic DNS) | Classical | Quantum.
- `tgv_triplet_t0.png`

Note: v2 classical/quantum are trained on **k=1** only; the denser k=2 field is exact-only.
