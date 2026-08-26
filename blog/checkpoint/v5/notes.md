# v5 — Taylor–Green animations

- **Dense / multi-vortex:** `media/tgv_dense.gif` with wavenumber **k=2**.
- **Solver comparison:** `media/tgv_triplet.gif` — Exact | Classical | Quantum (k=1, v2 polish weights).
- **Unstable (strong BL boost):** `unstable/media/tgv_unstable_triplet.gif` — DNS | Classical | Quantum.
  - Defaults: boost×**5.5**, ν=**0.03**, T=**12** (~12s @ 10 fps) so the imbalance stays visible far longer than the earlier ×2.4 / ν=0.1 / T=7 run.
- Colormap: merger-style deep blue → yellow.
- Generators: `scripts/plot_tgv_v5_contour.py`, `scripts/plot_tgv_unstable_triplet.py`

See also: [`../MODEL_CARD.md`](../MODEL_CARD.md), [`../blog.md`](../blog.md).
