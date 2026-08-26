# Orbit / swirl fidelity campaign — closed with v4

Attempted to add **peak co-rotation** (ω maxima orbit) to the Rel-L2 gate for
vortex merger. Concluded without promoting a swirl-faithful solver to
`v4/classical` or `v4/quantum`.

## Gates used (final)

- Rel-L2 ω / velocity ≤ **2%** (user: under 2% is fine)
- m4 co-rotation vs DNS ≤ **5%** through **t=15** (early-only gates were misleading)

## What worked

- **`orbit_omega ≈ −1.22` Fourier features** (`sin/cos(Ωt)`, `sin/cos(2Ωt)` in
  `FourierFeatureMapHarmonic`) — necessary for continuous co-rotation; plain `t,t²` freezes.
- **m4 annular phase** as the orbit metric (full-field pattern match was noisy).
- Classical HarmMLP **128–128**, `k_max=6`, orbit features:
  - **s4_ft**: Rel ~**1.5% / 1.5%** (best pointwise) but **late swirl freezes** after t≈6 if only early orbit is fit.
  - **s7_ft**: full-horizon swirl ~**≤5%**, Rel drifts to ~**4–5%**.
  - **s9_rel_ft**: Rel back to ~**1.87% / 1.80%**, swirl ~**7%** (close, not gated).
- Dense FD polish at **n_side=96 blew up** near-gate models; **64-grid** is safe.
- Adam aggressive FT on the old 96–96 basin **destroys ω**; gentle LR + reject blow-ups required.

## What did not work

- Pointwise ω Rel-L2 **alone** — can look ≤2% while peaks **angle-freeze** vs DNS.
- Orbit gate only on **t≤5** — false passes; merger after t≈6 is the hard part.
- Distill / e2e generators to h48 with orbit features — stalled ~15–20% ω; not competitive.
- Fair QT advantage under matched arch — already **null** on pointwise ω
  (`advantage_scoreboard.md`); swirl-gated QT never completed.
- Co-rotating spatial frame warm-start from lab-frame weights — destroys Rel (~200%).
- Promoting swirl-capable weights into product `classical/` — **not done**; product
  checkpoints remain the pre-orbit inject pair.

## Kept artifacts

| Path | Role |
|------|------|
| `../classical/`, `../quantum/` | Product v4 inject (pre-orbit) |
| `../advantage_*`, `../advantage_scoreboard.md` | Fair QT null result |
| `s4_ft/` | Best Rel + early orbit features |
| `s7_ft/` | Best full-horizon swirl (~5%) |
| `s9_rel_ft/` | Best Rel+swirl compromise (~1.9% Rel, ~7% swirl) |

## Disposition

v4 product story unchanged: **throughput from smaller MLP, not circuit**; **no robust QT win**.
Swirl-faithful merger PINN left as open research; v5 returns to **non-orbit TGV** media.
