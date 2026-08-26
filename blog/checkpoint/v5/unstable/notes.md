# v5 unstable TGV

Bottom-left lobe boost×5.5 (δ=1.05), ν=0.03, T=12 — destroys TGV exact-balance so vortices interact longer.

- Media: `media/tgv_unstable_triplet.gif` (121 frames @ 10 fps ≈ 12.1s)
- Classical: DirectNSMLP 128–128, soft IC, **no** v2 warm-start (data-heavy fit on this DNS)
- Quantum: TargetPINNNS tgv 32–32 deployed weights, same protocol
- Earlier weak run (boost×2.4, ν=0.1, T=7): `archive_boost2p4/`
