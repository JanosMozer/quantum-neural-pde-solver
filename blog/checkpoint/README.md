# Blog checkpoints

Each `vN/` is self-contained (models + media) unless noted.

| Version | Content |
|---------|---------|
| `v1/` | TGV demo gate-pass (~0.95% / ~1.36%) |
| `v2/` | TGV polish (~0.61% / ~0.62%) + `media/` |
| `v3/` | Vortex merger: DNS + classical/QT (ω ≤ 2%, matched size) + `media/` |
| `v4/` | Product HarmMLP teacher + inject student + `dns/` + `media/`; fair advantage / orbit evidence in `archive/` (**null** / not promoted) |
| `v5/` | TGV return: dense \|ω\|, Exact\|Classical\|Quantum triplet, **unstable** BL-boost triplet |

**Model card (best metrics):** [`../MODEL_CARD.md`](../MODEL_CARD.md)  
**Article:** [`../blog.md`](../blog.md)

v4 quantum (historical inject):
```bash
.venv/bin/python scripts/train_merger_qt_fast.py
```

v5 contours / unstable:
```bash
.venv/bin/python -u scripts/plot_tgv_v5_contour.py --n 768 --dpi 320
.venv/bin/python -u scripts/plot_tgv_unstable_triplet.py --device cuda
```
