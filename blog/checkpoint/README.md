# Blog checkpoints

Each `vN/` is self-contained (models + media) unless noted.

| Version | Content |
|---------|---------|
| `v1/` | TGV demo gate-pass (~0.95% / ~1.36%) |
| `v2/` | TGV polish (~0.61% / ~0.62%) + `media/` |
| `v3/` | Vortex merger: DNS + classical/QT (ω ≤ 2%, matched size) + `media/` |
| `v4/` | Fast quantum (≈2.5× throughput, ω within +0.5pp of classical) + notes/bench |

v4 quantum:
```bash
.venv/bin/python scripts/train_merger_qt_fast.py
```
