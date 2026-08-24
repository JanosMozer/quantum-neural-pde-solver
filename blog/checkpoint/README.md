# Blog checkpoints

Each `vN/` is self-contained (models + media).

| Version | Content |
|---------|---------|
| `v1/` | TGV demo gate-pass (~0.95% / ~1.36%) |
| `v2/` | TGV polish (~0.61% / ~0.62%) + `media/` |
| `v3/` | Vortex merger (4→1 swirl): DNS + classical/QT + `media/` |

Reproduce v3:
```bash
.venv/bin/python scripts/train_vortex_merger.py --stage all --overwrite
```
