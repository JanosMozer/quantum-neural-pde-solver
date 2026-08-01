"""Loads the Burgers2D PDE config.yaml."""

import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # src/qt_pinn/config_loader.py -> repo root
_CONFIG_PATH = _ROOT / "pdes" / "burgers2d" / "config.yaml"


def load() -> dict:
    return yaml.safe_load(_CONFIG_PATH.read_text())
