"""Loads config.yaml from the project root."""

import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # src/qt_pinn/config_loader.py -> repo root


def load() -> dict:
    return yaml.safe_load((_ROOT / "config.yaml").read_text())
