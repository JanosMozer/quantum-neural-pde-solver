"""Loads config.yaml from the project root (parent of this package)."""

import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def load() -> dict:
    return yaml.safe_load((_ROOT / "config.yaml").read_text())
