"""Configuration loader utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS = _ROOT / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIGS / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_thresholds() -> dict[str, Any]:
    return load_yaml("thresholds.yaml")


def get_llm_config() -> dict[str, Any]:
    return load_yaml("llm.yaml")


def project_root() -> Path:
    return _ROOT
