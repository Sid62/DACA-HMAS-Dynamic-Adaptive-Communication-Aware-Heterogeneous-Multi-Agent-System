"""Constants shared across coordination modules to ensure path-symmetric evaluation."""

from __future__ import annotations

from typing import Any
from src.config import get_thresholds


def get_completion_radius(
    thresholds: dict[str, Any] | None = None,
    scenario: str | None = None,
) -> float:
    """Retrieve completion radius from config (scenario-specific or global)."""
    if thresholds is not None:
        if scenario and isinstance(thresholds.get("scenarios"), dict):
            sc_cfg = thresholds["scenarios"].get(scenario)
            if isinstance(sc_cfg, dict) and "completion_radius" in sc_cfg:
                return float(sc_cfg["completion_radius"])
        if "completion_radius" in thresholds:
            return float(thresholds["completion_radius"])
    try:
        th = get_thresholds()
        if scenario and isinstance(th.get("scenarios"), dict):
            sc_cfg = th["scenarios"].get(scenario)
            if isinstance(sc_cfg, dict) and "completion_radius" in sc_cfg:
                return float(sc_cfg["completion_radius"])
        if "completion_radius" in th:
            return float(th["completion_radius"])
    except Exception:
        pass
    return 6.0


COMPLETION_RADIUS_M: float = get_completion_radius()


def __getattr__(name: str) -> Any:
    if name == "COMPLETION_RADIUS_M":
        return get_completion_radius()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
