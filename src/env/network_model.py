"""Lightweight, reusable helpers for believable (non-physical) network behaviour.

Deliberately simple: no fading models, no RF propagation equations. These
functions turn (distance, environment, randomness) into small multipliers
that src/env/network_conditions.py blends into its existing time-based curves.
"""

from __future__ import annotations

import numpy as np


def distance_quality(
    distance: float,
    communication_range: float,
    good_range: float,
    medium_range: float,
) -> float:
    """Map inter-agent distance to a link quality in [0, 1].

    <= good_range                      -> 1.0  (excellent)
    good_range .. medium_range         -> linear 1.0 -> 0.5
    medium_range .. communication_range -> linear 0.5 -> 0.0
    > communication_range              -> 0.0  (out of range)
    """
    if distance <= good_range:
        return 1.0
    if distance <= medium_range:
        span = max(medium_range - good_range, 1e-6)
        return 1.0 - 0.5 * (distance - good_range) / span
    if distance <= communication_range:
        span = max(communication_range - medium_range, 1e-6)
        return 0.5 - 0.5 * (distance - medium_range) / span
    return 0.0


def wireless_jitter(rng: np.random.Generator, scale: float) -> float:
    """Small zero-mean random wireless fluctuation."""
    if scale <= 0:
        return 0.0
    return float(rng.uniform(-scale, scale))


def environment_factor(environment: str, factors: dict[str, float]) -> float:
    """Environment-specific degradation multiplier (>= 1.0 means worse)."""
    return float(factors.get(environment, 1.0))


def interference_windows(
    rng: np.random.Generator,
    total_steps: int,
    count: int = 2,
    min_len: int = 5,
    max_len: int = 15,
) -> list[tuple[int, int]]:
    """A handful of short, infrequent interference windows over the run."""
    if total_steps <= 0 or count <= 0:
        return []
    windows: list[tuple[int, int]] = []
    for _ in range(count):
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, max(total_steps - length, 1)))
        windows.append((start, start + length))
    return windows


def in_any_window(t: int, windows: list[tuple[int, int]]) -> bool:
    return any(start <= t < end for start, end in windows)