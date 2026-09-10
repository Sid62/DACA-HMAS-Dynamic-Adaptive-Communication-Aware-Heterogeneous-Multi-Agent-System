"""Physically-grounded wireless communication model helpers.

Models distance attenuation (Log-Distance Path Loss), scenario-specific obstacle
shadowing (rubble, steel structures, warehouse racks), small-scale Rician/Rayleigh
AR(1) fading, periodic network congestion/recovery cycles, and burst packet errors.
"""

from __future__ import annotations

import numpy as np


def distance_quality(
    distance: float,
    communication_range: float = 50.0,
    good_range: float = 20.0,
    medium_range: float = 40.0,
    path_loss_exponent: float = 2.5,
) -> float:
    """
    Log-Distance Path Loss Model for wireless signal quality Q_dist(d) in [0, 1].

    - distance <= good_range: excellent link (Q ~ 1.0)
    - distance > good_range: Log-distance attenuation based on path loss exponent eta.
    - distance > communication_range: disconnected (Q = 0.0)
    """
    if distance <= 0.0:
        return 1.0
    if distance <= good_range:
        return 1.0
    if distance > communication_range:
        return 0.0

    # Log-distance path loss relative to reference distance d0 = good_range
    ratio = distance / max(good_range, 1e-3)
    loss_db = 10.0 * path_loss_exponent * np.log10(ratio)
    max_loss_db = 10.0 * path_loss_exponent * np.log10(max(communication_range / good_range, 1.001))

    q = 1.0 - (loss_db / max_loss_db)
    return float(np.clip(q, 0.05, 1.0))


def wireless_jitter(rng: np.random.Generator, scale: float) -> float:
    """Small zero-mean random wireless fluctuation."""
    if scale <= 0:
        return 0.0
    return float(rng.uniform(-scale, scale))


def environment_factor(environment: str, factors: dict[str, float]) -> float:
    """Environment-specific degradation multiplier (>= 1.0 means worse)."""
    return float(factors.get(environment, 1.0))


def scenario_shadowing(
    environment: str,
    t: int,
    cycle_period: float = 60.0,
) -> float:
    """
    Scenario-specific temporal/spatial shadowing and obstacle attenuation:

    - Search & Rescue ('disaster'): Smoke plumes, rubble, collapsed structures.
      Temporary dips during active clearing phases, followed by structural recovery.
    - Inspection ('urban'): Steel machinery enclosures, pipe corridors.
      Intermittent shielding while inspecting inside metal structures, restoring upon exit.
    - Logistics ('warehouse'): Moving forklifts, high metal shelving, aisle WiFi traffic.
      Short aisle interference cycles with clear corridor recovery.
    """
    phase = 2.0 * np.pi * t / cycle_period

    if environment == "disaster":
        # Search & Rescue: Dust/smoke events + structural obstacles
        dip = 0.25 * np.sin(phase) + 0.10 * np.cos(2.0 * phase)
        return float(np.clip(1.0 + dip, 0.60, 1.0))
    elif environment == "urban":
        # Inspection: Structural shielding when entering metal machinery/pipes
        dip = 0.30 * np.sin(phase) + 0.08 * np.sin(3.0 * phase)
        return float(np.clip(1.0 + dip, 0.55, 1.0))
    elif environment == "warehouse":
        # Logistics: Aisle congestion & forklift movement
        dip = 0.20 * np.sin(phase) + 0.05 * np.sin(4.0 * phase)
        return float(np.clip(1.0 + dip, 0.65, 1.0))
    else:
        return 1.0


def rician_fading_step(
    prev_state: float,
    correlation: float,
    rng: np.random.Generator,
    scale: float = 0.05,
    k_factor_db: float = 6.0,
) -> tuple[float, float]:
    """
    Rician AR(1) Gauss-Markov process for small-scale fast fading and multipath.

    Returns (new_fading_value, new_state).
    """
    noise = rng.normal(0.0, 1.0)
    new_state = (
        correlation * prev_state
        + np.sqrt(max(1.0 - correlation**2, 0.0)) * noise
    )
    # Rician K-factor weighting between LOS dominant path and multipath scatterers
    k_linear = 10.0 ** (k_factor_db / 10.0)
    los_weight = np.sqrt(k_linear / (k_linear + 1.0))
    multipath_weight = np.sqrt(1.0 / (k_linear + 1.0))

    fading_val = scale * (los_weight + multipath_weight * new_state)
    return float(fading_val), float(new_state)


def burst_loss_active(
    rng: np.random.Generator,
    t: int,
    burst_probability: float = 0.03,
    burst_duration: int = 3,
) -> bool:
    """
    Occasional burst error events (deep multipath nulls / temporary shadowing).
    """
    # Deterministic burst trigger based on seed and step window
    cycle = (t // burst_duration) * burst_duration
    val = float(rng.uniform(0.0, 1.0))
    return (t % burst_duration < 2) and (val < burst_probability)


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