"""Synthetic network condition injectors for communication profiles."""
 
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
 
import numpy as np
 
from src.env.agents import distance_matrix
from src.env.network_model import (
    distance_quality,
    environment_factor,
    in_any_window,
    interference_windows,
    wireless_jitter,
)

# Lightweight, fixed mapping from scenario to a plausible deployment
# environment (Goal 1). Not user-facing config since it's inherent to
# what each scenario represents.
_SCENARIO_ENVIRONMENT = {
    "logistics": "warehouse",
    "inspection": "urban",
    "search_rescue": "disaster",
}

 
class NetworkProfile(str, Enum):
    STABLE = "stable"
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    OSCILLATORY = "oscillatory"
 
 
@dataclass
class NetworkState:
    packet_loss_rate: float = 0.0
    latency: float = 0.01
    bandwidth_utilization: float = 0.1
    bytes_capacity: float = 10000.0
    bytes_delivered: float = 1000.0
    msg_sent: int = 0
    ack_received: int = 0
 
 
@dataclass
class NetworkConditionGenerator:
    """Inject synthetic packet loss, latency, bandwidth degradation."""
 
    profile: NetworkProfile = NetworkProfile.STABLE
    base_delay_prob: float = 0.0
    base_loss_rate: float = 0.0
    total_steps: int = 500
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    # --- Goal 1: lightweight realism additions (all optional/additive) ---
    fleet: Any | None = None
    communication_range: float = 50.0
    good_range: float = 20.0
    medium_range: float = 40.0
    wireless_jitter_scale: float = 0.05
    environment: str = "warehouse"
    environment_factors: dict[str, float] = field(
        default_factory=lambda: {"warehouse": 1.0, "urban": 1.15, "disaster": 1.35}
    )
    _comm_interference_windows: list[tuple[int, int]] = field(default_factory=list)

    def _distance_degradation(self) -> float | None:
       """1 - link quality in [0, 1] from average inter-agent distance.
       Returns None when no fleet is attached, leaving old behaviour intact.
       """
       if self.fleet is None or getattr(self.fleet, "n_agents", 0) < 2:
           return None
       d = distance_matrix(self.fleet.agents)
       n = d.shape[0]
       pair_dists = [d[i, j] for i in range(n) for j in range(i + 1, n)]
       if not pair_dists:
           return None
       avg_dist = float(np.mean(pair_dists))
       quality = distance_quality(
           avg_dist, self.communication_range, self.good_range, self.medium_range
       )
       return 1.0 - quality

    def loss_rate_at(self, t: int) -> float:
        if self.profile == NetworkProfile.STABLE:
            base = self.base_loss_rate
        elif self.profile == NetworkProfile.GRADUAL:
            progress = t / max(self.total_steps, 1)
            # return self.base_loss_rate + 0.10 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 50))
            base = self.base_loss_rate + 0.10 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 50))
            #base = self.base_loss_rate + 0.15 * progress
            
        elif self.profile == NetworkProfile.SUDDEN:
            base = self.base_loss_rate + (0.20 if t > self.total_steps * 0.4 else 0.0)
        elif self.profile == NetworkProfile.OSCILLATORY:
            base = min(0.45, self.base_loss_rate + 0.40 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 50)))
        else:
            base = self.base_loss_rate

        # Goal 1: distance / environment / jitter / interference (additive)
        env_mult = environment_factor(self.environment, self.environment_factors)
        degraded = base * env_mult
        dist_deg = self._distance_degradation()
        if dist_deg is not None:
            degraded += 0.5 * dist_deg
        if in_any_window(t, self._comm_interference_windows):
            degraded += 0.15
        degraded += wireless_jitter(self.rng, self.wireless_jitter_scale)
        return float(np.clip(degraded, 0.0, 1.0))
 
    def latency_at(self, t: int) -> float:
        loss = self.loss_rate_at(t)
        base = 0.01 + self.base_delay_prob * 0.5
        # return base + loss * 1.5 + self.rng.uniform(0, 0.02)
        # latency = base + loss * 4.0 + self.rng.uniform(0, 0.05)
        latency = base + loss * 1.5 + self.rng.uniform(0, 0.02)
        dist_deg = self._distance_degradation()
        if dist_deg is not None:
            latency += dist_deg * 0.3
        return latency

    def bandwidth_at(self, t: int) -> float:
        loss = self.loss_rate_at(t)
        # return max(0.0, 1.0 - loss - self.base_delay_prob * 0.3)
        # bw = max(0.15, 1.0 - 1.8 * loss - self.base_delay_prob)
        bw = max(0.0, 1.0 - loss - self.base_delay_prob * 0.3)
        dist_deg = self._distance_degradation()
        if dist_deg is not None:
            bw = max(0.10, bw - 0.3 * dist_deg)
        return bw
 
    def simulate_message(
        self, t: int, payload_bytes: float = 256.0
     ) -> NetworkState:
        loss = self.loss_rate_at(t)
        latency = self.latency_at(t)
        bw_avail = self.bandwidth_at(t)
        delivered = payload_bytes * bw_avail
        msg_sent = 1
        ack = 0 if self.rng.random() < loss else 1
        return NetworkState(
            packet_loss_rate=loss,
            latency=latency,
            bandwidth_utilization=1.0 - bw_avail,
            bytes_capacity=payload_bytes,
            bytes_delivered=delivered if ack else 0.0,
            msg_sent=msg_sent,
            ack_received=ack,
        )

 
    @classmethod
    def from_scenario(
        cls,
        scenario_name: str,
        profile: str,
        thresholds: dict[str, Any],
        seed: int = 0,
        total_steps: int = 500,
    ) ->NetworkConditionGenerator:
        sc = thresholds.get("scenarios", {}).get(scenario_name, {})
        prof = NetworkProfile(profile) if profile != "stable" else NetworkProfile.STABLE
        if scenario_name == "logistics" and profile == "stable":
            prof = NetworkProfile.STABLE

        net_cfg = thresholds.get("network", {})
        scenario_cfg = thresholds.get("scenario", {})
        rng = np.random.default_rng(seed)
        windows = (
            interference_windows(rng, total_steps)
            if scenario_cfg.get("communication_events", False)
            else []
        )
 
        return cls(
            profile=prof,
            base_delay_prob=sc.get("comm_delay_prob", 0.0),
            base_loss_rate=sc.get("packet_loss_rate", 0.0),
            total_steps=total_steps,
            rng=rng,
            communication_range=net_cfg.get("communication_range", 50.0),
            good_range=net_cfg.get("good_range", 20.0),
            medium_range=net_cfg.get("medium_range", 40.0),
            wireless_jitter_scale=net_cfg.get("wireless_jitter", 0.05),
            environment=_SCENARIO_ENVIRONMENT.get(scenario_name, "warehouse"),
            environment_factors=net_cfg.get(
                "environment_factor", {"warehouse": 1.0, "urban": 1.15, "disaster": 1.35}
            ),
            _comm_interference_windows=windows,
        )