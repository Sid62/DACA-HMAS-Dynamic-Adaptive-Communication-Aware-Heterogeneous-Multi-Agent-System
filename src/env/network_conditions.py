"""Synthetic network condition injectors for communication profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


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

    def loss_rate_at(self, t: int) -> float:
        if self.profile == NetworkProfile.STABLE:
            return self.base_loss_rate
        if self.profile == NetworkProfile.GRADUAL:
            progress = t / max(self.total_steps, 1)
            return self.base_loss_rate + 0.15 * progress
        if self.profile == NetworkProfile.SUDDEN:
            return self.base_loss_rate + (0.20 if t > self.total_steps * 0.4 else 0.0)
        if self.profile == NetworkProfile.OSCILLATORY:
            return self.base_loss_rate + 0.10 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 50))
        return self.base_loss_rate

    def latency_at(self, t: int) -> float:
        loss = self.loss_rate_at(t)
        base = 0.01 + self.base_delay_prob * 0.5
        return base + loss * 1.5 + self.rng.uniform(0, 0.02)

    def bandwidth_at(self, t: int) -> float:
        loss = self.loss_rate_at(t)
        return max(0.0, 1.0 - loss - self.base_delay_prob * 0.3)

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
    ) -> NetworkConditionGenerator:
        sc = thresholds.get("scenarios", {}).get(scenario_name, {})
        prof = NetworkProfile(profile) if profile != "stable" else NetworkProfile.STABLE
        if scenario_name == "logistics" and profile == "stable":
            prof = NetworkProfile.STABLE
        return cls(
            profile=prof,
            base_delay_prob=sc.get("comm_delay_prob", 0.0),
            base_loss_rate=sc.get("packet_loss_rate", 0.0),
            total_steps=total_steps,
            rng=np.random.default_rng(seed),
        )
