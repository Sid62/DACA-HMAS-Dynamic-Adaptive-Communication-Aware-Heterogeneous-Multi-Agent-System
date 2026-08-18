"""Communication Quality Monitor (Eqs 17-19)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.env.network_conditions import NetworkState


@dataclass
class NodeStats:

    latencies: deque = field(default_factory=lambda: deque(maxlen=50))
    bytes_delivered: deque = field(default_factory=lambda: deque(maxlen=10))
    bytes_capacity: deque = field(default_factory=lambda: deque(maxlen=10))
    # Windowed packet delivery outcomes -- replaces the lifetime-cumulative
    # msg_sent/ack_received counters below, which could never reflect a
    # channel's recovery once degraded (Eq 17a requires L_n(t), an
    # INSTANTANEOUS/recent rate, not an all-time average since t=0).
    delivery_outcomes: deque = field(default_factory=lambda: deque(maxlen=20))



@dataclass
class CommunicationQualityMonitor:
    """Passive CQM using existing instruction-feedback traffic."""

    weights: tuple[float, float, float] = (0.4, 0.35, 0.25)
    tau_min: float = 0.01
    tau_max: float = 2.0
    bandwidth_window: int = 10
    packet_loss_window: int = 20
    n_nodes: int = 1
    node_stats: dict[int, NodeStats] = field(default_factory=dict)
    pairwise_cqi: np.ndarray | None = None

    def __post_init__(self) -> None:
        for n in range(self.n_nodes):
            self.node_stats[n] = NodeStats(
                bytes_delivered=deque(maxlen=self.bandwidth_window),
                bytes_capacity=deque(maxlen=self.bandwidth_window),
                delivery_outcomes=deque(maxlen=self.packet_loss_window),
            )
        self.pairwise_cqi = np.ones((self.n_nodes, self.n_nodes))

    @classmethod
    def from_config(cls, thresholds: dict[str, Any], n_nodes: int) -> CommunicationQualityMonitor:
        w = thresholds.get("cqi_weights", {})
        lat = thresholds.get("latency", {})
        return cls(
            weights=(w.get("w1", 0.4), w.get("w2", 0.35), w.get("w3", 0.25)),
            tau_min=lat.get("tau_min", 0.01),
            tau_max=lat.get("tau_max", 2.0),
            bandwidth_window=thresholds.get("bandwidth_window", 10),
            packet_loss_window=thresholds.get("packet_loss_window", 20),
            n_nodes=n_nodes,
        )

    def packet_loss_rate(self, node_id: int) -> float:
        """Eq 17a: L_n(t)."""
        stats = self.node_stats[node_id]
        if not stats.delivery_outcomes:
            return 0.0
        return 1.0 - (sum(stats.delivery_outcomes) / len(stats.delivery_outcomes))

    def normalized_latency(self, node_id: int) -> float:
        """Eq 17b: tau_hat_n(t)."""
        stats = self.node_stats[node_id]
        if not stats.latencies:
            return 0.0
        tau = float(np.mean(stats.latencies))
        denom = self.tau_max - self.tau_min
        if denom <= 0:
            return 0.0
        return float(np.clip((tau - self.tau_min) / denom, 0.0, 1.0))

    def bandwidth_availability(self, node_id: int) -> float:
        """Eq 17c: B_n(t)."""
        stats = self.node_stats[node_id]
        if not stats.bytes_capacity:
            return 1.0
        delivered = sum(stats.bytes_delivered)
        capacity = sum(stats.bytes_capacity)
        if capacity <= 0:
            return 1.0
        return float(np.clip(delivered / capacity, 0.0, 1.0))

    def node_cqi(self, node_id: int) -> float:
        """Eq 18: CQI_n(t)."""
        w1, w2, w3 = self.weights
        ln = self.packet_loss_rate(node_id)
        tau_hat = self.normalized_latency(node_id)
        bn = self.bandwidth_availability(node_id)
        return w1 * (1 - ln) + w2 * (1 - tau_hat) + w3 * bn

    def system_cqi(self) -> float:
        """Eq 19: CQI(t)."""
        if self.n_nodes == 0:
            return 1.0
        return sum(self.node_cqi(n) for n in range(self.n_nodes)) / self.n_nodes

    def update_from_network(self, node_id: int, net: NetworkState) -> None:
        stats = self.node_stats[node_id]
        # Record each message's outcome (1=delivered, 0=lost) into a bounded
        # window instead of accumulating lifetime totals.
        for _ in range(max(net.msg_sent, 0)):
            stats.delivery_outcomes.append(1 if net.ack_received > 0 else 0)
        stats.latencies.append(net.latency)
        stats.bytes_delivered.append(net.bytes_delivered)
        stats.bytes_capacity.append(net.bytes_capacity)

    def update_pairwise(self, distance_matrix: np.ndarray, c1: float) -> np.ndarray:
        """Build N x N pairwise CQI matrix Q(t)."""
        n = distance_matrix.shape[0]
        q = np.zeros((n, n))
        sys_cqi = self.system_cqi()
        for i in range(n):
            for j in range(n):
                if i == j:
                    q[i, j] = 1.0
                elif distance_matrix[i, j] <= c1:
                    q[i, j] = sys_cqi
                else:
                    q[i, j] = 0.0
        self.pairwise_cqi = q
        return q

    def get_cqi_matrix(self) -> np.ndarray:
        if self.pairwise_cqi is None:
            return np.ones((self.n_nodes, self.n_nodes))
        return self.pairwise_cqi
