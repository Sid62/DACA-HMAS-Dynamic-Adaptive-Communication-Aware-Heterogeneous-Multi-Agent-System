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
    link_stats: dict[tuple[int, int], NodeStats] = field(default_factory=dict)
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
        stats = self.node_stats.get(node_id)
        if not stats or not stats.delivery_outcomes:
            return 0.0
        return 1.0 - (sum(stats.delivery_outcomes) / len(stats.delivery_outcomes))

    def normalized_latency(self, node_id: int) -> float:
        """Eq 17b: tau_hat_n(t)."""
        stats = self.node_stats.get(node_id)
        if not stats or not stats.latencies:
            return 0.0
        tau = float(np.mean(stats.latencies))
        denom = self.tau_max - self.tau_min
        if denom <= 0:
            return 0.0
        return float(np.clip((tau - self.tau_min) / denom, 0.0, 1.0))

    def bandwidth_availability(self, node_id: int) -> float:
        """Eq 17c: B_n(t)."""
        stats = self.node_stats.get(node_id)
        if not stats or not stats.bytes_capacity:
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
        return float(np.clip(w1 * (1 - ln) + w2 * (1 - tau_hat) + w3 * bn, 0.0, 1.0))

    def link_packet_loss_rate(self, sender_id: int, receiver_id: int) -> float:
        """Packet loss rate L_ij(t) on link sender -> receiver."""
        stats = self.link_stats.get((sender_id, receiver_id))
        if not stats or not stats.delivery_outcomes:
            return 0.0
        return 1.0 - (sum(stats.delivery_outcomes) / len(stats.delivery_outcomes))

    def link_normalized_latency(self, sender_id: int, receiver_id: int) -> float:
        """Normalized latency on link sender -> receiver."""
        stats = self.link_stats.get((sender_id, receiver_id))
        if not stats or not stats.latencies:
            return 0.0
        tau = float(np.mean(stats.latencies))
        denom = self.tau_max - self.tau_min
        if denom <= 0:
            return 0.0
        return float(np.clip((tau - self.tau_min) / denom, 0.0, 1.0))

    def link_bandwidth_availability(self, sender_id: int, receiver_id: int) -> float:
        """Bandwidth availability on link sender -> receiver."""
        stats = self.link_stats.get((sender_id, receiver_id))
        if not stats or not stats.bytes_capacity:
            return 1.0
        delivered = sum(stats.bytes_delivered)
        capacity = sum(stats.bytes_capacity)
        if capacity <= 0:
            return 1.0
        return float(np.clip(delivered / capacity, 0.0, 1.0))

    def link_cqi(self, sender_id: int, receiver_id: int) -> float | None:
        """Directional CQI on link sender -> receiver if link observations exist."""
        if (sender_id, receiver_id) not in self.link_stats:
            return None
        w1, w2, w3 = self.weights
        ln = self.link_packet_loss_rate(sender_id, receiver_id)
        tau_hat = self.link_normalized_latency(sender_id, receiver_id)
        bn = self.link_bandwidth_availability(sender_id, receiver_id)
        return float(np.clip(w1 * (1 - ln) + w2 * (1 - tau_hat) + w3 * bn, 0.0, 1.0))

    def system_cqi(self) -> float:
        """Eq 19: Global system-level CQI summary CQI(t) for switching/global decisions."""
        if self.n_nodes == 0:
            return 1.0
        return sum(self.node_cqi(n) for n in range(self.n_nodes)) / self.n_nodes

    def update_from_network(
        self,
        node_id: int,
        net: NetworkState,
        receiver_id: int | None = None,
    ) -> None:
        """Ingest network observation for a node and optionally a specific link."""
        if node_id not in self.node_stats:
            self.node_stats[node_id] = NodeStats(
                bytes_delivered=deque(maxlen=self.bandwidth_window),
                bytes_capacity=deque(maxlen=self.bandwidth_window),
                delivery_outcomes=deque(maxlen=self.packet_loss_window),
            )
        stats = self.node_stats[node_id]
        for _ in range(max(net.msg_sent, 0)):
            stats.delivery_outcomes.append(1 if net.ack_received > 0 else 0)
        stats.latencies.append(net.latency)
        stats.bytes_delivered.append(net.bytes_delivered)
        stats.bytes_capacity.append(net.bytes_capacity)

        if receiver_id is not None:
            self.update_link(node_id, receiver_id, net)

    def update_link(self, sender_id: int, receiver_id: int, net: NetworkState) -> None:
        """Record directional communication outcome on link sender -> receiver."""
        pair = (sender_id, receiver_id)
        if pair not in self.link_stats:
            self.link_stats[pair] = NodeStats(
                bytes_delivered=deque(maxlen=self.bandwidth_window),
                bytes_capacity=deque(maxlen=self.bandwidth_window),
                delivery_outcomes=deque(maxlen=self.packet_loss_window),
            )
        lstats = self.link_stats[pair]
        for _ in range(max(net.msg_sent, 0)):
            lstats.delivery_outcomes.append(1 if net.ack_received > 0 else 0)
        lstats.latencies.append(net.latency)
        lstats.bytes_delivered.append(net.bytes_delivered)
        lstats.bytes_capacity.append(net.bytes_capacity)

    def update_pairwise(
        self,
        distance_matrix: np.ndarray,
        c1: float,
        network: Any | None = None,
        directional: bool | None = None,
        step: int = 0,
    ) -> np.ndarray:
        """Build N x N pairwise CQI matrix Q(t) (Eq 23, 25).

        For each pair (i, j):
          - If i == j: Q[i, j] = 1.0 (self-loop).
          - If distance_matrix[i, j] > c1: Q[i, j] = 0.0 (out of range / disconnected).
          - If distance_matrix[i, j] <= c1:
            Q[i, j] reflects the quality of that specific link i -> j:
              1. Physical channel quality: uses network.link_channel_quality(step, d_ij, i, j)
                 when a network model is provided, capturing profile dynamics, scenario
                 shadowing, interference, and log-distance path loss. Otherwise falls back to
                 distance_quality(d_ij, c1).
              2. Endpoint observations:
                 - If directional per-link stats exist for (i, j), use link_cqi(i, j).
                 - If directional mode is explicitly active, use asymmetric transmitter/receiver weighting.
                 - Otherwise (symmetric reciprocal channel), both endpoints must be functional,
                   bottlenecked by min(node_cqi(i), node_cqi(j)).
        """
        from src.env.network_model import distance_quality

        n = distance_matrix.shape[0]
        q = np.zeros((n, n), dtype=float)

        good_range = getattr(network, "good_range", min(20.0, 0.4 * c1))
        medium_range = getattr(network, "medium_range", min(40.0, 0.8 * c1))

        node_cqis = [self.node_cqi(i) if i in self.node_stats else 1.0 for i in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    q[i, j] = 1.0
                    continue

                d_ij = float(distance_matrix[i, j])
                if d_ij > c1:
                    q[i, j] = 0.0
                    continue

                # 1. Physical channel attenuation on this specific link
                if network is not None and hasattr(network, "link_channel_quality"):
                    q_channel = network.link_channel_quality(
                        step, d_ij, sender_id=i, receiver_id=j
                    )
                else:
                    q_channel = distance_quality(
                        d_ij,
                        communication_range=c1,
                        good_range=good_range,
                        medium_range=medium_range,
                    )

                # 2. Endpoint communication quality
                link_q = self.link_cqi(i, j)
                if link_q is not None:
                    # Direct directional link observation exists
                    q_endpoint = link_q
                elif directional:
                    # Directional mode with sender (60%) and receiver (40%) weighting
                    q_endpoint = 0.6 * node_cqis[i] + 0.4 * node_cqis[j]
                else:
                    # Reciprocal symmetric wireless channel: communication requires both
                    # endpoints to be operational; bottleneck is min(node_cqi(i), node_cqi(j)).
                    q_endpoint = min(node_cqis[i], node_cqis[j])

                q[i, j] = float(np.clip(q_endpoint * q_channel, 0.0, 1.0))

        self.pairwise_cqi = q
        return q

    def get_cqi_matrix(self) -> np.ndarray:
        if self.pairwise_cqi is None:
            return np.ones((self.n_nodes, self.n_nodes))
        return self.pairwise_cqi
