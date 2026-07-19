"""Peer-to-peer communication manager for domain-level Device LLM nodes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.communication.models import PeerMessage
from src.debug_trace import dlog


@dataclass
class PeerCommunicationManager:
    """
    Simulated peer network layer for Device LLM domains.

    Peers are agent-type domains (e.g. uav, vehicle, robot) — never individual
    robots. Registration, routing, delay, loss, and CQI-aware delivery live here
  only; no LLM reasoning.
    """

    registered_nodes: set[str] = field(default_factory=set)
    inboxes: dict[str, list[PeerMessage]] = field(default_factory=dict)
    pending_outbox: list[PeerMessage] = field(default_factory=list)

    # Agent-level topology (legacy / CQM input)
    cqi_matrix: np.ndarray | None = None
    agent_id_to_idx: dict[str, int] = field(default_factory=dict)

    # Domain-level topology (primary for decentralized mode)
    domain_to_agents: dict[str, list[str]] = field(default_factory=dict)
    domain_id_to_idx: dict[str, int] = field(default_factory=dict)
    domain_cqi_matrix: np.ndarray | None = None

    base_delay_s: float = 0.01
    base_loss_rate: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    peer_messages: int = 0
    broadcast_count: int = 0
    consensus_rounds: int = 0
    consensus_latency_s: float = 0.0
    plan_merge_count: int = 0
    distributed_replanning_count: int = 0

    @property
    def is_domain_mode(self) -> bool:
        return bool(self.domain_id_to_idx)

    def register_peer(self, node_id: str) -> None:
        """Register a peer node (domain id or legacy agent id)."""
        self.registered_nodes.add(node_id)
        self.inboxes.setdefault(node_id, [])

    def register_peers(self, node_ids: list[str]) -> None:
        for nid in node_ids:
            self.register_peer(nid)

    def register_domain_peers(self, domain_ids: list[str]) -> None:
        """Register Device LLM domain peers only."""
        for domain_id in domain_ids:
            self.register_peer(domain_id)

    def set_topology(
        self,
        agent_ids: list[str],
        cqi_matrix: np.ndarray,
        base_loss_rate: float = 0.0,
    ) -> None:
        """Legacy agent-level topology (used by CQM input path)."""
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        self.cqi_matrix = cqi_matrix
        self.base_loss_rate = base_loss_rate

    def set_domain_topology(
        self,
        domain_to_agents: dict[str, list[str]],
        agent_ids: list[str],
        cqi_matrix: np.ndarray,
        base_loss_rate: float = 0.0,
    ) -> None:
        """
        Build domain-level CQI matrix by aggregating inter-agent links.

        CQI(domain_a, domain_b) = mean CQI(i, j) for all i in a, j in b.
        Intra-domain pairs are included when a == b.
        """
        self.domain_to_agents = {k: list(v) for k, v in domain_to_agents.items()}
        domains = sorted(self.domain_to_agents.keys())
        self.domain_id_to_idx = {d: i for i, d in enumerate(domains)}
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        self.cqi_matrix = cqi_matrix
        self.base_loss_rate = base_loss_rate

        n_domains = len(domains)
        dmat = np.zeros((n_domains, n_domains))
        for i, di in enumerate(domains):
            agents_i = self.domain_to_agents.get(di, [])
            idx_i = [self.agent_id_to_idx[a] for a in agents_i if a in self.agent_id_to_idx]
            for j, dj in enumerate(domains):
                agents_j = self.domain_to_agents.get(dj, [])
                idx_j = [self.agent_id_to_idx[a] for a in agents_j if a in self.agent_id_to_idx]
                if not idx_i or not idx_j:
                    dmat[i, j] = 0.0
                    continue
                values = [float(cqi_matrix[ii, jj]) for ii in idx_i for jj in idx_j]
                dmat[i, j] = float(np.mean(values)) if values else 0.0
        self.domain_cqi_matrix = dmat

    def _resolve_link_cqi(self, sender: str, receiver: str) -> float:
        """Resolve CQI using domain matrix when both ends are domains."""
        if (
            self.domain_cqi_matrix is not None
            and sender in self.domain_id_to_idx
            and receiver in self.domain_id_to_idx
        ):
            si = self.domain_id_to_idx[sender]
            ri = self.domain_id_to_idx[receiver]
            return float(self.domain_cqi_matrix[si, ri])
        if self.cqi_matrix is not None:
            si = self.agent_id_to_idx.get(sender)
            ri = self.agent_id_to_idx.get(receiver)
            if si is not None and ri is not None:
                return float(self.cqi_matrix[si, ri])
        return 1.0

    def _delivery_succeeds(self, sender: str, receiver: str) -> bool:
        cqi = self._resolve_link_cqi(sender, receiver)
        if cqi <= 0.0:
            return False
        loss = self.base_loss_rate + (1.0 - cqi) * 0.5
        return self.rng.random() >= loss

    def _delivery_delay(self, sender: str, receiver: str) -> float:
        cqi = max(self._resolve_link_cqi(sender, receiver), 0.01)
        return self.base_delay_s / cqi

    def send_message(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        payload: dict[str, Any],
        ttl: int = 5,
    ) -> bool:
        """Route a unicast message between Device LLM domains; returns True if delivered."""
        if receiver not in self.registered_nodes:
            return False
        if not self._delivery_succeeds(sender, receiver):
            return False
        msg = PeerMessage(
            sender=sender,
            receiver=receiver,
            timestamp=time.time() + self._delivery_delay(sender, receiver),
            message_type=message_type,
            payload=payload,
            ttl=ttl,
        )
        self.inboxes[receiver].append(msg)
        self.peer_messages += 1
        dlog("peer", "send_message", sender=sender, receiver=receiver, type=message_type)
        return True

    def broadcast(
        self,
        sender: str,
        message_type: str,
        payload: dict[str, Any],
        ttl: int = 5,
    ) -> int:
        """Broadcast to all registered domain peers except sender."""
        delivered = 0
        for receiver in self.registered_nodes:
            if receiver != sender:
                if self.send_message(sender, receiver, message_type, payload, ttl):
                    delivered += 1
        self.broadcast_count += 1
        dlog("peer", "broadcast", sender=sender, delivered=delivered)
        return delivered

    def multicast(
        self,
        sender: str,
        receivers: list[str],
        message_type: str,
        payload: dict[str, Any],
        ttl: int = 5,
    ) -> int:
        """Send to a specific subset of domain peers."""
        delivered = 0
        for receiver in receivers:
            if receiver != sender and self.send_message(
                sender, receiver, message_type, payload, ttl
            ):
                delivered += 1
        return delivered

    def receive_messages(self, node_id: str) -> list[PeerMessage]:
        """Drain and return all messages for a domain peer."""
        msgs = self.inboxes.get(node_id, [])
        self.inboxes[node_id] = []
        return msgs

    def peek_messages(self, node_id: str) -> list[PeerMessage]:
        return list(self.inboxes.get(node_id, []))

    def pending_messages_all(self) -> dict[str, list[dict[str, Any]]]:
        return {
            nid: [m.to_dict() for m in msgs]
            for nid, msgs in self.inboxes.items()
            if msgs
        }

    def restore_pending_messages(self, pending: dict[str, list[dict[str, Any]]]) -> None:
        for nid, msgs in pending.items():
            self.inboxes.setdefault(nid, [])
            self.inboxes[nid].extend(PeerMessage.from_dict(m) for m in msgs)

    def record_consensus_round(self, latency_s: float) -> None:
        self.consensus_rounds += 1
        self.consensus_latency_s += latency_s

    def record_plan_merge(self) -> None:
        self.plan_merge_count += 1

    def record_distributed_replanning(self) -> None:
        self.distributed_replanning_count += 1

    def metrics_snapshot(self) -> dict[str, float | int]:
        return {
            "peer_messages": self.peer_messages,
            "broadcast_count": self.broadcast_count,
            "consensus_rounds": self.consensus_rounds,
            "consensus_latency": self.consensus_latency_s,
            "plan_merge_count": self.plan_merge_count,
            "distributed_replanning_count": self.distributed_replanning_count,
        }

    def reset_metrics(self) -> None:
        self.peer_messages = 0
        self.broadcast_count = 0
        self.consensus_rounds = 0
        self.consensus_latency_s = 0.0
        self.plan_merge_count = 0
        self.distributed_replanning_count = 0
