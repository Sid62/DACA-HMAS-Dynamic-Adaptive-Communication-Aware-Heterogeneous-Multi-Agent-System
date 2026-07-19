"""Distributed communication data structures for domain-level multi-Device-LLM architecture."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.env.agents import AgentFleet, AgentState


@dataclass
class SharedPlan:
    """Coalition-level plan synchronized across peer Device LLM domains."""

    coalition_id: int
    version: int = 0
    leader: str = ""
    leader_domain: str = ""
    subtasks: list[str] = field(default_factory=list)
    agent_assignments: dict[str, str] = field(default_factory=dict)
    timestamps: dict[str, float] = field(default_factory=dict)

    def bump_version(self, event: str = "update") -> None:
        self.version += 1
        self.timestamps[event] = time.time()

    @property
    def coordinating_domain(self) -> str:
        """Device LLM domain responsible for this coalition."""
        return self.leader_domain or self.leader

    def to_dict(self) -> dict[str, Any]:
        return {
            "coalition_id": self.coalition_id,
            "version": self.version,
            "leader": self.leader_domain or self.leader,
            "leader_domain": self.leader_domain or self.leader,
            "subtasks": list(self.subtasks),
            "agent_assignments": dict(self.agent_assignments),
            "timestamps": dict(self.timestamps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedPlan:
        leader_domain = str(data.get("leader_domain", data.get("leader", "")))
        return cls(
            coalition_id=int(data.get("coalition_id", 0)),
            version=int(data.get("version", 0)),
            leader=leader_domain,
            leader_domain=leader_domain,
            subtasks=list(data.get("subtasks", [])),
            agent_assignments=dict(data.get("agent_assignments", {})),
            timestamps=dict(data.get("timestamps", {})),
        )


@dataclass
class PeerMessage:
    """Message exchanged between peer Device LLM domain nodes (not individual agents)."""

    sender: str
    receiver: str
    timestamp: float
    message_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ttl: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "timestamp": self.timestamp,
            "message_type": self.message_type,
            "payload": self.payload,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PeerMessage:
        return cls(
            sender=str(data["sender"]),
            receiver=str(data["receiver"]),
            timestamp=float(data.get("timestamp", time.time())),
            message_type=str(data.get("message_type", "generic")),
            payload=dict(data.get("payload", {})),
            ttl=int(data.get("ttl", 5)),
        )


@dataclass
class NodeState:
    """
    Per-Device-LLM-domain state (one instance per agent type, NOT per robot).

    node_id is the agent-type domain key (e.g. value of AgentType enum),
    discovered dynamically from the fleet — never hardcoded.
    """

    node_id: str
    managed_agent_ids: list[str] = field(default_factory=list)
    local_observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    belief_state: dict[str, Any] = field(default_factory=dict)
    neighbor_plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    received_messages: list[PeerMessage] = field(default_factory=list)
    shared_plan: dict[str, Any] = field(default_factory=dict)
    shared_plan_version: int = 0
    current_task: str | None = None
    # Legacy single-agent field — populated for backward-compatible deserialization
    local_observation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "managed_agent_ids": list(self.managed_agent_ids),
            "local_observations": self.local_observations,
            "belief_state": self.belief_state,
            "neighbor_plans": self.neighbor_plans,
            "received_messages": [m.to_dict() for m in self.received_messages],
            "shared_plan": self.shared_plan,
            "shared_plan_version": self.shared_plan_version,
            "current_task": self.current_task,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeState:
        local_observations = dict(data.get("local_observations", {}))
        legacy_obs = data.get("local_observation")
        node_id = str(data["node_id"])
        managed = list(data.get("managed_agent_ids", []))
        if legacy_obs and not local_observations:
            agent_id = legacy_obs.get("agent_id", node_id)
            local_observations = {agent_id: legacy_obs}
            if not managed:
                managed = [agent_id]
        return cls(
            node_id=node_id,
            managed_agent_ids=managed,
            local_observations=local_observations,
            belief_state=dict(data.get("belief_state", {})),
            neighbor_plans=dict(data.get("neighbor_plans", {})),
            received_messages=[
                PeerMessage.from_dict(m) for m in data.get("received_messages", [])
            ],
            shared_plan=dict(data.get("shared_plan", {})),
            shared_plan_version=int(data.get("shared_plan_version", 0)),
            current_task=data.get("current_task"),
            local_observation=legacy_obs or {},
        )


def discover_agent_type_domains(fleet: AgentFleet) -> dict[str, list[str]]:
    """
    Group fleet agents by agent type domain.

    Returns mapping: agent_type_value -> [agent_id, ...]
    Agent types are discovered from AgentState.agent_type — not hardcoded.
    """
    domains: dict[str, list[str]] = {}
    for agent in fleet.agents:
        domain_key = agent.agent_type.value
        domains.setdefault(domain_key, []).append(agent.agent_id)
    return domains


def domain_for_agent(agent: AgentState) -> str:
    """Return the Device LLM domain key for a single agent."""
    return agent.agent_type.value


def dominant_domain_for_coalition(
    member_agent_ids: list[str],
    fleet: AgentFleet,
) -> str:
    """
    Select the coordinating Device LLM domain for a coalition.

    Policy: agent type with the most members in the coalition wins;
    ties broken lexicographically by domain key.
    """
    counts: dict[str, int] = {}
    id_to_agent = {a.agent_id: a for a in fleet.agents}
    for aid in member_agent_ids:
        agent = id_to_agent.get(aid)
        if agent is None:
            continue
        domain = agent.agent_type.value
        counts[domain] = counts.get(domain, 0) + 1
    if not counts:
        return ""
    max_count = max(counts.values())
    candidates = sorted(d for d, c in counts.items() if c == max_count)
    return candidates[0]


def domains_in_coalition(
    member_agent_ids: list[str],
    fleet: AgentFleet,
) -> list[str]:
    """Unique Device LLM domains present in a coalition."""
    id_to_agent = {a.agent_id: a for a in fleet.agents}
    domains: set[str] = set()
    for aid in member_agent_ids:
        agent = id_to_agent.get(aid)
        if agent is not None:
            domains.add(agent.agent_type.value)
    return sorted(domains)
