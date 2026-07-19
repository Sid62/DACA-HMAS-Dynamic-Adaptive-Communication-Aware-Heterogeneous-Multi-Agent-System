"""State snapshot handoff (Eqs 28-29) with domain-level Device LLM support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.communication.models import NodeState, SharedPlan, domain_for_agent
from src.env.agents import AgentFleet, Position
from src.env.scenarios import Subtask


def _iter_domain_device_clients(
    device_llms: dict[str, Any] | None,
) -> list[tuple[str, Any]]:
    """Yield domain Device LLM clients, excluding centralized dispatch helpers."""
    if not device_llms:
        return []
    out: list[tuple[str, Any]] = []
    for key, client in device_llms.items():
        if key.startswith("_") or key == "dispatch":
            continue
        out.append((key, client))
    return out


def _client_for_node_state(
    node_id: str,
    ns_data: dict[str, Any],
    device_llms: dict[str, Any],
) -> Any | None:
    """Resolve domain client from domain id or legacy per-agent snapshot."""
    direct = device_llms.get(node_id)
    if direct is not None and not node_id.startswith("_") and node_id != "dispatch":
        return direct

    domain_key = str(ns_data.get("node_id", node_id))
    domain_client = device_llms.get(domain_key)
    if domain_client is not None:
        return domain_client

    for key, client in _iter_domain_device_clients(device_llms):
        managed = list(getattr(client, "managed_agent_ids", []))
        if client.node_state is not None:
            managed = managed or list(client.node_state.managed_agent_ids)
        if node_id in managed or domain_key in managed:
            return client
        if key == domain_key:
            return client
    return None


def _apply_node_state(client: Any, ns_data: dict[str, Any]) -> None:
    """Restore or merge NodeState into a domain Device LLM client."""
    restored = NodeState.from_dict(ns_data)
    if client.node_state is None:
        client.node_state = restored
        if getattr(client, "managed_agent_ids", None) is not None:
            client.managed_agent_ids = list(restored.managed_agent_ids)
        return

    if restored.node_id == client.node_id:
        client.node_state = restored
        if getattr(client, "managed_agent_ids", None) is not None:
            client.managed_agent_ids = list(restored.managed_agent_ids)
        return

    # Legacy per-robot snapshot -> merge into domain-level state
    client.node_state.local_observations.update(restored.local_observations)
    if restored.local_observation:
        agent_id = str(restored.local_observation.get("agent_id", restored.node_id))
        client.node_state.local_observations[agent_id] = restored.local_observation
    merged_managed = set(client.node_state.managed_agent_ids)
    merged_managed.update(restored.managed_agent_ids)
    merged_managed.update(client.node_state.local_observations.keys())
    client.node_state.managed_agent_ids = sorted(merged_managed)
    if getattr(client, "managed_agent_ids", None) is not None:
        client.managed_agent_ids = list(client.node_state.managed_agent_ids)
    client.node_state.shared_plan_version = max(
        client.node_state.shared_plan_version,
        restored.shared_plan_version,
    )
    if restored.shared_plan:
        client.node_state.shared_plan = restored.shared_plan
    if restored.current_task is not None:
        client.node_state.current_task = restored.current_task
    client.node_state.neighbor_plans.update(restored.neighbor_plans)
    client.node_state.received_messages.extend(restored.received_messages)
    client.node_state.belief_state.update(restored.belief_state)


@dataclass
class AgentSnapshot:
    agent_id: str
    position: list[float]
    assigned_subtasks: list[str]
    completed_subtasks: list[str]
    remaining_waypoints: list[list[float]]
    coalition_id: int | None


@dataclass
class GlobalSnapshot:
    """Eq 28: G(t*) global state at switch time."""

    timestep: int
    mode_before: int
    mode_after: int
    agents: list[AgentSnapshot] = field(default_factory=list)
    coalitions: list[dict] = field(default_factory=list)
    subtask_ids: list[str] = field(default_factory=list)
    completed_subtasks: list[str] = field(default_factory=list)
    shared_plans: list[dict] = field(default_factory=list)
    node_states: list[dict] = field(default_factory=list)
    pending_messages: dict[str, list[dict]] = field(default_factory=dict)
    device_domains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "mode_before": self.mode_before,
            "mode_after": self.mode_after,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "position": a.position,
                    "assigned_subtasks": a.assigned_subtasks,
                    "completed_subtasks": a.completed_subtasks,
                    "remaining_waypoints": a.remaining_waypoints,
                    "coalition_id": a.coalition_id,
                }
                for a in self.agents
            ],
            "coalitions": self.coalitions,
            "subtask_ids": self.subtask_ids,
            "completed_subtasks": self.completed_subtasks,
            "shared_plans": self.shared_plans,
            "node_states": self.node_states,
            "pending_messages": self.pending_messages,
            "device_domains": list(self.device_domains),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlobalSnapshot:
        agents = [
            AgentSnapshot(
                agent_id=a["agent_id"],
                position=a["position"],
                assigned_subtasks=a.get("assigned_subtasks", []),
                completed_subtasks=a.get("completed_subtasks", []),
                remaining_waypoints=a.get("remaining_waypoints", []),
                coalition_id=a.get("coalition_id"),
            )
            for a in data.get("agents", [])
        ]
        return cls(
            timestep=data["timestep"],
            mode_before=data["mode_before"],
            mode_after=data["mode_after"],
            agents=agents,
            coalitions=data.get("coalitions", []),
            subtask_ids=data.get("subtask_ids", []),
            completed_subtasks=data.get("completed_subtasks", []),
            shared_plans=data.get("shared_plans", []),
            node_states=data.get("node_states", []),
            pending_messages=data.get("pending_messages", {}),
            device_domains=list(data.get("device_domains", [])),
        )


def capture_snapshot(
    fleet: AgentFleet,
    subtasks: list[Subtask],
    coalitions: list[dict],
    timestep: int,
    mode_before: int,
    mode_after: int,
    shared_plans: dict[int, SharedPlan] | None = None,
    device_llms: dict[str, Any] | None = None,
    pending_messages: dict[str, list[dict]] | None = None,
) -> GlobalSnapshot:
    agents = []
    for a in fleet.agents:
        agents.append(
            AgentSnapshot(
                agent_id=a.agent_id,
                position=[a.position.x, a.position.y, a.position.z],
                assigned_subtasks=list(a.assigned_subtasks),
                completed_subtasks=list(a.completed_subtasks),
                remaining_waypoints=[
                    [w.x, w.y, w.z] for w in a.remaining_waypoints
                ],
                coalition_id=a.coalition_id,
            )
        )

    domain_clients = _iter_domain_device_clients(device_llms)
    device_domains = [key for key, _ in domain_clients]

    return GlobalSnapshot(
        timestep=timestep,
        mode_before=mode_before,
        mode_after=mode_after,
        agents=agents,
        coalitions=coalitions,
        subtask_ids=[s.subtask_id for s in subtasks],
        completed_subtasks=[s.subtask_id for s in subtasks if s.completed],
        shared_plans=[
            sp.to_dict() for sp in (shared_plans or {}).values()
        ],
        node_states=[
            client.node_state.to_dict()
            for _, client in domain_clients
            if client.node_state is not None
        ],
        pending_messages=pending_messages or {},
        device_domains=device_domains,
    )


def restore_distributed_state(
    snapshot: GlobalSnapshot,
    device_llms: dict[str, Any],
    peer_manager: Any | None = None,
    decentralized: Any | None = None,
) -> None:
    """Restore SharedPlan, domain NodeState, and pending messages after handoff."""
    if decentralized is not None and snapshot.shared_plans:
        decentralized.shared_plans = {
            int(sp["coalition_id"]): SharedPlan.from_dict(sp)
            for sp in snapshot.shared_plans
        }

    for ns_data in snapshot.node_states:
        node_id = str(ns_data.get("node_id", ""))
        client = _client_for_node_state(node_id, ns_data, device_llms)
        if client is not None:
            _apply_node_state(client, ns_data)

    if peer_manager is not None:
        if snapshot.device_domains:
            peer_manager.register_domain_peers(snapshot.device_domains)
        if snapshot.pending_messages:
            peer_manager.restore_pending_messages(snapshot.pending_messages)


def restore_snapshot(fleet: AgentFleet, snapshot: GlobalSnapshot) -> None:
    id_to_agent = {a.agent_id: a for a in fleet.agents}
    for snap in snapshot.agents:
        if snap.agent_id in id_to_agent:
            agent = id_to_agent[snap.agent_id]
            agent.position = Position(*snap.position)
            agent.assigned_subtasks = list(snap.assigned_subtasks)
            agent.completed_subtasks = list(snap.completed_subtasks)
            agent.remaining_waypoints = [
                Position(*w) for w in snap.remaining_waypoints
            ]
            agent.coalition_id = snap.coalition_id


def verify_task_preservation(
    before: GlobalSnapshot,
    after: GlobalSnapshot,
) -> bool:
    """Eq 29: task preservation constraint."""
    before_all = set(before.subtask_ids)
    after_all = set(after.subtask_ids)
    if before_all != after_all:
        return False
    before_completed = set(before.completed_subtasks)
    after_completed = set(after.completed_subtasks)
    return before_completed == after_completed


def slice_for_domain(
    snapshot: GlobalSnapshot,
    domain_id: str,
    managed_agent_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Per-Device-LLM-domain slice of G(t*)."""
    agent_filter = set(managed_agent_ids or [])
    domain_agents = [
        {
            "agent_id": a.agent_id,
            "position": a.position,
            "assigned_subtasks": a.assigned_subtasks,
            "completed_subtasks": a.completed_subtasks,
            "remaining_waypoints": a.remaining_waypoints,
            "coalition_id": a.coalition_id,
        }
        for a in snapshot.agents
        if not agent_filter or a.agent_id in agent_filter
    ]
    domain_node = next(
        (ns for ns in snapshot.node_states if ns.get("node_id") == domain_id),
        {},
    )
    return {
        "domain_id": domain_id,
        "managed_agent_ids": list(managed_agent_ids or []),
        "agents": domain_agents,
        "node_state": domain_node,
        "shared_plans": snapshot.shared_plans,
        "pending_messages": snapshot.pending_messages.get(domain_id, []),
        "coalitions": snapshot.coalitions,
        "mode": snapshot.mode_after,
        "timestep": snapshot.timestep,
    }


def slice_for_device(snapshot: GlobalSnapshot, agent_id: str) -> dict[str, Any]:
    """Per-agent slice of G(t*) — includes domain id when present in snapshot."""
    agent_snap = next((a for a in snapshot.agents if a.agent_id == agent_id), None)
    if agent_snap is None:
        return {}

    domain_id = ""
    for ns in snapshot.node_states:
        managed = ns.get("managed_agent_ids", [])
        if agent_id in managed:
            domain_id = str(ns.get("node_id", ""))
            break

    return {
        "agent": {
            "agent_id": agent_snap.agent_id,
            "position": agent_snap.position,
            "assigned_subtasks": agent_snap.assigned_subtasks,
            "remaining_waypoints": agent_snap.remaining_waypoints,
        },
        "domain_id": domain_id,
        "coalitions": snapshot.coalitions,
        "mode": snapshot.mode_after,
        "timestep": snapshot.timestep,
    }


def slice_for_agent_in_fleet(
    snapshot: GlobalSnapshot,
    fleet: AgentFleet,
    agent_id: str,
) -> dict[str, Any]:
    """Per-agent slice with explicit domain resolution from fleet metadata."""
    agent = fleet.get_agent(agent_id)
    domain_id = domain_for_agent(agent)
    domain_slice = slice_for_domain(
        snapshot,
        domain_id,
        managed_agent_ids=[
            a.agent_id
            for a in fleet.agents
            if domain_for_agent(a) == domain_id
        ],
    )
    agent_snap = next((a for a in snapshot.agents if a.agent_id == agent_id), None)
    if agent_snap is None:
        return domain_slice
    domain_slice["agent"] = {
        "agent_id": agent_snap.agent_id,
        "position": agent_snap.position,
        "assigned_subtasks": agent_snap.assigned_subtasks,
        "remaining_waypoints": agent_snap.remaining_waypoints,
    }
    return domain_slice
