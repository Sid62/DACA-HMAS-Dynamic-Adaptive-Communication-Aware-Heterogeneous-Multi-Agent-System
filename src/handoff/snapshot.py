"""State snapshot handoff (Eqs 28-29)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.env.agents import AgentFleet, Position
from src.env.scenarios import Subtask


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
        )


def capture_snapshot(
    fleet: AgentFleet,
    subtasks: list[Subtask],
    coalitions: list[dict],
    timestep: int,
    mode_before: int,
    mode_after: int,
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
    return GlobalSnapshot(
        timestep=timestep,
        mode_before=mode_before,
        mode_after=mode_after,
        agents=agents,
        coalitions=coalitions,
        subtask_ids=[s.subtask_id for s in subtasks],
        completed_subtasks=[s.subtask_id for s in subtasks if s.completed],
    )


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


def slice_for_device(snapshot: GlobalSnapshot, agent_id: str) -> dict[str, Any]:
    """Per-Device-LLM slice of G(t*)."""
    agent_snap = next((a for a in snapshot.agents if a.agent_id == agent_id), None)
    if agent_snap is None:
        return {}
    return {
        "agent": {
            "agent_id": agent_snap.agent_id,
            "position": agent_snap.position,
            "assigned_subtasks": agent_snap.assigned_subtasks,
            "remaining_waypoints": agent_snap.remaining_waypoints,
        },
        "coalitions": snapshot.coalitions,
        "mode": snapshot.mode_after,
        "timestep": snapshot.timestep,
    }
