"""Delta State Transfer Manager for DACA-HMAS.

Implements delta state handoff serialization.
Instead of serializing the entire world state during architecture handoff
or LLM prompt construction, transfers ONLY changed entities (changed tasks,
changed coalitions, moved agents, changed link CQIs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.env.agents import AgentFleet
from src.env.scenarios import Subtask
from src.handoff.snapshot import GlobalSnapshot, AgentSnapshot


@dataclass
class DeltaStateSnapshot:
    """Compact delta state representation for handoff and LLM prompt context."""
    timestep: int
    mode_before: int
    mode_after: int
    changed_agents: list[dict[str, Any]] = field(default_factory=list)
    changed_coalitions: list[dict[str, Any]] = field(default_factory=list)
    newly_completed_subtasks: list[str] = field(default_factory=list)
    cqi_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "mode_before": self.mode_before,
            "mode_after": self.mode_after,
            "changed_agents": self.changed_agents,
            "changed_coalitions": self.changed_coalitions,
            "newly_completed_subtasks": self.newly_completed_subtasks,
            "cqi_delta": self.cqi_delta,
        }


class DeltaStateTransferManager:
    """Computes and serializes delta state handoffs."""

    def __init__(self, position_displacement_threshold: float = 5.0):
        self.displacement_threshold = position_displacement_threshold
        self._last_snapshot: GlobalSnapshot | None = None

    def compute_delta_snapshot(
        self,
        current_snapshot: GlobalSnapshot,
    ) -> DeltaStateSnapshot:
        """Compute delta snapshot comparing current snapshot against previous baseline."""
        if self._last_snapshot is None:
            self._last_snapshot = current_snapshot
            return DeltaStateSnapshot(
                timestep=current_snapshot.timestep,
                mode_before=current_snapshot.mode_before,
                mode_after=current_snapshot.mode_after,
                changed_agents=[
                    {
                        "agent_id": a.agent_id,
                        "position": a.position,
                        "coalition_id": a.coalition_id,
                    }
                    for a in current_snapshot.agents
                ],
                changed_coalitions=current_snapshot.coalitions,
                newly_completed_subtasks=current_snapshot.completed_subtasks,
                cqi_delta=0.0,
            )

        prev = self._last_snapshot
        prev_agents = {a.agent_id: a for a in prev.agents}

        changed_agents = []
        for curr_agent in current_snapshot.agents:
            prev_agent = prev_agents.get(curr_agent.agent_id)
            if prev_agent is None:
                changed_agents.append({
                    "agent_id": curr_agent.agent_id,
                    "position": curr_agent.position,
                    "coalition_id": curr_agent.coalition_id,
                })
            else:
                dx = curr_agent.position[0] - prev_agent.position[0]
                dy = curr_agent.position[1] - prev_agent.position[1]
                if (dx * dx + dy * dy) ** 0.5 > self.displacement_threshold or curr_agent.coalition_id != prev_agent.coalition_id:
                    changed_agents.append({
                        "agent_id": curr_agent.agent_id,
                        "position": curr_agent.position,
                        "coalition_id": curr_agent.coalition_id,
                    })

        newly_completed = list(set(current_snapshot.completed_subtasks) - set(prev.completed_subtasks))
        changed_coalitions = current_snapshot.coalitions if current_snapshot.coalitions != prev.coalitions else []

        delta = DeltaStateSnapshot(
            timestep=current_snapshot.timestep,
            mode_before=current_snapshot.mode_before,
            mode_after=current_snapshot.mode_after,
            changed_agents=changed_agents,
            changed_coalitions=changed_coalitions,
            newly_completed_subtasks=newly_completed,
        )

        self._last_snapshot = current_snapshot
        return delta
