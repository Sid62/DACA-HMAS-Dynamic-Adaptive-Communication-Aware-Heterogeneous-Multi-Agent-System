"""Simplified NMPC for global trajectory planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.env.agents import AgentFleet, Position, dist
from src.control.pid import PIDController


@dataclass
class NMPCController:
    horizon: int = 5
    pid: PIDController | None = None

    def __post_init__(self) -> None:
        if self.pid is None:
            self.pid = PIDController()

    def plan_waypoint(
        self, fleet: AgentFleet, agent_id: str, target: Position
    ) -> Position:
        agent = fleet.get_agent(agent_id)
        best_wp = target
        best_cost = float("inf")
        for h in range(1, self.horizon + 1):
            alpha = h / self.horizon
            wp = Position(
                x=agent.position.x + alpha * (target.x - agent.position.x),
                y=agent.position.y + alpha * (target.y - agent.position.y),
            )
            cost = dist(agent.position, wp)
            for other in fleet.agents:
                if other.agent_id != agent_id:
                    cost += max(0, 5.0 - dist(wp, other.position))
            if cost < best_cost:
                best_cost = cost
                best_wp = wp
        return best_wp

    def step(
        self,
        fleet: AgentFleet,
        assignments: dict[str, str],
        targets: dict[str, Position],
    ) -> None:
        planned: dict[str, Position] = {}
        for agent_id, subtask_id in assignments.items():
            if subtask_id in targets:
                planned[agent_id] = self.plan_waypoint(
                    fleet, agent_id, targets[subtask_id]
                )
        self.pid.step(fleet, assignments, planned)

    @property
    def active(self) -> bool:
        return True
