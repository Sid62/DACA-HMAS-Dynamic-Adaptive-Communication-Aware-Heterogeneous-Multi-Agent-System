"""PID controller for centralized collision avoidance."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.env.agents import AgentFleet, Position, dist


@dataclass
class PIDController:
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    safe_distance: float = 5.0
    _integral: float = 0.0
    _prev_error: float = 0.0

    def compute_avoidance_offset(
        self, fleet: AgentFleet, agent_id: str, target: Position, dt: float = 0.1
    ) -> Position:
        agent = fleet.get_agent(agent_id)
        error = dist(agent.position, target)
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        correction = self.kp * error + self.ki * self._integral + self.kd * derivative

        offset_x, offset_y = 0.0, 0.0
        for other in fleet.agents:
            if other.agent_id == agent_id:
                continue
            d = dist(agent.position, other.position)
            if d < self.safe_distance and d > 0.01:
                dx = agent.position.x - other.position.x
                dy = agent.position.y - other.position.y
                scale = (self.safe_distance - d) / d
                offset_x += dx * scale
                offset_y += dy * scale

        adjusted = Position(
            x=target.x + offset_x * 0.5,
            y=target.y + offset_y * 0.5,
        )
        return adjusted

    def step(self, fleet: AgentFleet, assignments: dict[str, str], targets: dict[str, Position]) -> None:
        for agent_id, subtask_id in assignments.items():
            if subtask_id in targets:
                safe_target = self.compute_avoidance_offset(
                    fleet, agent_id, targets[subtask_id]
                )
                fleet.step_toward(agent_id, safe_target)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
