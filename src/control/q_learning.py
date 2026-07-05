"""Tabular Q-learning for decentralized collision avoidance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.env.agents import AgentFleet, Position, dist


@dataclass
class QLearningCA:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1
    safe_distance: float = 5.0
    q_table: dict[tuple, np.ndarray] = field(default_factory=dict)
    _standby: bool = True
    n_actions: int = 8

    def _discretize_state(self, fleet: AgentFleet, agent_id: str) -> tuple:
        agent = fleet.get_agent(agent_id)
        nearest = float("inf")
        for other in fleet.agents:
            if other.agent_id != agent_id:
                nearest = min(nearest, dist(agent.position, other.position))
        bucket = int(min(nearest, 20.0) / 2.0)
        return (agent_id, bucket)

    def _action_offset(self, action: int) -> tuple[float, float]:
        angle = 2 * np.pi * action / self.n_actions
        return np.cos(angle) * 2.0, np.sin(angle) * 2.0

    def select_action(self, state: tuple) -> int:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
    ) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)
        q = self.q_table[state][action]
        self.q_table[state][action] = q + self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - q
        )

    def step(
        self,
        fleet: AgentFleet,
        assignments: dict[str, str],
        targets: dict[str, Position],
    ) -> None:
        for agent_id, subtask_id in assignments.items():
            if subtask_id not in targets:
                continue
            state = self._discretize_state(fleet, agent_id)
            action = self.select_action(state)
            ox, oy = self._action_offset(action)
            target = targets[subtask_id]
            adjusted = Position(x=target.x + ox, y=target.y + oy)
            fleet.step_toward(agent_id, adjusted)
            agent = fleet.get_agent(agent_id)
            reward = -1.0
            for other in fleet.agents:
                if other.agent_id != agent_id:
                    d = dist(agent.position, other.position)
                    if d < self.safe_distance:
                        reward -= 5.0
                    else:
                        reward += 0.1
            next_state = self._discretize_state(fleet, agent_id)
            self.update(state, action, reward, next_state)

    def activate(self) -> None:
        self._standby = False

    def standby_mode(self) -> None:
        self._standby = True

    @property
    def active(self) -> bool:
        return not self._standby

    def warmup_step(self, fleet: AgentFleet) -> None:
        """Keep Q-table warm during centralized mode (Eq 30)."""
        for agent in fleet.agents:
            state = self._discretize_state(fleet, agent.agent_id)
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.n_actions)
