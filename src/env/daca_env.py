"""Gymnasium-style multi-agent environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.env.agents import AgentFleet, create_fleet_from_scenario, distance_matrix
from src.env.network_conditions import NetworkConditionGenerator, NetworkProfile
from src.env.scenarios import Scenario, get_scenario


@dataclass
class EnvState:
    timestep: int = 0
    mode: int = 0
    mission_complete: bool = False
    completed_subtasks: list[str] = field(default_factory=list)


class DACAEnv:
    """Multi-agent coordination environment for DACA-HMAS experiments."""

    def __init__(
        self,
        scenario_name: str,
        thresholds: dict[str, Any],
        network_profile: str = "stable",
        seed: int = 0,
        max_steps: int = 500,
    ):
        self.scenario_name = scenario_name
        self.thresholds = thresholds
        self.seed = seed
        self.max_steps = max_steps
        self.scenario = get_scenario(scenario_name, thresholds, seed)
        self.fleet = create_fleet_from_scenario(
            self.scenario.agent_config,
            thresholds.get("kinematics", {}),
            c1=thresholds.get("C1", 50.0),
            c2=thresholds.get("C2", 5.0),
            seed=seed,
        )
        
        self.network = NetworkConditionGenerator.from_scenario(
            scenario_name, network_profile, thresholds, seed, max_steps
        )
        self.network.fleet = self.fleet  # connect network model to fleet (Goal 1 wiring)
        self.state = EnvState()
        self._subtasks = {s.subtask_id: s for s in self.scenario.subtasks}

    def reset(self) -> dict[str, Any]:
        self.state = EnvState()
        self.scenario = get_scenario(self.scenario_name, self.thresholds, self.seed)
        self.fleet = create_fleet_from_scenario(
            self.scenario.agent_config,
            self.thresholds.get("kinematics", {}),
            c1=self.thresholds.get("C1", 50.0),
            c2=self.thresholds.get("C2", 5.0),
            seed=self.seed,
        )
        self.network.fleet = self.fleet  # re-connect after fleet rebuild
        self._subtasks = {s.subtask_id: s for s in self.scenario.subtasks}
        return self.get_observation()

    def get_observation(self) -> dict[str, Any]:
        return {
            "timestep": self.state.timestep,
            "mode": self.state.mode,
            "agents": self.fleet.to_dict_list(),
            "subtasks": [
                {
                    "id": s.subtask_id,
                    "target": [s.target.x, s.target.y],
                    "skills": s.required_skills,
                    "completed": s.completed,
                    "assigned": s.assigned_agents,
                }
                for s in self._subtasks.values()
            ],
            "distance_matrix": distance_matrix(self.fleet.agents).tolist(),
            "instruction": self.scenario.instruction,
        }

    def step_agents_toward_targets(self, assignments: dict[str, str]) -> None:
        """Move agents toward assigned subtask targets."""
        for agent_id, subtask_id in assignments.items():
            if subtask_id in self._subtasks:
                target = self._subtasks[subtask_id].target
                self.fleet.step_toward(agent_id, target)

    def mark_subtask_complete(self, subtask_id: str) -> None:
        if subtask_id in self._subtasks:
            self._subtasks[subtask_id].completed = True
            print(f"[COMPLETE] {subtask_id}")
            if subtask_id not in self.state.completed_subtasks:
                self.state.completed_subtasks.append(subtask_id)

    def check_mission_complete(self) -> bool:
        return all(s.completed for s in self._subtasks.values())

    def advance(self) -> dict[str, Any]:
        self.state.timestep += 1
        if self.check_mission_complete() or self.state.timestep >= self.max_steps:
            self.state.mission_complete = True
        return self.get_observation()

    @property
    def num_subtasks(self) -> int:
        return len(self._subtasks)

    @property
    def subtask_list(self) -> list:
        return list(self._subtasks.values())

    def success_rate(self) -> float:
        total = len(self._subtasks)
        if total == 0:
            return 0.0
        return len(self.state.completed_subtasks) / total
