"""Static Centralized Hybrid architecture (m=0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.coalition.formation import CoalitionFormation
from src.control.nmpc import NMPCController
from src.decomposition.distance_feasible_decomp import DistanceFeasibleDecomposer
from src.env.agents import distance_matrix
from src.env.daca_env import DACAEnv
from src.llm.cloud_llm_client import CloudLLMClient
from src.llm.device_llm_client import DeviceLLMClient


@dataclass
class CentralizedHybridCoordinator:
    cloud_llm: CloudLLMClient
    device_llm: DeviceLLMClient
    decomposer: DistanceFeasibleDecomposer | None = None
    coalition_formation: CoalitionFormation | None = None
    nmpc: NMPCController = field(default_factory=NMPCController)
    use_distance_decomp: bool = False
    use_coalition_feasibility: bool = False

    def plan(
        self,
        env: DACAEnv,
        cqi_matrix: np.ndarray | None = None,
    ) -> tuple[dict[str, list[str]], list[dict]]:
        obs = env.get_observation()
        fleet = env.fleet
        subtasks = env.subtask_list
        dist_mat = distance_matrix(fleet.agents)
        if cqi_matrix is None:
            cqi_matrix = np.ones(dist_mat.shape)

        if self.use_distance_decomp and self.decomposer:
            assignments_map = self.decomposer.decompose(
                obs["instruction"], fleet, subtasks
            )
        else:
            assignments_map = self.cloud_llm.decompose(
                obs["instruction"],
                obs["agents"],
                obs["subtasks"],
            )

        if self.use_coalition_feasibility and self.coalition_formation:
            coalitions = self.coalition_formation.form(
                fleet, subtasks, dist_mat, cqi_matrix
            )
        else:
            coalitions = self.cloud_llm.form_coalitions(
                obs["subtasks"], obs["agents"]
            )

        self.device_llm.dispatch(coalitions, mode=0)
        return assignments_map, coalitions

    def execute_step(
        self,
        env: DACAEnv,
        assignments: dict[str, str],
    ) -> None:
        targets = {
            s.subtask_id: s.target for s in env.subtask_list
        }
        agent_assignments = {}
        for sid, agents in assignments.items():
            if agents:
                agent_assignments[agents[0]] = sid
        self.nmpc.step(env.fleet, agent_assignments, targets)

        for sid, agent_list in assignments.items():
            if not agent_list:
                continue
            agent = env.fleet.get_agent(agent_list[0])
            subtask = next((s for s in env.subtask_list if s.subtask_id == sid), None)
            if subtask:
                from src.env.agents import dist
                if dist(agent.position, subtask.target) < 5.0:
                    env.mark_subtask_complete(sid)
