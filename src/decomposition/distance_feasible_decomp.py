"""Distance-feasible task decomposition (Gap 1, extended Eq 12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.env.agents import AgentFleet, AgentState, Position, dist
from src.env.scenarios import Subtask
from src.llm.cloud_llm_client import CloudLLMClient


def delta_feasibility(
    agent_i: AgentState,
    agent_j: AgentState,
    subtask: Subtask,
    c_task: float,
    r_reach: float,
) -> float:
    """Eq: delta_ii'j(t) — task-level distance feasibility indicator."""
    inter_agent = 1.0 if dist(agent_i.position, agent_j.position) <= c_task else 0.0
    reach_i = 1.0 if dist(agent_i.position, subtask.target) <= r_reach else 0.0
    reach_j = 1.0 if dist(agent_j.position, subtask.target) <= r_reach else 0.0
    return inter_agent * reach_i * reach_j


def subtask_feasibility_matrix(
    agents: list[AgentState],
    subtask: Subtask,
    c_task: float,
    r_reach: float,
) -> np.ndarray:
    """D_j(t): subtask distance feasibility matrix."""
    n = len(agents)
    d = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = delta_feasibility(agents[i], agents[j], subtask, c_task, r_reach)
            elif dist(agents[i].position, subtask.target) > r_reach:
                d[i, j] = 0.0
    return d


def validate_joint_assignment(
    agent_ids: list[str],
    subtask: Subtask,
    fleet: AgentFleet,
    c_task: float,
    r_reach: float,
) -> bool:
    """Check all pairs in joint assignment satisfy delta = 1."""
    agents = [fleet.get_agent(aid) for aid in agent_ids]
    if len(agents) <= 1:
        if agents:
            return dist(agents[0].position, subtask.target) <= r_reach
        return False
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if delta_feasibility(agents[i], agents[j], subtask, c_task, r_reach) < 1.0:
                return False
    return True


def compute_tfr(
    assignments: dict[str, list[str]],
    subtasks: list[Subtask],
    fleet: AgentFleet,
    c_task: float,
    r_reach: float,
) -> float:
    """Task Feasibility Rate (TFR)."""
    if not subtasks:
        return 1.0
    feasible = 0
    for st in subtasks:
        agent_ids = assignments.get(st.subtask_id, [])
        if not agent_ids:
            continue
        if validate_joint_assignment(agent_ids, st, fleet, c_task, r_reach):
            feasible += 1
    assigned = sum(1 for st in subtasks if assignments.get(st.subtask_id))
    if assigned == 0:
        return 0.0
    return feasible / assigned


@dataclass
class DistanceFeasibleDecomposer:
    cloud_llm: CloudLLMClient
    c_task: float = 30.0
    r_reach: float = 100.0

    def decompose(
        self,
        instruction: str,
        fleet: AgentFleet,
        subtasks: list[Subtask],
    ) -> dict[str, list[str]]:
        """Extended Eq 12: T = LLM(I, E, Delta, D(t))."""
        d_matrix = fleet.agents
        from src.env.agents import distance_matrix

        dist_mat = distance_matrix(d_matrix).tolist()
        agents_ctx = fleet.to_dict_list()
        subtasks_ctx = [
            {
                "id": s.subtask_id,
                "target": [s.target.x, s.target.y],
                "skills": s.required_skills,
            }
            for s in subtasks
        ]
        raw_assignments = self.cloud_llm.decompose(
            instruction, agents_ctx, subtasks_ctx, dist_mat
        )

        validated: dict[str, list[str]] = {}
        for st in subtasks:
            sid = st.subtask_id
            candidates = raw_assignments.get(sid, [])
            if isinstance(candidates, str):
                candidates = [candidates]
            if validate_joint_assignment(candidates, st, fleet, self.c_task, self.r_reach):
                validated[sid] = candidates
            else:
                best = self._find_feasible_agents(st, fleet)
                if best:
                    validated[sid] = best
        return validated

    def _find_feasible_agents(
        self, subtask: Subtask, fleet: AgentFleet
    ) -> list[str]:
        """Greedy fallback: pick nearest reachable agent."""
        best_id = None
        best_dist = float("inf")
        for agent in fleet.agents:
            d = dist(agent.position, subtask.target)
            if d <= self.r_reach and d < best_dist:
                best_dist = d
                best_id = agent.agent_id
        return [best_id] if best_id else []
