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


def validate_assignment_skills(
    agent_ids: list[str],
    subtask: Subtask,
    fleet: AgentFleet,
) -> bool:
    """Validate that agent_ids is non-empty, all agents exist in fleet,
    and the assigned team collectively covers all subtask.required_skills as a hard constraint."""
    if not agent_ids:
        return False
    valid_ids = set(fleet._id_to_idx.keys())
    for aid in agent_ids:
        if aid not in valid_ids:
            return False
    required = set(subtask.required_skills)
    team_skills: set[str] = set()
    for aid in agent_ids:
        team_skills.update(fleet.get_agent(aid).skills)
    return required.issubset(team_skills)


def validate_joint_assignment(
    agent_ids: list[str],
    subtask: Subtask,
    fleet: AgentFleet,
    c_task: float,
    r_reach: float,
) -> bool:
    """Check all pairs in joint assignment satisfy delta = 1 and cover required skills."""
    if not validate_assignment_skills(agent_ids, subtask, fleet):
        return False
    
    agents = [fleet.get_agent(aid) for aid in agent_ids]
    if len(agents) <= 1:
        return dist(agents[0].position, subtask.target) <= r_reach

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if delta_feasibility(agents[i], agents[j], subtask, c_task, r_reach) < 1.0:
                return False
    return True


def validate_task_completion(
    agent_ids: list[str],
    subtask: Subtask,
    fleet: AgentFleet,
    completion_radius: float = 8.0,
) -> bool:
    """Validate that the assigned team is valid, covers all required skills,
    and all assigned agents have arrived within completion_radius of the target."""
    if subtask.completed:
        return False
    if not validate_assignment_skills(agent_ids, subtask, fleet):
        return False
    agents = [fleet.get_agent(aid) for aid in agent_ids]
    for a in agents:
        if dist(a.position, subtask.target) >= completion_radius:
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
    cloud_llm: Any
    c_task: float = 30.0
    r_reach: float = 100.0

    @property
    def llm_client(self) -> Any:
        return self.cloud_llm

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
        # Filter invalid agent IDs returned by the LLM
        valid_ids = {a.agent_id for a in fleet.agents}

        for task_id, ids in raw_assignments.items():
            if isinstance(ids, str):
                ids = [ids]

            filtered = [aid for aid in ids if aid in valid_ids]

            if len(filtered) != len(ids):
                removed = set(ids) - set(filtered)
                print(f"[WARNING] Invalid IDs for {task_id}: {removed}")

            raw_assignments[task_id] = filtered

        validated: dict[str, list[str]] = {}

        for st in subtasks:
            sid = st.subtask_id
            candidates = raw_assignments.get(sid, [])

            if isinstance(candidates, str):
                candidates = [candidates]

            if validate_joint_assignment(
                candidates,
                st,
                fleet,
                self.c_task,
                self.r_reach,
            ):
                validated[sid] = candidates
            else:
                best = self._find_feasible_agents(st, fleet, validated)
                if best:
                    validated[sid] = best

        return validated

    @staticmethod
    def _agent_speed(agent, fleet) -> float:
        """Type-specific max speed from the fleet kinematics config.

        Falls back to 1.0 so the cost function degrades to distance-ranking if
        kinematics are unavailable, rather than raising.
        """
        try:
            return float(fleet.kinematics[agent.agent_type.value].max_speed)
        except Exception:
            return 1.0

    def _find_feasible_agents(
        self,
        subtask: Subtask,
        fleet: AgentFleet,
        current_assignments: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """Skill-aware and workload-balanced fallback assignment."""
        workload: dict[str, int] = {}
        if current_assignments:
            for aids in current_assignments.values():
                for aid in aids:
                    workload[aid] = workload.get(aid, 0) + 1

        # Hard skill constraint: only candidates that collectively cover all required skills
        required = set(subtask.required_skills)
        full_skill_candidates = [
            a for a in fleet.agents if required.issubset(set(a.skills))
        ]

        n_tasks = max(len(current_assignments) if current_assignments else 1, 1)

        # Convex weights: dist=0.50, workload=0.50 (sum = 1.0)
        w_dist = 0.50
        w_workload = 0.50

        # Fleet reference speed for travel-time normalization (see _agent_speed).
        v_ref = max(
            (self._agent_speed(a, fleet) for a in fleet.agents), default=1.0
        ) or 1.0

        def _best_in_reach(cands: list, use_eta: bool = False) -> str | None:
            best_id = None
            best_cost = float("inf")
            for agent in cands:
                d = dist(agent.position, subtask.target)
                if d <= self.r_reach:
                    if use_eta:
                        v_a = self._agent_speed(agent, fleet)
                        eta = d / v_a if v_a > 0 else float("inf")
                        eta_ref = self.r_reach / v_ref if v_ref > 0 else 1.0
                        norm_dist = min(eta / eta_ref, 1.0) if eta_ref > 0 else 1.0
                    else:
                        norm_dist = min(d / self.r_reach, 1.0)
                    norm_workload = min(
                        workload.get(agent.agent_id, 0) / n_tasks, 1.0
                    )
                    cost = w_dist * norm_dist + w_workload * norm_workload
                    if cost < best_cost or (
                        cost == best_cost
                        and best_id is not None
                        and agent.agent_id < best_id
                    ):
                        best_cost = cost
                        best_id = agent.agent_id
            return best_id

        # 1. Single agent within reach covering all required skills
        if full_skill_candidates:
            chosen = _best_in_reach(full_skill_candidates, use_eta=False)
            if chosen:
                return [chosen]

        # 2. Multi-agent pair within reach collectively covering all required skills
        agents_in_reach = [a for a in fleet.agents if dist(a.position, subtask.target) <= self.r_reach]
        best_pair = None
        best_pair_cost = float("inf")
        for i in range(len(agents_in_reach)):
            for j in range(i + 1, len(agents_in_reach)):
                a1, a2 = agents_in_reach[i], agents_in_reach[j]
                if required.issubset(set(a1.skills) | set(a2.skills)):
                    if dist(a1.position, a2.position) <= self.c_task:
                        pair_cost = dist(a1.position, subtask.target) + dist(a2.position, subtask.target)
                        if pair_cost < best_pair_cost:
                            best_pair_cost = pair_cost
                            best_pair = [a1.agent_id, a2.agent_id]
        if best_pair:
            return best_pair

        # 3. Single agent covering all skills beyond r_reach (can travel to target)
        if full_skill_candidates:
            nearest = min(
                full_skill_candidates,
                key=lambda a: (
                    dist(a.position, subtask.target) / max(self._agent_speed(a, fleet), 1e-9),
                    a.agent_id,
                ),
            )
            return [nearest.agent_id]

        # 4. Multi-agent pair beyond r_reach collectively covering all skills
        best_pair_all = None
        best_pair_all_cost = float("inf")
        for i in range(len(fleet.agents)):
            for j in range(i + 1, len(fleet.agents)):
                a1, a2 = fleet.agents[i], fleet.agents[j]
                if required.issubset(set(a1.skills) | set(a2.skills)):
                    cost = dist(a1.position, subtask.target) + dist(a2.position, subtask.target)
                    if cost < best_pair_all_cost:
                        best_pair_all_cost = cost
                        best_pair_all = [a1.agent_id, a2.agent_id]
        if best_pair_all:
            return best_pair_all

        # No agent or team can satisfy required skills
        return []


