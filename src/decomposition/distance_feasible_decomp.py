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
    and all assigned agents are within completion_radius of the target."""
    if not agent_ids or subtask.completed:
        return False
    if not validate_assignment_skills(agent_ids, subtask, fleet):
        return False

    for aid in agent_ids:
        agent = fleet.get_agent(aid)
        if agent is None or dist(agent.position, subtask.target) >= completion_radius:
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
        distance_matrix: list[list[float]] | None = None,
    ) -> dict[str, list[str]]:
        """Decompose mission into task assignments that satisfy delta_feasibility (Gap 1)."""
        agents_ctx = fleet.to_dict_list()
        subtasks_ctx = [
            {"id": s.subtask_id, "skills": s.required_skills, "target": [s.target.x, s.target.y]}
            for s in subtasks
        ]

        raw_assignments = self.cloud_llm.decompose(
            instruction,
            agents_ctx,
            subtasks_ctx,
            distance_matrix=distance_matrix,
        )

        return self.validate_assignments(raw_assignments, fleet, subtasks)

    def validate_assignments(
        self,
        raw_assignments: dict[str, list[str]],
        fleet: AgentFleet,
        subtasks: list[Subtask],
    ) -> dict[str, list[str]]:
        """Filter out unknown agents and validate assignments using feasibility metric."""
        valid_ids = {a.agent_id for a in fleet.agents}

        for task_id, ids in raw_assignments.items():
            if isinstance(ids, str):
                ids = [ids]

            # Reject unknown IDs immediately: do not silently discard unknown IDs and validate remaining members
            if any(aid not in valid_ids for aid in ids):
                removed = set(ids) - valid_ids
                print(f"[WARNING] Invalid IDs for {task_id}: {removed} -- rejecting assignment")
                raw_assignments[task_id] = []
            else:
                raw_assignments[task_id] = list(ids)


        validated: dict[str, list[str]] = {}
        assigned_agent_ids: set[str] = set()
        for st in subtasks:
            sid = st.subtask_id
            candidates = raw_assignments.get(sid, [])

            if isinstance(candidates, str):
                candidates = [candidates]

            if (
                validate_joint_assignment(
                    candidates,
                    st,
                    fleet,
                    self.c_task,
                    self.r_reach,
                )
                and not any(aid in assigned_agent_ids for aid in candidates)
            ):
                validated[sid] = candidates
                assigned_agent_ids.update(candidates)
            else:
                best = self._find_feasible_agents(st, fleet, validated)
                if best:
                    validated[sid] = best
                    assigned_agent_ids.update(best)

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
        """Skill-aware and workload-balanced fallback assignment with complementary coalition support."""
        workload: dict[str, int] = {}
        if current_assignments:
            for aids in current_assignments.values():
                for aid in aids:
                    workload[aid] = workload.get(aid, 0) + 1

        required = set(subtask.required_skills)
        full_skill_candidates = [
            a for a in fleet.agents if required.issubset(set(a.skills))
        ]

        w_dist = 0.50

        v_ref = max(
            (self._agent_speed(a, fleet) for a in fleet.agents), default=1.0
        ) or 1.0
        eta_ref = self.r_reach / v_ref if v_ref > 0 else 1.0

        def _single_cost(agent: AgentState, use_eta: bool = True) -> float:
            d = dist(agent.position, subtask.target)
            if use_eta:
                v_a = self._agent_speed(agent, fleet)
                eta = d / v_a if v_a > 0 else float("inf")
                norm_dist = min(eta / eta_ref, 1.0) if eta_ref > 0 else 1.0
            else:
                norm_dist = min(d / self.r_reach, 1.0)
            wl = workload.get(agent.agent_id, 0)
            return w_dist * norm_dist + (10.0 * wl)

        # 1. Free single agent within r_reach covering all required skills
        free_singles_in_reach = [
            a for a in full_skill_candidates
            if dist(a.position, subtask.target) <= self.r_reach and workload.get(a.agent_id, 0) == 0
        ]
        if free_singles_in_reach:
            best_single = min(
                free_singles_in_reach,
                key=lambda a: (
                    dist(a.position, subtask.target) / max(self._agent_speed(a, fleet), 1e-9),
                    a.agent_id,
                ),
            )
            return [best_single.agent_id]

        # 2. Free multi-agent pair within r_reach collectively covering all required skills
        free_agents_in_reach = [
            a for a in fleet.agents
            if dist(a.position, subtask.target) <= self.r_reach and workload.get(a.agent_id, 0) == 0
        ]
        best_free_pair = None
        best_free_pair_cost = float("inf")
        for i in range(len(free_agents_in_reach)):
            for j in range(i + 1, len(free_agents_in_reach)):
                a1, a2 = free_agents_in_reach[i], free_agents_in_reach[j]
                if required.issubset(set(a1.skills) | set(a2.skills)):
                    if dist(a1.position, a2.position) <= self.c_task:
                        v1 = self._agent_speed(a1, fleet)
                        v2 = self._agent_speed(a2, fleet)
                        eta1 = dist(a1.position, subtask.target) / v1 if v1 > 0 else float("inf")
                        eta2 = dist(a2.position, subtask.target) / v2 if v2 > 0 else float("inf")
                        cost = max(eta1, eta2)
                        pair_ids = sorted([a1.agent_id, a2.agent_id])
                        if cost < best_free_pair_cost or (cost == best_free_pair_cost and best_free_pair is not None and pair_ids < best_free_pair):
                            best_free_pair_cost = cost
                            best_free_pair = pair_ids
        if best_free_pair:
            return best_free_pair

        # 3. Free single agent beyond r_reach (can travel to target)
        free_singles_all = [
            a for a in full_skill_candidates if workload.get(a.agent_id, 0) == 0
        ]
        if free_singles_all:
            nearest_free = min(
                free_singles_all,
                key=lambda a: (
                    dist(a.position, subtask.target) / max(self._agent_speed(a, fleet), 1e-9),
                    a.agent_id,
                ),
            )
            return [nearest_free.agent_id]

        # 4. Free multi-agent pair beyond r_reach collectively covering all skills
        free_agents_all = [
            a for a in fleet.agents if workload.get(a.agent_id, 0) == 0
        ]
        best_free_pair_all = None
        best_free_pair_all_cost = float("inf")
        for i in range(len(free_agents_all)):
            for j in range(i + 1, len(free_agents_all)):
                a1, a2 = free_agents_all[i], free_agents_all[j]
                if required.issubset(set(a1.skills) | set(a2.skills)):
                    v1 = self._agent_speed(a1, fleet)
                    v2 = self._agent_speed(a2, fleet)
                    eta1 = dist(a1.position, subtask.target) / v1 if v1 > 0 else float("inf")
                    eta2 = dist(a2.position, subtask.target) / v2 if v2 > 0 else float("inf")
                    cost = max(eta1, eta2)
                    pair_ids = sorted([a1.agent_id, a2.agent_id])
                    if cost < best_free_pair_all_cost or (cost == best_free_pair_all_cost and best_free_pair_all is not None and pair_ids < best_free_pair_all):
                        best_free_pair_all_cost = cost
                        best_free_pair_all = pair_ids
        if best_free_pair_all:
            return best_free_pair_all

        # 5. If no completely free agents exist, pick the least-loaded single agent or pair
        best_fallback_single = None
        best_fallback_single_cost = float("inf")
        for a in full_skill_candidates:
            cost = _single_cost(a, use_eta=True)
            if cost < best_fallback_single_cost or (cost == best_fallback_single_cost and best_fallback_single is not None and a.agent_id < best_fallback_single):
                best_fallback_single_cost = cost
                best_fallback_single = a.agent_id
        if best_fallback_single:
            return [best_fallback_single]

        best_fallback_pair = None
        best_fallback_pair_cost = float("inf")
        for i in range(len(fleet.agents)):
            for j in range(i + 1, len(fleet.agents)):
                a1, a2 = fleet.agents[i], fleet.agents[j]
                if required.issubset(set(a1.skills) | set(a2.skills)):
                    v1 = self._agent_speed(a1, fleet)
                    v2 = self._agent_speed(a2, fleet)
                    eta1 = dist(a1.position, subtask.target) / v1 if v1 > 0 else float("inf")
                    eta2 = dist(a2.position, subtask.target) / v2 if v2 > 0 else float("inf")
                    team_eta = max(eta1, eta2)
                    wl = workload.get(a1.agent_id, 0) + workload.get(a2.agent_id, 0)
                    cost = team_eta + (10000.0 * wl)
                    pair_ids = sorted([a1.agent_id, a2.agent_id])
                    if cost < best_fallback_pair_cost or (cost == best_fallback_pair_cost and best_fallback_pair is not None and pair_ids < best_fallback_pair):
                        best_fallback_pair_cost = cost
                        best_fallback_pair = pair_ids
        if best_fallback_pair:
            return best_fallback_pair


        # Strict: No agent or team can satisfy required skills
        return []


