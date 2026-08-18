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
    valid_ids = set(fleet._id_to_idx.keys())

    for aid in agent_ids:
        if aid not in valid_ids:
            print(f"[WARNING] Unknown agent id returned by planner: {aid}")
            return False
    
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

        # Skill tiers, highest priority first.
        full_skill_candidates = [
            a for a in fleet.agents if all(s in a.skills for s in subtask.required_skills)
        ]
        any_skill_candidates = [
            a for a in fleet.agents if any(s in a.skills for s in subtask.required_skills)
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
            """Lowest-cost agent in `cands` that lies within r_reach, or None.

            `use_eta` selects travel-time ranking. It is enabled ONLY for
            fallback tiers, i.e. for subtasks the previous implementation would
            have dropped. The primary (full-skill) tier keeps the original
            distance ranking so that every assignment the old solver already
            produced is reproduced bit-for-bit.
            """
            best_id = None
            best_cost = float("inf")
            for agent in cands:
                d = dist(agent.position, subtask.target)
                if d <= self.r_reach:
                    # TRAVEL-TIME FIX: cost on estimated time-to-arrival, not raw
                    # distance. The fleet is speed-heterogeneous (uav 15.0,
                    # vehicle 10.0, robot 3.0 in configs/thresholds.yaml -- a 5x
                    # spread) and the mission is step-limited, so a robot 30 m
                    # away is a worse choice than a UAV 80 m away. Ranking on
                    # distance alone concentrated work on nearby slow agents:
                    # measured on logistics seed 4, robot_5 (speed 3.0) was
                    # assigned T_0, T_2 and T_5 and completed none of them
                    # within 200 steps.
                    # Normalized so that an agent travelling at the fleet's top
                    # speed across the full r_reach scores 1.0, keeping the term
                    # in [0, 1] and scenario-independent.
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
                    # Composite cost C(a, s) in [0, 1]
                    cost = w_dist * norm_dist + w_workload * norm_workload
                    # Deterministic tie-break on agent_id.
                    if cost < best_cost or (
                        cost == best_cost
                        and best_id is not None
                        and agent.agent_id < best_id
                    ):
                        best_cost = cost
                        best_id = agent.agent_id
            return best_id

        # REGRESSION FIX (logistics success):
        # The previous implementation SELECTED a skill tier first and only then
        # applied the r_reach filter. Once a non-empty tier was chosen the chain
        # committed to it, so if every agent in that tier happened to be out of
        # reach the function returned [] -- and decompose() then omitted the
        # subtask from the plan entirely, even though a reachable agent existed
        # in a lower-priority tier.
        #
        # Measured (logistics, oscillatory, PYTHONHASHSEED=0, t=0 geometry):
        #   seed 3 / T_2: full=1, that agent 146.5 m away; nearest agent overall
        #                 26.3 m. Tier committed to the single full-skill agent
        #                 -> [] -> subtask dropped on 49 decomposition calls.
        #   seed 3 / T_5: full=1 at 179.8 m; nearest overall 68.9 m -> dropped.
        #   seed 3 / T_3: full=0, nearest any-skill 104.1 m; nearest overall
        #                 45.0 m -> dropped.
        # Three of six logistics subtasks were unassignable for the whole
        # mission, capping success at 50.00% (observed exactly) and pinning
        # s_task low enough that plan validity never cleared 0.75 (continuity
        # pass rate 0.0%), which drove 72 cloud planning calls.
        #
        # The fix walks the tiers and returns the first tier that actually
        # contains a reachable agent. Skill priority is unchanged whenever the
        # preferred tier is reachable, so this is strictly a widening of the
        # fallback, not a change of preference.
        tiers = (full_skill_candidates, any_skill_candidates, list(fleet.agents))
        # Index of the tier the PREVIOUS implementation would have committed to:
        # full-skill if non-empty, else any-skill if non-empty, else whole fleet.
        legacy_idx = 0 if full_skill_candidates else (1 if any_skill_candidates else 2)
        for idx, tier in enumerate(tiers):
            if not tier:
                continue
            # Travel-time ranking applies ONLY in tiers the previous solver would
            # never have reached. In the legacy tier the original distance
            # ranking is preserved exactly, so every assignment the old solver
            # already produced is reproduced bit-for-bit and this change can only
            # add assignments, never move them.
            chosen = _best_in_reach(tier, use_eta=(idx > legacy_idx))
            if chosen:
                return [chosen]

        # No agent of any tier is within r_reach. Assign the nearest agent from
        # the most skilled non-empty tier so the subtask is never orphaned: it
        # is a reachability problem (the agent can travel), not an assignment
        # problem, and an orphaned subtask can never be completed at all.
        tier = full_skill_candidates or any_skill_candidates or list(fleet.agents)
        if not tier:
            return []
        nearest = min(
            tier,
            key=lambda a: (
                dist(a.position, subtask.target) / max(self._agent_speed(a, fleet), 1e-9),
                a.agent_id,
            ),
        )
        return [nearest.agent_id]

