"""Plan Continuity Engine for DACA-HMAS.

Evaluates Plan Validity Score (V_plan) upon architecture switching (Centralized <-> Decentralized)
or state changes to maintain plan continuity, avoiding unnecessary LLM calls when the active plan
remains executable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.env.agents import AgentFleet, dist
from src.env.scenarios import Subtask


@dataclass
class PlanValidityScore:
    """Quantitative evaluation breakdown for active global plan continuity."""
    task_completion_score: float = 1.0
    distance_feasibility_score: float = 1.0
    communication_quality_score: float = 1.0
    coalition_feasibility_score: float = 1.0
    resource_network_score: float = 1.0
    total_validity_score: float = 1.0
    validity_threshold: float = 0.75

    @property
    def is_valid(self) -> bool:
        return self.total_validity_score >= self.validity_threshold


@dataclass
class ActivePlanContext:
    """State context captured along with the active plan."""
    assignments: dict[str, list[str]] = field(default_factory=dict)
    coalitions: list[dict[str, Any]] = field(default_factory=list)
    completed_subtask_ids: set[str] = field(default_factory=set)
    subtask_targets: dict[str, tuple[float, float]] = field(default_factory=dict)
    subtask_required_skills: dict[str, set[str]] = field(default_factory=dict)
    mode: int = 0
    sys_cqi: float = 1.0
    packet_loss: float = 0.0
    latency: float = 0.0
    step: int = 0


class PlanContinuityEngine:
    """Engine responsible for computing plan validity and preserving continuity across architecture switches."""

    def __init__(
        self,
        validity_threshold: float = 0.75,
        r_reach: float = 100.0,
        c_task: float = 30.0,
        cqi_min_threshold: float = 0.4,
    ):
        self.validity_threshold = validity_threshold
        self.r_reach = r_reach
        self.c_task = c_task
        self.cqi_min_threshold = cqi_min_threshold
        self.active_context: ActivePlanContext | None = None

    def set_active_plan(
        self,
        assignments: dict[str, list[str]],
        coalitions: list[dict[str, Any]],
        subtasks: Sequence[Subtask],
        mode: int,
        sys_cqi: float = 1.0,
        packet_loss: float = 0.0,
        latency: float = 0.0,
        step: int = 0,
    ) -> None:
        """Store active global plan baseline for continuity tracking."""
        targets = {s.subtask_id: (s.target.x, s.target.y) for s in subtasks}
        skills = {s.subtask_id: set(s.required_skills) for s in subtasks}
        completed = {s.subtask_id for s in subtasks if s.completed}
        self.active_context = ActivePlanContext(
            assignments=dict(assignments),
            coalitions=list(coalitions),
            completed_subtask_ids=completed,
            subtask_targets=targets,
            subtask_required_skills=skills,
            mode=mode,
            sys_cqi=sys_cqi,
            packet_loss=packet_loss,
            latency=latency,
            step=step,
        )

    def mark_subtask_completed(self, subtask_id: str) -> None:
        """Record completed subtask in active context and remove from executable assignments."""
        if self.active_context is not None:
            self.active_context.completed_subtask_ids.add(subtask_id)
            self.active_context.assignments.pop(subtask_id, None)

    def evaluate_plan_validity(
        self,
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        cqi_matrix: np.ndarray | None = None,
        sys_cqi: float = 1.0,
        packet_loss: float = 0.0,
        latency: float = 0.0,
    ) -> PlanValidityScore:
        """Compute quantitative Plan Validity Score (V_plan) with pre-scoring hard feasibility gates."""
        if self.active_context is None or not self.active_context.assignments:
            return PlanValidityScore(
                total_validity_score=0.0, validity_threshold=self.validity_threshold
            )

        ctx = self.active_context
        completed_ids = ctx.completed_subtask_ids | {s.subtask_id for s in subtasks if s.completed}
        incomplete_subtasks = [
            s for s in subtasks if s.subtask_id not in completed_ids and not s.completed
        ]
        if not incomplete_subtasks:
            return PlanValidityScore(
                total_validity_score=1.0, validity_threshold=self.validity_threshold
            )

        # ── Hard Feasibility Gates before weighted scoring ────────────────────
        from src.decomposition.distance_feasible_decomp import validate_joint_assignment

        agent_map = {a.agent_id: a for a in fleet.agents}
        assigned_agents_seen: set[str] = set()

        for s in incomplete_subtasks:
            sid = s.subtask_id
            assigned = ctx.assignments.get(sid, [])
            # 1. Non-empty assignment
            if not assigned:
                return PlanValidityScore(
                    task_completion_score=0.0,
                    distance_feasibility_score=0.0,
                    communication_quality_score=0.0,
                    coalition_feasibility_score=0.0,
                    resource_network_score=0.0,
                    total_validity_score=0.0,
                    validity_threshold=self.validity_threshold,
                )
            # 2. Known agent IDs
            if any(aid not in agent_map for aid in assigned):
                return PlanValidityScore(
                    task_completion_score=0.0,
                    distance_feasibility_score=0.0,
                    communication_quality_score=0.0,
                    coalition_feasibility_score=0.0,
                    resource_network_score=0.0,
                    total_validity_score=0.0,
                    validity_threshold=self.validity_threshold,
                )
            # 3. No agent assigned to multiple incomplete tasks
            for aid in assigned:
                if aid in assigned_agents_seen:
                    return PlanValidityScore(
                        task_completion_score=0.0,
                        distance_feasibility_score=0.0,
                        communication_quality_score=0.0,
                        coalition_feasibility_score=0.0,
                        resource_network_score=0.0,
                        total_validity_score=0.0,
                        validity_threshold=self.validity_threshold,
                    )
                assigned_agents_seen.add(aid)
            # 4. Required skills satisfied & joint distance feasibility passes
            if not validate_joint_assignment(assigned, s, fleet, self.c_task, self.r_reach):
                return PlanValidityScore(
                    task_completion_score=0.0,
                    distance_feasibility_score=0.0,
                    communication_quality_score=0.0,
                    coalition_feasibility_score=0.0,
                    resource_network_score=0.0,
                    total_validity_score=0.0,
                    validity_threshold=self.validity_threshold,
                )

        # 1. Task Completion Alignment
        valid_assignments = 0
        total_active = 0
        for s in incomplete_subtasks:
            sid = s.subtask_id
            total_active += 1
            if sid in ctx.assignments and len(ctx.assignments[sid]) > 0:
                valid_assignments += 1
        s_task = (valid_assignments / total_active) if total_active > 0 else 1.0

        # 2. Distance Feasibility
        feasible_distances = 0
        dist_count = 0
        agent_map = {a.agent_id: a for a in fleet.agents}
        for s in incomplete_subtasks:
            sid = s.subtask_id
            assigned_agents = ctx.assignments.get(sid, [])
            for aid in assigned_agents:
                agent = agent_map.get(aid)
                if agent is not None:
                    dist_count += 1
                    d = dist(agent.position, s.target)
                    if d <= self.r_reach:
                        feasible_distances += 1
        s_dist = (feasible_distances / dist_count) if dist_count > 0 else 1.0

        # 3. Communication Quality
        if cqi_matrix is not None and cqi_matrix.size > 0:
            avg_cqi = float(np.mean(cqi_matrix))
            s_comm = min(1.0, max(0.0, avg_cqi / self.cqi_min_threshold))
        else:
            s_comm = min(1.0, max(0.0, sys_cqi / self.cqi_min_threshold))

        # 4. Coalition Skill Satisfaction
        satisfied_coalitions = 0
        coalition_count = 0
        agent_skills = {a.agent_id: set(a.skills) for a in fleet.agents}
        for c in ctx.coalitions:
            members = c.get("members", [])
            if not members:
                continue
            coalition_count += 1
            c_skills: set[str] = set()
            for m in members:
                c_skills.update(agent_skills.get(m, set()))
            covered = True
            for s in incomplete_subtasks:
                if any(m in ctx.assignments.get(s.subtask_id, []) for m in members):
                    if not set(s.required_skills).issubset(c_skills):
                        covered = False
                        break
            if covered:
                satisfied_coalitions += 1
        s_coalition = (satisfied_coalitions / coalition_count) if coalition_count > 0 else 1.0

        # 5. Resource & Network Condition Score
        s_res = 1.0
        if packet_loss > 0.4 or latency > 0.8:
            s_res = 0.5

        v_plan = (
            0.30 * s_task
            + 0.25 * s_dist
            + 0.20 * s_comm
            + 0.15 * s_coalition
            + 0.10 * s_res
        )

        return PlanValidityScore(
            task_completion_score=s_task,
            distance_feasibility_score=s_dist,
            communication_quality_score=s_comm,
            coalition_feasibility_score=s_coalition,
            resource_network_score=s_res,
            total_validity_score=v_plan,
            validity_threshold=self.validity_threshold,
        )

    def apply_target_commitment_lock(
        self,
        new_assignments: dict[str, list[str]],
        previous_assignments: dict[str, list[str]],
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        lock_threshold: float = 35.0,
    ) -> dict[str, list[str]]:
        """Lock agent assignment to an incomplete subtask if agent is within lock_threshold distance."""
        if not previous_assignments:
            return new_assignments

        agent_map = {a.agent_id: a for a in fleet.agents}
        locked_assignments = {sid: list(agents) for sid, agents in new_assignments.items()}
        locked_agent_ids: set[str] = set()

        from src.decomposition.distance_feasible_decomp import validate_assignment_skills

        for st in subtasks:
            if st.completed:
                continue
            sid = st.subtask_id
            prev_agents = previous_assignments.get(sid, [])
            if prev_agents and validate_assignment_skills(prev_agents, st, fleet):
                # If any assigned agent is committed within lock threshold, lock the assigned team
                if any(aid in agent_map and dist(agent_map[aid].position, st.target) < lock_threshold for aid in prev_agents):
                    locked_assignments[sid] = [aid for aid in prev_agents if aid in agent_map]
                    locked_agent_ids.update(locked_assignments[sid])

        # Clear duplicate assignments for locked agents in other subtasks
        for sid, agents in list(locked_assignments.items()):
            st = next((s for s in subtasks if s.subtask_id == sid), None)
            if st and st.completed:
                continue
            curr_agents = [aid for aid in agents if aid not in locked_agent_ids or aid in previous_assignments.get(sid, [])]
            if st and curr_agents and validate_assignment_skills(curr_agents, st, fleet):
                locked_assignments[sid] = curr_agents
            else:
                locked_assignments[sid] = []

        return locked_assignments

    @staticmethod
    def clean_duplicate_assignments(
        assignments: dict[str, list[str]],
        subtasks: Sequence[Subtask],
        fleet: AgentFleet,
    ) -> dict[str, list[str]]:
        """Remove duplicate agent assignments across tasks, ensuring each agent is assigned to at most one task."""
        from src.decomposition.distance_feasible_decomp import validate_assignment_skills

        agent_map = {a.agent_id: a for a in fleet.agents}
        task_map = {s.subtask_id: s for s in subtasks}

        agent_to_task: dict[str, str] = {}
        cleaned: dict[str, list[str]] = {sid: list(aids) for sid, aids in assignments.items()}

        for sid, aids in list(cleaned.items()):
            st = task_map.get(sid)
            kept_agents: list[str] = []
            for aid in aids:
                if aid not in agent_map:
                    continue
                if aid in agent_to_task:
                    other_sid = agent_to_task[aid]
                    other_st = task_map.get(other_sid)
                    pos = agent_map[aid].position
                    d_current = dist(pos, st.target) if st else float("inf")
                    d_other = dist(pos, other_st.target) if other_st else float("inf")
                    if d_current < d_other:
                        # Reassign to current task; remove from other task
                        agent_to_task[aid] = sid
                        kept_agents.append(aid)
                        if other_sid in cleaned and aid in cleaned[other_sid]:
                            cleaned[other_sid].remove(aid)
                            if other_st and not validate_assignment_skills(cleaned[other_sid], other_st, fleet):
                                cleaned[other_sid] = []
                    else:
                        # Keep in other task; drop from current task
                        pass
                else:
                    agent_to_task[aid] = sid
                    kept_agents.append(aid)

            if st and kept_agents and validate_assignment_skills(kept_agents, st, fleet):
                cleaned[sid] = kept_agents
            else:
                cleaned[sid] = []

        return cleaned

    def get_updated_executable_assignments(
        self,
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        lock_threshold: float = 35.0,
    ) -> dict[str, list[str]]:
        """Return executable assignments for uncompleted subtasks, preserving active commitments."""
        if self.active_context is None:
            return {}

        ctx = self.active_context
        completed_ids = ctx.completed_subtask_ids | {s.subtask_id for s in subtasks if s.completed}
        incomplete_subtasks = [
            s for s in subtasks if s.subtask_id not in completed_ids and not s.completed
        ]
        if not incomplete_subtasks:
            return {}

        from src.decomposition.distance_feasible_decomp import validate_assignment_skills, validate_joint_assignment

        # 1. Filter assignments to incomplete subtasks only, validating required skills
        updated_assignments: dict[str, list[str]] = {}
        assigned_agents: set[str] = set()

        agent_map = {a.agent_id: a for a in fleet.agents}
        for s in incomplete_subtasks:
            sid = s.subtask_id
            raw_agents = ctx.assignments.get(sid, [])
            if not raw_agents or any(aid not in agent_map for aid in raw_agents):
                updated_assignments[sid] = []
                continue
            curr_agents = list(raw_agents)
            if (
                curr_agents
                and validate_assignment_skills(curr_agents, s, fleet)
                and not any(aid in assigned_agents for aid in curr_agents)
            ):
                updated_assignments[sid] = curr_agents
                assigned_agents.update(curr_agents)
            else:
                updated_assignments[sid] = []

        # 2. Identify freed / idle agents
        all_agent_ids = set(agent_map.keys())
        freed_agents = all_agent_ids - assigned_agents

        # 3. Lightweight local reassignment for freed agents to incomplete subtasks
        if freed_agents:
            for sid, agents in list(updated_assignments.items()):
                if not agents:
                    st = next((s for s in incomplete_subtasks if s.subtask_id == sid), None)
                    if st:
                        req_skills = set(st.required_skills)
                        # Option A: Single agent covering all required skills and distance feasible
                        eligible_singles = [
                            aid for aid in freed_agents
                            if req_skills.issubset(set(agent_map[aid].skills))
                            and dist(agent_map[aid].position, st.target) <= self.r_reach
                        ]
                        if eligible_singles:
                            best_agent = min(
                                eligible_singles,
                                key=lambda aid: (
                                    dist(agent_map[aid].position, st.target) / max(getattr(fleet.kinematics.get(agent_map[aid].agent_type.value, None), "max_speed", 1.0), 1e-9)
                                    if hasattr(fleet, "kinematics") and fleet.kinematics and agent_map[aid].agent_type.value in fleet.kinematics
                                    else dist(agent_map[aid].position, st.target),
                                    aid,
                                ),
                            )
                            if validate_joint_assignment([best_agent], st, fleet, self.c_task, self.r_reach):
                                updated_assignments[sid] = [best_agent]
                                freed_agents.remove(best_agent)
                                continue

                        # Option B: Complementary pair from freed agents covering all required skills
                        freed_list = sorted(list(freed_agents))
                        best_pair = None
                        best_pair_cost = float("inf")
                        for i in range(len(freed_list)):
                            for j in range(i + 1, len(freed_list)):
                                aid1, aid2 = freed_list[i], freed_list[j]
                                if req_skills.issubset(set(agent_map[aid1].skills) | set(agent_map[aid2].skills)):
                                    if validate_joint_assignment([aid1, aid2], st, fleet, self.c_task, self.r_reach):
                                        d1 = dist(agent_map[aid1].position, st.target)
                                        d2 = dist(agent_map[aid2].position, st.target)
                                        cost = d1 + d2
                                        if cost < best_pair_cost:
                                            best_pair_cost = cost
                                            best_pair = [aid1, aid2]
                        if best_pair:
                            updated_assignments[sid] = best_pair
                            freed_agents.remove(best_pair[0])
                            freed_agents.remove(best_pair[1])
                            continue

        # 4. Apply Target Commitment Locking
        updated_assignments = self.apply_target_commitment_lock(
            updated_assignments, ctx.assignments, fleet, subtasks, lock_threshold
        )

        # 5. Clean duplicate assignments across all tasks, including when some tasks are empty
        updated_assignments = self.clean_duplicate_assignments(
            updated_assignments, incomplete_subtasks, fleet
        )

        # 6. Revalidate every final assignment before returning; invalid teams become []
        for sid, agents in list(updated_assignments.items()):
            if agents:
                st = next((s for s in incomplete_subtasks if s.subtask_id == sid), None)
                if not st or not validate_joint_assignment(agents, st, fleet, self.c_task, self.r_reach):
                    updated_assignments[sid] = []

        # Update active context with newly updated execution assignments
        ctx.assignments = updated_assignments
        ctx.completed_subtask_ids.update({s.subtask_id for s in subtasks if s.completed})
        return updated_assignments


    def can_continue_plan(
        self,
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        cqi_matrix: np.ndarray | None = None,
        sys_cqi: float = 1.0,
        packet_loss: float = 0.0,
        latency: float = 0.0,
    ) -> bool:
        """Return True if active plan validity score exceeds threshold."""
        if self.active_context is None:
            return False

        score = self.evaluate_plan_validity(
            fleet, subtasks, cqi_matrix, sys_cqi, packet_loss, latency
        )
        if not score.is_valid:
            return False

        # Dynamically refresh Layer 2 execution assignments for incomplete subtasks
        self.get_updated_executable_assignments(fleet, subtasks)
        return True

