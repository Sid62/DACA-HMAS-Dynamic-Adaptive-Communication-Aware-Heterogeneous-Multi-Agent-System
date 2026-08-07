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

    @property
    def is_valid(self) -> bool:
        return self.total_validity_score >= 0.75


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

    def evaluate_plan_validity(
        self,
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        cqi_matrix: np.ndarray | None = None,
        sys_cqi: float = 1.0,
        packet_loss: float = 0.0,
        latency: float = 0.0,
    ) -> PlanValidityScore:
        """Compute quantitative Plan Validity Score (V_plan)."""
        if self.active_context is None or not self.active_context.assignments:
            return PlanValidityScore(total_validity_score=0.0)

        ctx = self.active_context
        incomplete_subtasks = [s for s in subtasks if not s.completed]
        if not incomplete_subtasks:
            return PlanValidityScore(total_validity_score=1.0)

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

        for st in subtasks:
            if st.completed:
                continue
            sid = st.subtask_id
            prev_agents = previous_assignments.get(sid, [])
            if prev_agents:
                aid = prev_agents[0]
                if aid in agent_map:
                    agent = agent_map[aid]
                    if dist(agent.position, st.target) < lock_threshold:
                        # Lock agent to this subtask
                        locked_assignments[sid] = [aid]
                        locked_agent_ids.add(aid)

        # Clear duplicate assignments for locked agents in other subtasks
        for sid, agents in locked_assignments.items():
            st = next((s for s in subtasks if s.subtask_id == sid), None)
            if st and st.completed:
                continue
            curr_agents = [aid for aid in agents if aid not in locked_agent_ids or aid in previous_assignments.get(sid, [])]
            if curr_agents:
                locked_assignments[sid] = curr_agents

        return locked_assignments

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
        incomplete_subtasks = [s for s in subtasks if not s.completed]
        if not incomplete_subtasks:
            return {}

        # 1. Filter assignments to incomplete subtasks only
        updated_assignments: dict[str, list[str]] = {}
        assigned_agents: set[str] = set()

        agent_map = {a.agent_id: a for a in fleet.agents}
        for s in incomplete_subtasks:
            sid = s.subtask_id
            curr_agents = [aid for aid in ctx.assignments.get(sid, []) if aid in agent_map]
            if curr_agents:
                updated_assignments[sid] = curr_agents
                assigned_agents.update(curr_agents)
            else:
                updated_assignments[sid] = []

        # 2. Identify freed / idle agents
        all_agent_ids = set(agent_map.keys())
        freed_agents = all_agent_ids - assigned_agents

        # 3. Lightweight local reassignment for freed agents to incomplete subtasks
        if freed_agents:
            for sid, agents in updated_assignments.items():
                if not agents:
                    st = next((s for s in incomplete_subtasks if s.subtask_id == sid), None)
                    if st:
                        eligible = [
                            aid for aid in freed_agents
                            if set(st.required_skills).issubset(set(agent_map[aid].skills)) or not st.required_skills
                        ]
                        if not eligible:
                            eligible = list(freed_agents)
                        if eligible:
                            best_agent = min(
                                eligible, key=lambda aid: dist(agent_map[aid].position, st.target)
                            )
                            updated_assignments[sid] = [best_agent]
                            freed_agents.remove(best_agent)

        # 4. Apply Target Commitment Locking
        updated_assignments = self.apply_target_commitment_lock(
            updated_assignments, ctx.assignments, fleet, subtasks, lock_threshold
        )

        # Update active context with newly updated execution assignments
        ctx.assignments = updated_assignments
        ctx.completed_subtask_ids = {s.subtask_id for s in subtasks if s.completed}
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

