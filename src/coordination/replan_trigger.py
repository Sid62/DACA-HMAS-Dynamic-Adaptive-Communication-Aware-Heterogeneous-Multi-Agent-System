"""Event-driven Cloud LLM replanning trigger for DACA-HMAS.

Design principle
-----------------
The Cloud LLM (task decomposition + coalition formation) should only be
re-consulted when the *state it reasoned over* has materially changed in a
way that makes the currently stored plan provably invalid or incomplete.
A fixed replanning interval or a CQI-threshold trigger are both policy
choices about *when it is convenient* to refresh the plan, independent of
whether the existing plan is still correct -- neither is used here.

Every trigger below is a correctness condition: if it fires, continuing to
execute the stored plan would either (a) ignore new mission scope, (b)
leave freed agent capacity idle, or (c) execute an assignment that is no
longer feasible. If none fire, the stored plan is still a valid answer to
the same planning problem the Cloud LLM already solved, so re-asking it
would return an equivalent result at the cost of an API call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.env.agents import AgentFleet
from src.env.scenarios import Subtask


@dataclass
class PlanState:
    """Snapshot of what the most recently accepted Cloud LLM plan was
    reasoned over. Compared against current state each step to decide
    whether that plan is still valid, without contacting the Cloud LLM.
    """

    initialized: bool = False
    known_subtask_ids: set[str] = field(default_factory=set)
    known_completed_ids: set[str] = field(default_factory=set)
    known_agent_ids: set[str] = field(default_factory=set)
    coalition_members: dict[Any, frozenset[str]] = field(default_factory=dict)
    subtask_required_skills: dict[str, frozenset[str]] = field(default_factory=dict)
    subtask_coalition: dict[str, Any] = field(default_factory=dict)
    subtask_skill_satisfied_at_plan: dict[str, bool] = field(default_factory=dict)
    known_mode: int = -1
    known_sys_cqi: float = 1.0
    cqi_ema: float = 1.0
    cqi_drift_steps: int = 0
    known_packet_loss: float = 0.0
    known_latency: float = 0.0
    last_replan_step: int = -9999


class ReplanDecision(tuple):
    """2-tuple (replan_now: bool, reason: str) with additional metadata attributes for backward compatibility.

    Attributes:
        replan_now (bool): True if a replanning operation is needed.
        reason (str): Identifier or description of the triggering event.
        scope (str): 'global' (requires Cloud LLM central planner),
                     'local' (resolved locally by Device LLM / PlanContinuity local repair),
                     'none' (no action needed, continue existing plan).
    """
    replan_now: bool
    reason: str
    scope: str

    def __new__(cls, replan_now: bool, reason: str, scope: str = "none"):
        return super().__new__(cls, (replan_now, reason))

    def __init__(self, replan_now: bool, reason: str, scope: str = "none"):
        self.replan_now = replan_now
        self.reason = reason
        self.scope = scope

    def __bool__(self) -> bool:
        return self.replan_now


def should_replan(
    plan_state: PlanState,
    subtasks: list[Subtask],
    fleet: AgentFleet,
    coalitions: list[dict[str, Any]],
    mode: int = 0,
    sys_cqi: float = 1.0,
    packet_loss: float = 0.0,
    latency: float = 0.0,
    cqi_delta_threshold: float = 0.08,
    packet_loss_threshold: float = 0.3,
    latency_threshold: float = 0.5,
    current_step: int = 0,
    minimum_replanning_interval: int = 0,
    continuity_engine: Any | None = None,
    cqi_matrix: Any | None = None,
) -> ReplanDecision:
    """Return ReplanDecision(replan_now, reason, scope) where scope is 'global',
    'local', or 'none'. For backward compatibility, this evaluates as a 2-tuple
    (replan_now, reason) when unpacked.
    """

    # --- Trigger 1: Mission initialization --------------------------------
    # No plan exists yet, so there is nothing to reuse. This is the only
    # unconditional call: an initial decomposition/coalition assignment is
    # a prerequisite for any execution at all, not an optimization choice.
    if not plan_state.initialized:
        return ReplanDecision(True, "mission_initialization", scope="global")

    # --- Trigger 1b: Architecture switch -----------------------------------
    # Evaluate Plan Continuity on architecture switch (Centralized <-> Decentralized).
    # If the active plan is still valid (V_plan >= threshold), preserve and continue execution!
    if plan_state.known_mode == -1:
        plan_state.known_mode = mode
    elif mode != plan_state.known_mode:
        if continuity_engine is not None:
            if continuity_engine.can_continue_plan(
                fleet, subtasks, cqi_matrix, sys_cqi, packet_loss, latency
            ):
                plan_state.known_mode = mode  # Record absorbed switch to prevent re-triggering
                return ReplanDecision(False, "", scope="none")
        return ReplanDecision(True, f"architecture_switched:{plan_state.known_mode}->{mode}", scope="global")

    # --- Rate limiter: everything below this line is subject to the
    # minimum replanning interval. Architecture switch above is exempt
    # because it is a safety/consistency requirement.
    steps_since_replan = current_step - plan_state.last_replan_step
    if minimum_replanning_interval > 0 and steps_since_replan < minimum_replanning_interval:
        return ReplanDecision(False, "", scope="none")

    # --- Trigger 1g: Active plan capacity violation (duplicate agents across tasks) ---
    # Subject to minimum_replanning_interval cooldown. A duplicate-agent assignment
    # must first be treated as a potentially local execution-layer problem. If the
    # assignment can be repaired locally using the existing global plan, reuse the
    # global plan without triggering Cloud planning.
    if continuity_engine is not None and continuity_engine.active_context is not None:
        active_assignments = continuity_engine.active_context.assignments
        incomplete_sids = {s.subtask_id for s in subtasks if not s.completed}
        seen_agents = set()
        capacity_violated = False
        for sid in incomplete_sids:
            aids = active_assignments.get(sid, [])
            for aid in aids:
                if aid in seen_agents:
                    capacity_violated = True
                    break
                seen_agents.add(aid)
            if capacity_violated:
                break
        if capacity_violated:
            repaired = continuity_engine.get_updated_executable_assignments(fleet, subtasks)
            has_executable_work = any(len(aids) > 0 for aids in repaired.values())
            if has_executable_work:
                return ReplanDecision(True, "active_plan_capacity_violation_repaired_locally", scope="local")
            else:
                return ReplanDecision(True, "active_plan_capacity_violation_duplicate_agent", scope="global")

    # --- Trigger 1c: Communication quality changed significantly ----------
    # Filter high-frequency wireless noise using Exponential Moving Average (EMA)
    # and require sustained drift over a 3-step persistence window before triggering a replan.
    alpha = 0.3
    if plan_state.cqi_ema is None:
        plan_state.cqi_ema = sys_cqi
    else:
        plan_state.cqi_ema = alpha * sys_cqi + (1.0 - alpha) * plan_state.cqi_ema

    cqi_delta = abs(plan_state.cqi_ema - plan_state.known_sys_cqi)
    if cqi_delta > cqi_delta_threshold:
        plan_state.cqi_drift_steps += 1
        if plan_state.cqi_drift_steps >= 3:
            # Check if active plan remains valid despite CQI drift
            if continuity_engine is not None and continuity_engine.can_continue_plan(
                fleet, subtasks, cqi_matrix, sys_cqi, packet_loss, latency
            ):
                return ReplanDecision(False, "", scope="none")
            return ReplanDecision(
                True,
                f"cqi_changed_significantly:{plan_state.known_sys_cqi:.3f}->{sys_cqi:.3f} (ema={plan_state.cqi_ema:.3f})",
                scope="global",
            )
    else:
        plan_state.cqi_drift_steps = 0

    # --- Trigger 1d: Packet loss crossed threshold -------------------------
    if (plan_state.known_packet_loss < packet_loss_threshold <= packet_loss) or (
        plan_state.known_packet_loss >= packet_loss_threshold > packet_loss
    ):
        if continuity_engine is not None and continuity_engine.can_continue_plan(
            fleet, subtasks, cqi_matrix, sys_cqi, packet_loss, latency
        ):
            plan_state.known_packet_loss = packet_loss
            return ReplanDecision(False, "", scope="none")
        return ReplanDecision(True, f"packet_loss_crossed_threshold:{packet_loss:.3f}", scope="global")

    # --- Trigger 1e: Latency crossed threshold ------------------------------
    if (plan_state.known_latency < latency_threshold <= latency) or (
        plan_state.known_latency >= latency_threshold > latency
    ):
        if continuity_engine is not None and continuity_engine.can_continue_plan(
            fleet, subtasks, cqi_matrix, sys_cqi, packet_loss, latency
        ):
            plan_state.known_latency = latency
            return ReplanDecision(False, "", scope="none")
        return ReplanDecision(True, f"latency_crossed_threshold:{latency:.3f}", scope="global")

    # --- Trigger 1f: Agent battery crossed threshold ------------------------
    for agent in fleet.agents:
        level = getattr(agent, "battery_level", None)
        if level is not None and level < 20.0:
            return ReplanDecision(True, f"agent_battery_low:{agent.agent_id}", scope="global")

    # --- Trigger 2: New task/subtask discovered ----------------------------
    current_subtask_ids = {s.subtask_id for s in subtasks}
    new_ids = current_subtask_ids - plan_state.known_subtask_ids
    if new_ids:
        return ReplanDecision(True, f"new_subtask_discovered:{sorted(new_ids)}", scope="global")

    # --- Trigger 3: Task completed, remaining work needs reassignment -----
    # Completing a subtask frees the agent(s) that were working it.
    current_completed_ids = {s.subtask_id for s in subtasks if s.completed}
    newly_completed = current_completed_ids - plan_state.known_completed_ids
    if newly_completed:
        incomplete_subtasks = [s for s in subtasks if not s.completed]
        if not incomplete_subtasks:
            plan_state.known_completed_ids.update(newly_completed)
            return ReplanDecision(False, "", scope="none")

        if continuity_engine is not None:
            active_assignments = (
                continuity_engine.active_context.assignments
                if continuity_engine.active_context
                else {}
            )
            unassigned_pending = [
                s for s in incomplete_subtasks if not active_assignments.get(s.subtask_id)
            ]

            # If no unassigned pending tasks and plan is healthy, continue locally
            if not unassigned_pending and continuity_engine.can_continue_plan(
                fleet, subtasks, cqi_matrix, sys_cqi, packet_loss, latency
            ):
                plan_state.known_completed_ids.update(newly_completed)
                return ReplanDecision(False, "", scope="none")

            # Check if local reassignment can assign freed agents without Cloud
            updated = continuity_engine.get_updated_executable_assignments(fleet, subtasks)
            all_pending_covered = all(
                len(updated.get(s.subtask_id, [])) > 0 for s in incomplete_subtasks
            )
            if all_pending_covered and continuity_engine.can_continue_plan(
                fleet, subtasks, cqi_matrix, sys_cqi, packet_loss, latency
            ):
                plan_state.known_completed_ids.update(newly_completed)
                return ReplanDecision(
                    True,
                    f"task_completed_local_reallocation:{sorted(newly_completed)}",
                    scope="local",
                )

        # Incomplete subtasks require global re-allocation by Cloud planner
        plan_state.known_completed_ids.update(newly_completed)
        return ReplanDecision(
            True,
            f"task_completed_needs_reassignment:{sorted(newly_completed)}",
            scope="global",
        )

    # --- Trigger 4: Coalition invalidated by agent unavailability ---------
    current_agent_ids = {a.agent_id for a in fleet.agents}
    missing_agents = plan_state.known_agent_ids - current_agent_ids
    if missing_agents:
        affected = [
            cid
            for cid, members in plan_state.coalition_members.items()
            if members & missing_agents
        ]
        if affected:
            return ReplanDecision(True, f"coalition_invalidated_agent_unavailable:{affected}", scope="global")

    # --- Trigger 5: Coalition can no longer satisfy required skills -------
    agent_skills = {a.agent_id: set(a.skills) for a in fleet.agents}
    for subtask_id, coalition_id in plan_state.subtask_coalition.items():
        if not plan_state.subtask_skill_satisfied_at_plan.get(subtask_id, False):
            continue
        required = plan_state.subtask_required_skills.get(subtask_id)
        members = plan_state.coalition_members.get(coalition_id)
        if not required or not members:
            continue
        current_skills: set[str] = set()
        for mid in members:
            current_skills.update(agent_skills.get(mid, set()))
        if not required.issubset(current_skills):
            return ReplanDecision(True, f"subtask_skills_no_longer_satisfied:{subtask_id}", scope="global")

    # --- Trigger 7: Coalition membership changed underneath the plan ------
    current_groups = {frozenset(c.get("members", [])) for c in coalitions}
    known_groups = set(plan_state.coalition_members.values())
    if current_groups and current_groups != known_groups:
        return ReplanDecision(True, "coalition_membership_changed", scope="global")

    return ReplanDecision(False, "", scope="none")


def update_plan_state(
    plan_state: PlanState,
    subtasks: list[Subtask],
    fleet: AgentFleet,
    coalitions: list[dict[str, Any]],
    assignments: dict[str, list[str]],
    mode: int = 0,
    sys_cqi: float = 1.0,
    packet_loss: float = 0.0,
    latency: float = 0.0,
    current_step: int = 0,
) -> None:
    """Record what the plan just returned by the Cloud LLM was reasoned
    over, so the next should_replan() call has a correct baseline to diff
    against. Called only immediately after a Cloud LLM call is accepted.
    """
    plan_state.initialized = True
    plan_state.last_replan_step = current_step
    plan_state.known_mode = mode
    plan_state.known_sys_cqi = sys_cqi
    plan_state.cqi_ema = sys_cqi
    plan_state.cqi_drift_steps = 0
    plan_state.known_packet_loss = packet_loss
    plan_state.known_latency = latency
    plan_state.known_subtask_ids = {s.subtask_id for s in subtasks}
    plan_state.known_completed_ids = {s.subtask_id for s in subtasks if s.completed}
    plan_state.coalition_members = {
        c.get("coalition_id"): frozenset(c.get("members", [])) for c in coalitions
    }
    plan_state.subtask_required_skills = {
        s.subtask_id: frozenset(s.required_skills) for s in subtasks
    }
    agent_to_coalition = {
        member: cid
        for cid, members in plan_state.coalition_members.items()
        for member in members
    }
    plan_state.subtask_coalition = {}
    plan_state.subtask_skill_satisfied_at_plan = {}
    agent_skills = {a.agent_id: set(a.skills) for a in fleet.agents}
    for sid, agent_list in assignments.items():
        if not agent_list:
            continue
        cid = agent_to_coalition.get(agent_list[0])
        if cid is None:
            continue
        plan_state.subtask_coalition[sid] = cid
        required = plan_state.subtask_required_skills.get(sid, frozenset())
        members = plan_state.coalition_members.get(cid, frozenset())
        covered: set[str] = set()
        for member_id in members:
            covered |= agent_skills.get(member_id, set())
        plan_state.subtask_skill_satisfied_at_plan[sid] = required.issubset(covered)