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
    known_packet_loss: float = 0.0
    known_latency: float = 0.0


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
) -> tuple[bool, str]:
    """Return (True, reason) iff a mission event has occurred that can
    invalidate the currently stored plan; (False, "") otherwise, in which
    case the caller MUST reuse the existing plan and MUST NOT call the
    Cloud LLM this step.
    """

    # --- Trigger 1: Mission initialization --------------------------------
    # No plan exists yet, so there is nothing to reuse. This is the only
    # unconditional call: an initial decomposition/coalition assignment is
    # a prerequisite for any execution at all, not an optimization choice.
    if not plan_state.initialized:
        return True, "mission_initialization"
     # --- Trigger 1b: Architecture switch -----------------------------------
    # Centralized <-> decentralized changes who is allowed to plan; the
    # stored plan was produced under the previous architecture's authority
    # and must be re-derived, not merely re-executed.
    if mode != plan_state.known_mode:
        return True, f"architecture_switched:{plan_state.known_mode}->{mode}"

    # --- Trigger 1c: Communication quality changed significantly ----------
    if abs(sys_cqi - plan_state.known_sys_cqi) > cqi_delta_threshold:
        return True, (
            f"cqi_changed_significantly:{plan_state.known_sys_cqi:.3f}->{sys_cqi:.3f}"
        )
    # --- Trigger 1d: Packet loss crossed threshold -------------------------
    if (plan_state.known_packet_loss < packet_loss_threshold <= packet_loss) or (
        plan_state.known_packet_loss >= packet_loss_threshold > packet_loss
    ):
        return True, f"packet_loss_crossed_threshold:{packet_loss:.3f}"

    # --- Trigger 1e: Latency crossed threshold ------------------------------
    if (plan_state.known_latency < latency_threshold <= latency) or (
        plan_state.known_latency >= latency_threshold > latency
    ):
        return True, f"latency_crossed_threshold:{latency:.3f}"

    # --- Trigger 1f: Agent battery crossed threshold ------------------------
    # Inert today (battery.enabled: false in config, and AgentState carries
    # no battery attribute yet) -- present so a future battery-aware fleet
    # is covered automatically without touching this function again, same
    # pattern as Trigger 2 below for dynamic task generation.
    for agent in fleet.agents:
        level = getattr(agent, "battery_level", None)
        if level is not None and level < 20.0:
            return True, f"agent_battery_low:{agent.agent_id}"

    # --- Trigger 2: New task/subtask discovered ----------------------------
    # A subtask the Cloud LLM never reasoned about cannot be represented in
    # the existing decomposition by construction; reusing the stored plan
    # would silently drop it from the mission. Static scenarios in this
    # codebase never add subtasks after initialization, so this trigger is
    # inert today -- it exists so a future dynamic-task-generator feature
    # is covered automatically, without touching this function again.
    current_subtask_ids = {s.subtask_id for s in subtasks}
    new_ids = current_subtask_ids - plan_state.known_subtask_ids
    if new_ids:
        return True, f"new_subtask_discovered:{sorted(new_ids)}"

    # --- Trigger 3: Task completed, remaining work needs reassignment -----
    # Completing a subtask frees the agent(s) that were working it. The
    # stored plan was computed assuming they were still occupied, so it is
    # now stale with respect to the remaining incomplete subtasks -- this
    # is a genuine change to the planning input, not a scheduled refresh.
    current_completed_ids = {s.subtask_id for s in subtasks if s.completed}
    newly_completed = current_completed_ids - plan_state.known_completed_ids
    if newly_completed:
        return True, f"task_completed_needs_reassignment:{sorted(newly_completed)}"

    # --- Trigger 4: Coalition invalidated by agent unavailability ---------
    # If an agent a coalition depends on is no longer present in the fleet
    # (failed / left / unavailable), that coalition's membership no longer
    # matches what the Cloud LLM reasoned over. Continuing to execute it
    # would assign work to an agent that doesn't exist, which corrupts
    # results rather than testing the planner under real conditions.
    current_agent_ids = {a.agent_id for a in fleet.agents}
    missing_agents = plan_state.known_agent_ids - current_agent_ids
    if missing_agents:
        affected = [
            cid
            for cid, members in plan_state.coalition_members.items()
            if members & missing_agents
        ]
        if affected:
            return True, f"coalition_invalidated_agent_unavailable:{affected}"

    # --- Trigger 5: Coalition can no longer satisfy required skills -------
    # A coalition is only a valid solution to a subtask if its members'
    # combined skills cover that subtask's required_skills. This trigger
    # fires only on a *regression*: coverage held when the plan was made
    # but no longer holds now (e.g. a skilled member became unavailable
    # and Trigger 4 didn't already catch it, or a member's skill set
    # changed). It deliberately does NOT fire for a gap that already
    # existed at planning time -- the decomposition/coalition-formation
    # algorithms are unmodified and are not guaranteed to produce full
    # skill coverage in the first place (e.g. the mock planner assigns
    # round-robin without checking skills); treating that pre-existing
    # property as a new "event" every step would defeat event-driven
    # replanning entirely and is not what "no longer satisfy" describes.
    agent_skills = {a.agent_id: set(a.skills) for a in fleet.agents}
    for subtask_id, coalition_id in plan_state.subtask_coalition.items():
        if not plan_state.subtask_skill_satisfied_at_plan.get(subtask_id, False):
            continue  # was already unsatisfied at plan time -- not a new event
        required = plan_state.subtask_required_skills.get(subtask_id)
        members = plan_state.coalition_members.get(coalition_id)
        if not required or not members:
            continue
        covered: set[str] = set()
        for member_id in members:
            covered |= agent_skills.get(member_id, set())
        if not required.issubset(covered):
            return True, f"coalition_skill_regressed:{subtask_id}"

    return False, ""


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
) -> None:
    """Record what the plan just returned by the Cloud LLM was reasoned
    over, so the next should_replan() call has a correct baseline to diff
    against. Called only immediately after a Cloud LLM call is accepted.
    """
    plan_state.initialized = True
    plan_state.known_mode = mode
    plan_state.known_sys_cqi = sys_cqi
    plan_state.known_packet_loss = packet_loss
    plan_state.known_latency = latency
    plan_state.known_subtask_ids = {s.subtask_id for s in subtasks}
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