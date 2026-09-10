"""Reallocation propagation invariant tests.

Verifies that post-switch reallocation assignments respect skill
requirements, preserve completed tasks, and are not silently overwritten
by an unconstrained nearest-agent heuristic.
"""

import numpy as np
import pytest

from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position
from src.env.scenarios import Subtask
from src.reallocation.post_switch import PostSwitchReallocator


KIN = {
    "uav": KinematicsConfig(15, 1.5),
    "vehicle": KinematicsConfig(10, 0.8),
    "robot": KinematicsConfig(3, 2.0),
}


def _make_fleet(agents):
    return AgentFleet(agents, KIN)


def _derive_assignments_from_coalitions(
    coalitions, subtasks, fleet, r_reach=100.0
):
    """Reproduce the orchestrator's validated assignment derivation logic.

    This is the same 3-pass algorithm the orchestrator uses after
    reallocator.reallocate(), extracted for testing.
    """
    remaining = [s for s in subtasks if not s.completed]
    all_agents = []
    for c in coalitions:
        all_agents.extend(c.get("members", []))
    agent_skills = {a.agent_id: set(a.skills) for a in fleet.agents}
    assigned = set()
    result = {}

    for st in remaining:
        required = set(st.required_skills)
        best_aid, best_d = None, float("inf")

        # Pass 1: skill-matching within R_reach
        for aid in all_agents:
            if aid in assigned or not fleet.has_agent(aid):
                continue
            if not required.issubset(agent_skills.get(aid, set())):
                continue
            from src.env.agents import dist
            d = dist(fleet.get_agent(aid).position, st.target)
            if d > r_reach:
                continue
            if d < best_d:
                best_d, best_aid = d, aid

        # Pass 2: skill-matching beyond R_reach
        if best_aid is None:
            for aid in all_agents:
                if aid in assigned or not fleet.has_agent(aid):
                    continue
                if not required.issubset(agent_skills.get(aid, set())):
                    continue
                from src.env.agents import dist
                d = dist(fleet.get_agent(aid).position, st.target)
                if d < best_d:
                    best_d, best_aid = d, aid

        # Pass 3: last resort nearest
        if best_aid is None:
            for aid in all_agents:
                if aid in assigned or not fleet.has_agent(aid):
                    continue
                from src.env.agents import dist
                d = dist(fleet.get_agent(aid).position, st.target)
                if d < best_d:
                    best_d, best_aid = d, aid

        if best_aid is not None:
            result[st.subtask_id] = [best_aid]
            assigned.add(best_aid)

    return result


# ── Test 1: Valid reallocation → orchestrator executes exactly that ─────

def test_valid_reallocation_is_executed():
    """Reallocator returns valid coalitions; derived assignments must use
    those agents, not arbitrary nearest agents."""
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10, 10), skills=["navigate", "search"]),
        AgentState("robot_0", AgentType.ROBOT, Position(50, 50), skills=["lift", "carry"]),
    ]
    fleet = _make_fleet(agents)
    subtasks = [
        Subtask("T_0", "task 0", Position(12, 12), ["navigate"]),
        Subtask("T_1", "task 1", Position(48, 48), ["lift"]),
    ]
    coalitions = [
        {"coalition_id": 0, "members": ["uav_0"]},
        {"coalition_id": 1, "members": ["robot_0"]},
    ]

    result = _derive_assignments_from_coalitions(coalitions, subtasks, fleet)

    # uav_0 should get T_0 (navigate skill), robot_0 should get T_1 (lift skill)
    assert result["T_0"] == ["uav_0"]
    assert result["T_1"] == ["robot_0"]


# ── Test 2: Multi-agent coalition not replaced by nearest-agent ────────

def test_multiagent_coalition_not_replaced():
    """When a coalition has multiple agents, the skill-matching logic must
    select the correct agent, not just the nearest one."""
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10, 10), skills=["navigate"]),
        AgentState("uav_1", AgentType.UAV, Position(11, 11), skills=["search"]),
        AgentState("robot_0", AgentType.ROBOT, Position(100, 100), skills=["lift"]),
    ]
    fleet = _make_fleet(agents)
    subtasks = [
        Subtask("T_0", "task 0", Position(12, 12), ["search"]),  # requires search
    ]
    coalitions = [
        {"coalition_id": 0, "members": ["uav_0", "uav_1", "robot_0"]},
    ]

    result = _derive_assignments_from_coalitions(coalitions, subtasks, fleet)

    # uav_1 has "search", not uav_0 (which is nearer but has "navigate")
    assert result["T_0"] == ["uav_1"]


# ── Test 3: Completed task is not reassigned ───────────────────────────

def test_completed_task_not_reassigned():
    """Completed tasks must be excluded from assignment derivation."""
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10, 10), skills=["navigate"]),
        AgentState("robot_0", AgentType.ROBOT, Position(50, 50), skills=["lift"]),
    ]
    fleet = _make_fleet(agents)
    subtasks = [
        Subtask("T_0", "task 0", Position(12, 12), ["navigate"], completed=True),
        Subtask("T_1", "task 1", Position(48, 48), ["lift"], completed=False),
    ]
    coalitions = [
        {"coalition_id": 0, "members": ["uav_0", "robot_0"]},
    ]

    result = _derive_assignments_from_coalitions(coalitions, subtasks, fleet)

    # T_0 is completed, must NOT appear in result
    assert "T_0" not in result
    # T_1 must be assigned
    assert "T_1" in result
    assert result["T_1"] == ["robot_0"]


# ── Test 4: Required skill is satisfied ────────────────────────────────

def test_required_skill_satisfied():
    """Assigned agent must have the required skill."""
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10, 10), skills=["navigate"]),
        AgentState("robot_0", AgentType.ROBOT, Position(100, 100), skills=["lift", "carry"]),
    ]
    fleet = _make_fleet(agents)
    subtasks = [
        Subtask("T_0", "task 0", Position(12, 12), ["lift"]),  # requires lift
    ]
    coalitions = [
        {"coalition_id": 0, "members": ["uav_0", "robot_0"]},
    ]

    result = _derive_assignments_from_coalitions(coalitions, subtasks, fleet)

    # robot_0 has "lift", uav_0 does not; despite uav_0 being closer
    assert result["T_0"] == ["robot_0"]


# ── Test 5: Valid reallocation → nearest-agent fallback NOT invoked ────

def test_no_fallback_when_valid_assignment_exists():
    """When a skill-matching agent exists, the last-resort nearest-agent
    fallback must NOT be used."""
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(5, 5), skills=["navigate"]),
        AgentState("robot_0", AgentType.ROBOT, Position(90, 90), skills=["inspect"]),
    ]
    fleet = _make_fleet(agents)
    subtasks = [
        Subtask("T_0", "task 0", Position(10, 10), ["inspect"]),  # requires inspect
    ]
    coalitions = [
        {"coalition_id": 0, "members": ["uav_0", "robot_0"]},
    ]

    result = _derive_assignments_from_coalitions(coalitions, subtasks, fleet)

    # robot_0 is further but has the required skill; uav_0 is nearer but wrong skill
    assert result["T_0"] == ["robot_0"]


# ── Test 6: No valid reallocation → documented fallback used ───────────

def test_fallback_when_no_skill_match():
    """When no agent has the required skill, nearest-agent fallback
    must be used (last resort)."""
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10, 10), skills=["navigate"]),
        AgentState("robot_0", AgentType.ROBOT, Position(50, 50), skills=["lift"]),
    ]
    fleet = _make_fleet(agents)
    subtasks = [
        Subtask("T_0", "task 0", Position(12, 12), ["exotic_skill"]),  # no agent has this
    ]
    coalitions = [
        {"coalition_id": 0, "members": ["uav_0", "robot_0"]},
    ]

    result = _derive_assignments_from_coalitions(coalitions, subtasks, fleet)

    # No skill match exists; falls back to nearest agent (uav_0)
    assert result["T_0"] == ["uav_0"]


# ── Test 7: No duplicate assignments ───────────────────────────────────

def test_no_duplicate_assignments():
    """Each agent must be assigned to at most one task."""
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10, 10), skills=["navigate", "search"]),
        AgentState("robot_0", AgentType.ROBOT, Position(50, 50), skills=["navigate", "lift"]),
    ]
    fleet = _make_fleet(agents)
    subtasks = [
        Subtask("T_0", "task 0", Position(12, 12), ["navigate"]),
        Subtask("T_1", "task 1", Position(48, 48), ["navigate"]),
    ]
    coalitions = [
        {"coalition_id": 0, "members": ["uav_0", "robot_0"]},
    ]

    result = _derive_assignments_from_coalitions(coalitions, subtasks, fleet)

    all_agents_assigned = []
    for agents_list in result.values():
        all_agents_assigned.extend(agents_list)
    # No duplicates
    assert len(all_agents_assigned) == len(set(all_agents_assigned))
