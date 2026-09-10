"""Regression tests for Issue 1: Validate assignments and task completion.

Tests cover:
1. valid agent + correct required skill -> valid
2. valid agent + missing required skill -> invalid
3. multiple agents collectively covering required skills -> valid
4. multiple agents missing one required skill -> invalid
5. unknown agent ID -> invalid
6. empty coalition -> invalid
7. unknown-only coalition -> invalid
8. multi-agent task where only first agent reaches target but required team is not valid -> NOT complete
9. valid team reaches completion conditions -> complete
10. repaired/reused assignments are validated with the same skill constraints
"""

import numpy as np
import pytest

from src.coalition.feasibility import (
    build_psi_matrix,
    validate_coalition_members,
)
from src.decomposition.distance_feasible_decomp import (
    validate_assignment_skills,
    validate_joint_assignment,
    validate_task_completion,
)
from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position, dist
from src.env.scenarios import Subtask
from src.coordination.plan_continuity import PlanContinuityEngine


KIN = {
    "uav": KinematicsConfig(15.0, 1.5),
    "vehicle": KinematicsConfig(10.0, 0.8),
    "robot": KinematicsConfig(3.0, 2.0),
}


@pytest.fixture
def mock_fleet():
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["transport", "navigate"]),
        AgentState("vehicle_0", AgentType.VEHICLE, Position(15.0, 15.0), skills=["transport", "inspect"]),
        AgentState("robot_0", AgentType.ROBOT, Position(20.0, 20.0), skills=["lift", "rescue"]),
    ]
    return AgentFleet(agents, KIN)


# ── Test 1: Valid agent + correct required skill -> valid ─────────────────────

def test_valid_agent_with_correct_skill(mock_fleet):
    subtask = Subtask("T_0", "delivery", Position(12.0, 12.0), required_skills=["navigate"])
    assert validate_assignment_skills(["uav_0"], subtask, mock_fleet) is True
    assert validate_joint_assignment(["uav_0"], subtask, mock_fleet, c_task=30.0, r_reach=100.0) is True


# ── Test 2: Valid agent + missing required skill -> invalid ───────────────────

def test_valid_agent_with_missing_skill(mock_fleet):
    subtask = Subtask("T_0", "heavy lift", Position(12.0, 12.0), required_skills=["lift"])
    # uav_0 has ["transport", "navigate"], lacks "lift"
    assert validate_assignment_skills(["uav_0"], subtask, mock_fleet) is False
    assert validate_joint_assignment(["uav_0"], subtask, mock_fleet, c_task=30.0, r_reach=100.0) is False


# ── Test 3: Multiple agents collectively covering required skills -> valid ────

def test_multiple_agents_collectively_covering_skills(mock_fleet):
    # requires lift + transport: robot_0 has lift, vehicle_0 has transport
    subtask = Subtask("T_0", "lift and transport", Position(16.0, 16.0), required_skills=["lift", "transport"])
    assert validate_assignment_skills(["robot_0", "vehicle_0"], subtask, mock_fleet) is True
    assert validate_joint_assignment(["robot_0", "vehicle_0"], subtask, mock_fleet, c_task=30.0, r_reach=100.0) is True


# ── Test 4: Multiple agents missing one required skill -> invalid ─────────────

def test_multiple_agents_missing_one_skill(mock_fleet):
    # requires lift, transport, and sense: robot_0 has lift, vehicle_0 has transport, neither has sense
    subtask = Subtask("T_0", "complex", Position(16.0, 16.0), required_skills=["lift", "transport", "sense"])
    assert validate_assignment_skills(["robot_0", "vehicle_0"], subtask, mock_fleet) is False
    assert validate_joint_assignment(["robot_0", "vehicle_0"], subtask, mock_fleet, c_task=30.0, r_reach=100.0) is False


# ── Test 5: Unknown agent ID -> invalid ───────────────────────────────────────

def test_unknown_agent_id_fails_validation(mock_fleet):
    subtask = Subtask("T_0", "test", Position(12.0, 12.0), required_skills=["navigate"])
    assert validate_assignment_skills(["unknown_999"], subtask, mock_fleet) is False
    assert validate_joint_assignment(["unknown_999"], subtask, mock_fleet, c_task=30.0, r_reach=100.0) is False
    # Mixture of valid and unknown agent ID must also fail
    assert validate_assignment_skills(["uav_0", "unknown_999"], subtask, mock_fleet) is False
    assert validate_joint_assignment(["uav_0", "unknown_999"], subtask, mock_fleet, c_task=30.0, r_reach=100.0) is False


# ── Test 6: Empty coalition -> invalid ────────────────────────────────────────

def test_empty_coalition_fails_validation(mock_fleet):
    psi = np.ones((3, 3))
    id_to_idx = {a.agent_id: i for i, a in enumerate(mock_fleet.agents)}
    # Empty coalition must return False
    assert validate_coalition_members([], id_to_idx, psi, gamma_min=0.3) is False

    # Empty assignment must also fail assignment validation
    subtask = Subtask("T_0", "test", Position(12.0, 12.0), required_skills=["navigate"])
    assert validate_assignment_skills([], subtask, mock_fleet) is False
    assert validate_joint_assignment([], subtask, mock_fleet, c_task=30.0, r_reach=100.0) is False


# ── Test 7: Unknown-only coalition -> invalid ─────────────────────────────────

def test_unknown_only_coalition_fails_validation(mock_fleet):
    psi = np.ones((3, 3))
    id_to_idx = {a.agent_id: i for i, a in enumerate(mock_fleet.agents)}
    # All-unknown coalition must return False, not fall through with empty indices
    assert validate_coalition_members(["ghost_0", "ghost_1"], id_to_idx, psi, gamma_min=0.3) is False
    # Partially unknown coalition must also return False
    assert validate_coalition_members(["uav_0", "ghost_1"], id_to_idx, psi, gamma_min=0.3) is False


# ── Test 8: Multi-agent task: first agent arrives but team not valid/arrived -> NOT complete ──

def test_multiagent_completion_fails_if_team_incomplete_or_invalid():
    # Subtask requires ["lift", "transport"]
    subtask = Subtask("T_0", "cooperative", Position(10.0, 10.0), required_skills=["lift", "transport"])
    
    # Case 8a: Single agent arrives at target, but lacks "lift" (only has transport)
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["transport", "navigate"]),
        AgentState("robot_0", AgentType.ROBOT, Position(50.0, 50.0), skills=["lift"]),
    ], KIN)
    # Only uav_0 assigned -> missing "lift" -> cannot complete even though at target (dist=0)
    assert validate_task_completion(["uav_0"], subtask, fleet, completion_radius=8.0) is False

    # Case 8b: Both agents assigned, covering skills, but robot_0 is still 40m away
    # First agent (uav_0) is at target (dist=0), but second agent (robot_0) is not at target
    assert validate_task_completion(["uav_0", "robot_0"], subtask, fleet, completion_radius=8.0) is False


# ── Test 9: Valid team reaches completion conditions -> complete ──────────────

def test_valid_team_reaches_completion(mock_fleet):
    subtask = Subtask("T_0", "cooperative", Position(10.0, 10.0), required_skills=["lift", "transport"])
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(12.0, 12.0), skills=["transport", "navigate"]),
        AgentState("robot_0", AgentType.ROBOT, Position(14.0, 14.0), skills=["lift"]),
    ], KIN)
    # Both agents are within 8.0m of (10.0, 10.0):
    # dist(uav_0) = sqrt(4+4) = 2.83m < 8.0m
    # dist(robot_0) = sqrt(16+16) = 5.66m < 8.0m
    # Collective skills cover ["lift", "transport"]
    assert validate_task_completion(["uav_0", "robot_0"], subtask, fleet, completion_radius=8.0) is True


# ── Test 10: Repaired/reused assignments validated with same constraints ───────

def test_reused_repaired_assignments_enforce_skills(mock_fleet):
    continuity = PlanContinuityEngine()
    subtasks = [
        Subtask("T_0", "navigate task", Position(12.0, 12.0), required_skills=["navigate"]),
        Subtask("T_1", "lift task", Position(22.0, 22.0), required_skills=["lift"]),
    ]
    # Set active plan where T_0 has uav_0 (valid) and T_1 has uav_0 (INVALID: lacks lift)
    continuity.set_active_plan(
        assignments={"T_0": ["uav_0"], "T_1": ["uav_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0", "robot_0"]}],
        subtasks=subtasks,
        mode=0,
    )
    # get_updated_executable_assignments must drop invalid assignment T_1
    updated = continuity.get_updated_executable_assignments(mock_fleet, subtasks)
    assert updated.get("T_0") == ["uav_0"]
    # T_1 should either be reassigned to robot_0 (which has lift) or left unassigned, NEVER kept as uav_0
    assert "uav_0" not in updated.get("T_1", [])
    if updated.get("T_1"):
        assert "robot_0" in updated["T_1"]
