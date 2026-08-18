"""Unit tests for conditional post-switch reallocation (Eq 31)."""

import numpy as np
import pytest

from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position
from src.env.scenarios import Subtask
from src.reallocation.post_switch import PostSwitchReallocator


@pytest.fixture
def mock_fleet():
    agents = [
        AgentState(
            "uav_0",
            AgentType.UAV,
            Position(10.0, 10.0, 5.0),
            skills=["navigate", "search"],
            coalition_id=0,
        ),
        AgentState(
            "robot_0",
            AgentType.ROBOT,
            Position(15.0, 15.0, 0.0),
            skills=["lift", "carry"],
            coalition_id=1,
        ),
    ]
    kin = {
        "uav": KinematicsConfig(15, 1.5),
        "vehicle": KinematicsConfig(10, 0.8),
        "robot": KinematicsConfig(3, 2.0),
    }
    return AgentFleet(agents, kin)


@pytest.fixture
def mock_subtasks():
    return [
        Subtask("T_0", "task 0", Position(12.0, 12.0), ["navigate"], completed=False),
        Subtask("T_1", "task 1", Position(16.0, 16.0), ["lift"], completed=False),
    ]


@pytest.fixture
def mock_coalitions():
    return [
        {"coalition_id": 0, "members": ["uav_0"]},
        {"coalition_id": 1, "members": ["robot_0"]},
    ]


@pytest.fixture
def mock_assignments():
    return {
        "T_0": ["uav_0"],
        "T_1": ["robot_0"],
    }


def test_no_trigger_when_mode_not_changed(mock_fleet, mock_subtasks, mock_coalitions, mock_assignments):
    """TEST 1: If architecture mode did not change, should_trigger must return False."""
    realloc = PostSwitchReallocator()
    dist_mat = np.array([[0.0, 7.0], [7.0, 0.0]])
    cqi_mat = np.array([[1.0, 0.9], [0.9, 1.0]])

    assert not realloc.should_trigger(
        mode_changed=False,
        coalitions=mock_coalitions,
        fleet=mock_fleet,
        distance_matrix=dist_mat,
        cqi_matrix=cqi_mat,
        subtasks=mock_subtasks,
        assignments=mock_assignments,
    )
    assert realloc.reallocation_trigger_count == 0
    assert realloc.reallocation_skip_count == 0


def test_skip_reallocation_when_state_valid(mock_fleet, mock_subtasks, mock_coalitions, mock_assignments):
    """TEST 2: If mode changed but state is fully valid, skip reallocation."""
    realloc = PostSwitchReallocator()
    dist_mat = np.array([[0.0, 7.0], [7.0, 0.0]])
    cqi_mat = np.array([[1.0, 0.9], [0.9, 1.0]])

    triggered = realloc.should_trigger(
        mode_changed=True,
        coalitions=mock_coalitions,
        fleet=mock_fleet,
        distance_matrix=dist_mat,
        cqi_matrix=cqi_mat,
        subtasks=mock_subtasks,
        assignments=mock_assignments,
    )
    assert not triggered
    assert realloc.reallocation_trigger_count == 0
    assert realloc.reallocation_skip_count == 1
    assert realloc.last_decision_reason == "state_valid_after_switch"


def test_skip_reallocation_when_all_tasks_completed(mock_fleet, mock_coalitions, mock_assignments):
    """TEST 3: If all tasks are completed, skip reallocation."""
    realloc = PostSwitchReallocator()
    completed_subtasks = [
        Subtask("T_0", "task 0", Position(12.0, 12.0), ["navigate"], completed=True),
        Subtask("T_1", "task 1", Position(16.0, 16.0), ["lift"], completed=True),
    ]
    dist_mat = np.array([[0.0, 7.0], [7.0, 0.0]])
    cqi_mat = np.array([[1.0, 0.9], [0.9, 1.0]])

    triggered = realloc.should_trigger(
        mode_changed=True,
        coalitions=mock_coalitions,
        fleet=mock_fleet,
        distance_matrix=dist_mat,
        cqi_matrix=cqi_mat,
        subtasks=completed_subtasks,
        assignments=mock_assignments,
    )
    assert not triggered
    assert realloc.reallocation_skip_count == 1
    assert realloc.last_decision_reason == "state_valid_after_switch"


def test_trigger_reallocation_on_coalition_infeasibility(mock_fleet, mock_subtasks, mock_assignments):
    """TEST 4: Trigger reallocation if a multi-agent coalition violates gamma_min."""
    realloc = PostSwitchReallocator()
    # Coalition with 2 members
    coalitions = [{"coalition_id": 0, "members": ["uav_0", "robot_0"]}]
    dist_mat = np.array([[0.0, 10.0], [10.0, 0.0]])
    # Degraded CQI between members below gamma_min (0.3)
    cqi_mat = np.array([[1.0, 0.1], [0.1, 1.0]])

    triggered = realloc.should_trigger(
        mode_changed=True,
        coalitions=coalitions,
        fleet=mock_fleet,
        distance_matrix=dist_mat,
        cqi_matrix=cqi_mat,
        subtasks=mock_subtasks,
        assignments=mock_assignments,
        gamma_min=0.3,
    )
    assert triggered
    assert realloc.reallocation_trigger_count == 1
    assert realloc.reallocation_reasons.get("coalition_infeasible") == 1
    assert realloc.last_decision_reason == "coalition_infeasible"


def test_trigger_reallocation_on_agent_unavailable(mock_fleet, mock_subtasks, mock_coalitions):
    """TEST 5: Trigger reallocation if an assigned agent is missing from the fleet."""
    realloc = PostSwitchReallocator()
    dist_mat = np.array([[0.0, 7.0], [7.0, 0.0]])
    cqi_mat = np.array([[1.0, 0.9], [0.9, 1.0]])
    corrupted_assignments = {
        "T_0": ["uav_999"],  # Non-existent agent
        "T_1": ["robot_0"],
    }

    triggered = realloc.should_trigger(
        mode_changed=True,
        coalitions=mock_coalitions,
        fleet=mock_fleet,
        distance_matrix=dist_mat,
        cqi_matrix=cqi_mat,
        subtasks=mock_subtasks,
        assignments=corrupted_assignments,
    )
    assert triggered
    assert realloc.reallocation_trigger_count == 1
    assert realloc.reallocation_reasons.get("agent_unavailable") == 1
    assert realloc.last_decision_reason == "agent_unavailable"


def test_trigger_reallocation_on_uncovered_task(mock_fleet, mock_subtasks, mock_coalitions):
    """TEST 6: Trigger reallocation if a remaining subtask is unassigned."""
    realloc = PostSwitchReallocator()
    dist_mat = np.array([[0.0, 7.0], [7.0, 0.0]])
    cqi_mat = np.array([[1.0, 0.9], [0.9, 1.0]])
    incomplete_assignments = {
        "T_0": ["uav_0"],
        "T_1": [],  # Uncovered task
    }

    triggered = realloc.should_trigger(
        mode_changed=True,
        coalitions=mock_coalitions,
        fleet=mock_fleet,
        distance_matrix=dist_mat,
        cqi_matrix=cqi_mat,
        subtasks=mock_subtasks,
        assignments=incomplete_assignments,
    )
    assert triggered
    assert realloc.reallocation_trigger_count == 1
    assert realloc.reallocation_reasons.get("remaining_task_uncovered") == 1
    assert realloc.last_decision_reason == "remaining_task_uncovered"


def test_trigger_reallocation_on_capability_violation(mock_fleet, mock_subtasks, mock_coalitions):
    """TEST 7: Trigger reallocation if an assigned agent lacks required skills for the task."""
    realloc = PostSwitchReallocator()
    dist_mat = np.array([[0.0, 7.0], [7.0, 0.0]])
    cqi_mat = np.array([[1.0, 0.9], [0.9, 1.0]])
    # robot_0 has ['lift', 'carry'], but T_0 requires ['navigate']
    mismatched_assignments = {
        "T_0": ["robot_0"],
        "T_1": ["robot_0"],
    }

    triggered = realloc.should_trigger(
        mode_changed=True,
        coalitions=mock_coalitions,
        fleet=mock_fleet,
        distance_matrix=dist_mat,
        cqi_matrix=cqi_mat,
        subtasks=mock_subtasks,
        assignments=mismatched_assignments,
    )
    assert triggered
    assert realloc.reallocation_trigger_count == 1
    assert realloc.reallocation_reasons.get("capability_violation") == 1
    assert realloc.last_decision_reason == "capability_violation"


def test_trigger_reallocation_on_distance_infeasibility(mock_fleet, mock_coalitions, mock_assignments):
    """TEST 8: Trigger reallocation if target distance exceeds R_reach."""
    realloc = PostSwitchReallocator()
    distant_subtasks = [
        Subtask("T_0", "task 0", Position(999.0, 999.0), ["navigate"], completed=False),
        Subtask("T_1", "task 1", Position(16.0, 16.0), ["lift"], completed=False),
    ]
    dist_mat = np.array([[0.0, 7.0], [7.0, 0.0]])
    cqi_mat = np.array([[1.0, 0.9], [0.9, 1.0]])

    triggered = realloc.should_trigger(
        mode_changed=True,
        coalitions=mock_coalitions,
        fleet=mock_fleet,
        distance_matrix=dist_mat,
        cqi_matrix=cqi_mat,
        subtasks=distant_subtasks,
        assignments=mock_assignments,
        r_reach=100.0,
    )
    assert triggered
    assert realloc.reallocation_trigger_count == 1
    assert realloc.reallocation_reasons.get("assignment_invalid") == 1
    assert realloc.last_decision_reason == "assignment_invalid"
