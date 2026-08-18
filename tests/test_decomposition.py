"""Tests for distance-feasible decomposition."""

import pytest

from src.decomposition.distance_feasible_decomp import compute_tfr, validate_joint_assignment
from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position
from src.env.scenarios import Subtask


@pytest.fixture
def fleet_and_subtask():
    agents = [
        AgentState("a0", AgentType.UAV, Position(10, 10)),
        AgentState("a1", AgentType.ROBOT, Position(15, 15)),
    ]
    kin = {"uav": KinematicsConfig(15, 1.5), "vehicle": KinematicsConfig(10, 0.8),
           "robot": KinematicsConfig(3, 2.0)}
    fleet = AgentFleet(agents, kin)
    subtask = Subtask("T_0", "test", Position(20, 20), ["navigate"])
    return fleet, subtask


def test_feasible_pair(fleet_and_subtask):
    fleet, subtask = fleet_and_subtask
    assert validate_joint_assignment(["a0", "a1"], subtask, fleet, c_task=30, r_reach=100)


def test_infeasible_distance(fleet_and_subtask):
    fleet, subtask = fleet_and_subtask
    fleet.agents[1].position = Position(200, 200)
    assert not validate_joint_assignment(["a0", "a1"], subtask, fleet, c_task=30, r_reach=100)


def test_tfr_metric(fleet_and_subtask):
    fleet, subtask = fleet_and_subtask
    assignments = {"T_0": ["a0", "a1"]}
    tfr = compute_tfr(assignments, [subtask], fleet, c_task=30, r_reach=100)
    assert tfr == 1.0
