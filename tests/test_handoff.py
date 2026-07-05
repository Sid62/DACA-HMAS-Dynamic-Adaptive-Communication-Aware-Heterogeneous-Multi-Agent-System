"""Tests for state snapshot handoff (Eqs 28-29)."""

import pytest

from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position
from src.env.scenarios import Subtask
from src.handoff.snapshot import (
    capture_snapshot,
    restore_snapshot,
    verify_task_preservation,
)


@pytest.fixture
def fleet():
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10, 20), skills=["navigate"]),
        AgentState("robot_0", AgentType.ROBOT, Position(30, 40), skills=["lift"]),
    ]
    kin = {
        "uav": KinematicsConfig(15, 1.5),
        "vehicle": KinematicsConfig(10, 0.8),
        "robot": KinematicsConfig(3, 2.0),
    }
    return AgentFleet(agents, kin)


@pytest.fixture
def subtasks():
    return [
        Subtask("T_0", "task 0", Position(50, 50), ["navigate"]),
        Subtask("T_1", "task 1", Position(60, 60), ["lift"], completed=True),
    ]


def test_capture_and_restore(fleet, subtasks):
    snap = capture_snapshot(fleet, subtasks, [{"coalition_id": 0, "members": ["uav_0"]}], 10, 0, 1)
    fleet.agents[0].position = Position(99, 99)
    restore_snapshot(fleet, snap)
    assert fleet.agents[0].position.x == pytest.approx(10)


def test_task_preservation(fleet, subtasks):
    before = capture_snapshot(fleet, subtasks, [], 5, 0, 1)
    after = capture_snapshot(fleet, subtasks, [], 5, 0, 1)
    assert verify_task_preservation(before, after)


def test_task_preservation_fails_on_loss(fleet, subtasks):
    before = capture_snapshot(fleet, subtasks, [], 5, 0, 1)
    subtasks.pop()
    after = capture_snapshot(fleet, subtasks, [], 5, 0, 1)
    assert not verify_task_preservation(before, after)
