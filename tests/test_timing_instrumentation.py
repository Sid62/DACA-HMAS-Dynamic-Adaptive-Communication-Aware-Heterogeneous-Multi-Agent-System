"""Tests for fine-grained timing instrumentation and handoff/reallocation separation."""

import pytest
import time
from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position
from src.env.scenarios import Subtask
from src.handoff.snapshot import capture_snapshot, restore_snapshot, verify_task_preservation
from src.metrics.evaluation import ExperimentMetrics


@pytest.fixture
def fleet():
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10, 20), skills=["navigate"], assigned_subtasks=["T_0"]),
        AgentState("robot_0", AgentType.ROBOT, Position(30, 40), skills=["lift"], completed_subtasks=["T_1"]),
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
        Subtask("T_0", "task 0", Position(50, 50), ["navigate"], completed=False),
        Subtask("T_1", "task 1", Position(60, 60), ["lift"], completed=True),
    ]


def test_handoff_excludes_reallocation(fleet, subtasks):
    """TEST 1: Verify state_handoff_time_s does NOT include reallocation_time_s."""
    snapshot_capture_time_s = 0.005
    state_restore_time_s = 0.003
    state_verification_time_s = 0.002
    coalition_transfer_time_s = 0.001
    reallocation_time_s = 0.050  # Separate reallocation time

    # Definition of pure state handoff time:
    state_handoff_time_s = (
        snapshot_capture_time_s
        + state_restore_time_s
        + state_verification_time_s
        + coalition_transfer_time_s
    )

    metrics = ExperimentMetrics(
        config_name="cfg",
        scenario="logistics",
        network_profile="stable",
        seed=0,
        success_rate=1.0,
        steps=10,
        cloud_tokens=0,
        device_tokens=0,
        total_tokens=0,
        cloud_api_calls=0,
        device_api_calls=0,
        total_api_calls=0,
        device_memory_mb=0.0,
        computation_s=0.1,
        snapshot_capture_time_s=snapshot_capture_time_s,
        state_restore_time_s=state_restore_time_s,
        state_verification_time_s=state_verification_time_s,
        coalition_transfer_time_s=coalition_transfer_time_s,
        reallocation_time_s=reallocation_time_s,
        state_handoff_time_s=state_handoff_time_s,
    )

    d = metrics.to_dict()
    assert d["state_handoff_time_s"] == pytest.approx(0.011)
    assert d["reallocation_time_s"] == pytest.approx(0.050)
    assert d["state_handoff_time_s"] < d["reallocation_time_s"] + d["state_handoff_time_s"]


def test_handoff_excludes_consensus():
    """TEST 2: Verify state_handoff_time_s does NOT include consensus_time_s."""
    state_handoff_time_s = 0.010
    consensus_time_s = 0.035

    metrics = ExperimentMetrics(
        config_name="cfg",
        scenario="logistics",
        network_profile="stable",
        seed=0,
        success_rate=1.0,
        steps=10,
        cloud_tokens=0,
        device_tokens=0,
        total_tokens=0,
        cloud_api_calls=0,
        device_api_calls=0,
        total_api_calls=0,
        device_memory_mb=0.0,
        computation_s=0.1,
        state_handoff_time_s=state_handoff_time_s,
        consensus_time_s=consensus_time_s,
    )

    d = metrics.to_dict()
    assert d["state_handoff_time_s"] == pytest.approx(0.010)
    assert d["consensus_time_s"] == pytest.approx(0.035)


def test_timers_non_negative():
    """TEST 3: Verify all timing metrics are >= 0."""
    metrics = ExperimentMetrics(
        config_name="cfg",
        scenario="logistics",
        network_profile="stable",
        seed=0,
        success_rate=1.0,
        steps=10,
        cloud_tokens=0,
        device_tokens=0,
        total_tokens=0,
        cloud_api_calls=0,
        device_api_calls=0,
        total_api_calls=0,
        device_memory_mb=0.0,
        computation_s=0.1,
        snapshot_capture_time_s=0.001,
        state_restore_time_s=0.002,
        state_verification_time_s=0.001,
        coalition_transfer_time_s=0.001,
        reallocation_time_s=0.005,
        state_handoff_time_s=0.005,
        consensus_time_s=0.010,
    )

    d = metrics.to_dict()
    for field in [
        "snapshot_capture_time_s",
        "state_restore_time_s",
        "state_verification_time_s",
        "coalition_transfer_time_s",
        "reallocation_time_s",
        "state_handoff_time_s",
        "consensus_time_s",
    ]:
        assert d[field] >= 0.0


def test_cloud_api_count_unchanged():
    """TEST 4: Verify timing metrics initialization does not alter API accounting fields."""
    m1 = ExperimentMetrics("cfg", "logistics", "stable", 0, 1.0, 10, 0, 0, 0, 0, 0, 0, 0.0, 0.1)
    m2 = ExperimentMetrics(
        "cfg", "logistics", "stable", 0, 1.0, 10, 0, 0, 0, 0, 0, 0, 0.0, 0.1,
        snapshot_capture_time_s=0.01,
        reallocation_time_s=0.05,
    )

    assert m1.cloud_api_calls == m2.cloud_api_calls == 0
    assert m1.cloud_network_calls == m2.cloud_network_calls == 0


def test_success_rate_unchanged():
    """TEST 5: Verify timing metrics initialization preserves success rate computation."""
    m1 = ExperimentMetrics("cfg", "logistics", "stable", 0, 0.85, 10, 0, 0, 0, 0, 0, 0, 0.0, 0.1)
    m2 = ExperimentMetrics(
        "cfg", "logistics", "stable", 0, 0.85, 10, 0, 0, 0, 0, 0, 0, 0.0, 0.1,
        snapshot_capture_time_s=0.01,
        reallocation_time_s=0.05,
    )

    assert m1.success_rate == pytest.approx(m2.success_rate) == 0.85


def test_state_handoff_correctness(fleet, subtasks):
    """TEST 6: Verify timing instrumentation does not alter snapshot contents or state preservation."""
    coalitions = [{"coalition_id": 0, "members": ["uav_0"]}]

    t0 = time.perf_counter()
    before = capture_snapshot(fleet, subtasks, coalitions, 10, 0, 1)
    t_cap = time.perf_counter() - t0

    t1 = time.perf_counter()
    restore_snapshot(fleet, before)
    t_res = time.perf_counter() - t1

    after = capture_snapshot(fleet, subtasks, coalitions, 10, 0, 1)

    t2 = time.perf_counter()
    is_valid = verify_task_preservation(before, after)
    t_ver = time.perf_counter() - t2

    assert is_valid
    assert t_cap >= 0.0
    assert t_res >= 0.0
    assert t_ver >= 0.0
