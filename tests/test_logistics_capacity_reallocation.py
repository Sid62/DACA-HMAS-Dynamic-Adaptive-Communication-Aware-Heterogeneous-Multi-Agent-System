"""Tests for capacity-aware task allocation, reallocation, and plan reuse in Logistics."""

import numpy as np
import pytest

from src.coordination.decentralized_hybrid import DecentralizedHybridCoordinator
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.coordination.plan_continuity import PlanContinuityEngine
from src.decomposition.distance_feasible_decomp import (
    DistanceFeasibleDecomposer,
    domain_skill_affinity,
    validate_joint_assignment,
)
from src.env.agents import AgentFleet, AgentState, AgentType, Position, dist
from src.env.daca_env import DACAEnv
from src.env.scenarios import Scenario, Subtask, get_scenario


def _make_logistics_env(seed: int = 1) -> DACAEnv:
    cfg = {
        "scenarios": {
            "logistics": {
                "num_uav": 3,
                "num_vehicle": 4,
                "num_robot": 3,
                "num_subtasks": 6,
            }
        },
        "C1": 50.0,
        "C_task": 30.0,
        "R_reach": 100.0,
    }
    return DACAEnv(scenario_name="logistics", thresholds=cfg, seed=seed)


def test_closer_domain_mate_capacity_aware():
    """Verify _closer_domain_mate never returns an already occupied agent."""
    env = _make_logistics_env(seed=1)
    coord = DecentralizedHybridCoordinator(cloud_llm=None, device_llms={})

    # robot_7 is at (50, 50), robot_8 is at (20, 20), target is at (10, 10)
    # robot_8 is closer to target than robot_7.
    # But robot_8 is occupied by another task!
    target = Position(10.0, 10.0)
    domain = "robot"
    current_dist = dist(env.fleet.get_agent("robot_7").position, target)

    # Without occupancy restriction, robot_8 might be closer
    # With occupied_agents={"robot_8"}, robot_8 must NOT be returned
    result = coord._closer_domain_mate(
        agent_id="robot_7",
        target=target,
        fleet=env.fleet,
        domain=domain,
        current_dist=current_dist,
        occupied_agents={"robot_8"},
        r_reach=100.0,
    )
    assert result != "robot_8"


def test_local_reassign_preserves_valid_and_never_double_assigns():
    """Verify _local_reassign preserves valid feasible assignments and never double-books an agent."""
    env = _make_logistics_env(seed=1)
    coord = DecentralizedHybridCoordinator(cloud_llm=None, device_llms={})

    # Initial valid 1-to-1 assignments
    assignments = {
        "T_0": ["vehicle_4"],
        "T_1": ["robot_8"],
        "T_2": ["uav_0"],
        "T_3": ["vehicle_6"],
        "T_4": ["robot_7"],
        "T_5": ["uav_1"],
    }
    coalitions = [
        {"coalition_id": 0, "members": ["vehicle_4", "vehicle_6"]},
        {"coalition_id": 1, "members": ["robot_8", "robot_7"]},
        {"coalition_id": 2, "members": ["uav_0", "uav_1"]},
    ]
    cqi_matrix = np.ones((10, 10))

    coord._local_reassign(env, assignments, coalitions, cqi_matrix)

    # Check capacity constraint: all assigned agents across all active tasks must be distinct
    assigned_agents = [aid for aids in assignments.values() for aid in aids]
    assert len(assigned_agents) == len(set(assigned_agents)), "Duplicate agent assignment detected!"

    # Specifically: T1 and T4 must not share the same robot
    assert assignments["T_1"] != assignments["T_4"]
    assert set(assignments["T_1"]).isdisjoint(set(assignments["T_4"]))


def test_domain_skill_affinity_prioritizes_correct_domains():
    """Verify domain_skill_affinity prefers specialized domains for heterogeneous skills."""
    uav = AgentState("uav_0", AgentType.UAV, Position(0, 0), skills=["navigate", "sense", "inspect"])
    vehicle = AgentState("vehicle_3", AgentType.VEHICLE, Position(0, 0), skills=["navigate", "transport", "rescue"])
    robot = AgentState("robot_7", AgentType.ROBOT, Position(0, 0), skills=["lift", "rescue", "transport"])

    # lift tasks must prefer robots
    assert domain_skill_affinity(robot, {"lift"}) == 0
    assert domain_skill_affinity(uav, {"lift"}) == 2
    assert domain_skill_affinity(vehicle, {"lift"}) == 2

    # navigate + sense prefers UAVs
    assert domain_skill_affinity(uav, {"navigate", "sense"}) == 0
    assert domain_skill_affinity(vehicle, {"navigate", "sense"}) == 1

    # navigate + transport prefers vehicles
    assert domain_skill_affinity(vehicle, {"navigate", "transport"}) == 0
    assert domain_skill_affinity(uav, {"navigate", "transport"}) == 1

    # lift + transport prefers robots
    assert domain_skill_affinity(robot, {"lift", "transport"}) == 0


def test_plan_continuity_safe_locking_and_duplicate_cleanup():
    """Verify target commitment locking and clean_duplicate_assignments respect capacity."""
    env = _make_logistics_env(seed=1)
    engine = PlanContinuityEngine()

    # Intentionally malformed assignments where robot_8 is duplicated across T_1 and T_4
    bad_assignments = {
        "T_1": ["robot_8"],
        "T_4": ["robot_8"],
    }

    # clean_duplicate_assignments should resolve the collision and repair T_4 with a free robot
    cleaned = engine.clean_duplicate_assignments(bad_assignments, env.subtask_list, env.fleet)

    assigned = [aid for aids in cleaned.values() for aid in aids]
    assert len(assigned) == len(set(assigned)), "Cleaned assignments still contain duplicates!"
    assert len(cleaned["T_1"]) == 1
    assert len(cleaned["T_4"]) == 1
    assert cleaned["T_1"] != cleaned["T_4"]
    # Both T_1 and T_4 must have valid robots
    assert cleaned["T_1"][0].startswith("robot_")
    assert cleaned["T_4"][0].startswith("robot_")


def test_mission_progress_metric_percentage():
    """Verify success_rate calculation and percentage formatting."""
    env = _make_logistics_env(seed=1)

    assert env.success_rate() == 0.0

    # Mark 4 out of 6 complete
    env.mark_subtask_complete("T_0")
    env.mark_subtask_complete("T_2")
    env.mark_subtask_complete("T_3")
    env.mark_subtask_complete("T_5")

    # Success rate fraction
    assert env.success_rate() == pytest.approx(4.0 / 6.0)

    # Percentage string should be approximately 66.67%
    formatted = f"{env.success_rate() * 100.0:.2f}%"
    assert formatted == "66.67%"


def test_logistics_end_to_end_full_accuracy():
    """Run full Logistics simulation and verify 100% completion without robot double-booking stall."""
    for seed in [0, 1]:
        orch = DACAOrchestrator(
            scenario="logistics",
            network_profile="stable",
            seed=seed,
            config=CONFIGS["A5"],
            max_steps=250,
        )
        orch.cloud_llm.config["use_mock"] = True
        for dc in orch.device_llms.values():
            dc.config["use_mock"] = True

        metrics = orch.run()

        # All 6 tasks should successfully complete without stalling!
        assert metrics.success_rate == 1.0, f"Seed {seed}: Expected 100% completion, got {metrics.success_rate * 100.0:.2f}%"
        assert metrics.steps <= 250
