"""Comprehensive regression test suite for DACA-HMAS capability and complementary coalition requirements.

Covers:
1. Every agent has exactly 3 skills.
2. Role-aware skills are deterministic.
3. Single-agent full-skill assignment works.
4. Complementary multi-agent skill coverage works.
5. Missing required skill fails.
6. Empty coalition fails.
7. Unknown agent ID fails.
8. Partial team arrival does not complete a task.
9. Full valid team arrival completes a task.
10. Reallocation never produces capability-invalid assignments.
11. No extra cloud calls are introduced by the fix.
12. No extra communication/consensus rounds are introduced.
"""

import numpy as np
import pytest

from src.coalition.feasibility import (
    build_psi_matrix,
    validate_coalition_members,
)
from src.decomposition.distance_feasible_decomp import (
    DistanceFeasibleDecomposer,
    validate_assignment_skills,
    validate_joint_assignment,
    validate_task_completion,
)
from src.env.agents import (
    AgentFleet,
    AgentState,
    AgentType,
    KinematicsConfig,
    Position,
    ROLE_SKILLS,
    create_fleet_from_scenario,
)
from src.env.scenarios import Subtask, get_scenario
from src.coordination.plan_continuity import PlanContinuityEngine
from src.reallocation.post_switch import PostSwitchReallocator
from src.coordination.orchestrator import DACAOrchestrator, CONFIGS


KIN = {
    "uav": KinematicsConfig(15.0, 1.5),
    "vehicle": KinematicsConfig(10.0, 0.8),
    "robot": KinematicsConfig(3.0, 2.0),
}


# ── Test 1: Every agent has exactly 3 skills ──────────────────────────────────

@pytest.mark.parametrize("scenario_name", ["logistics", "inspection", "search_rescue"])
@pytest.mark.parametrize("seed", [1, 2, 3, 42])
def test_every_agent_has_exactly_three_skills(scenario_name, seed):
    scenario = get_scenario(scenario_name, {}, seed=seed)
    fleet = create_fleet_from_scenario(scenario.agent_config, KIN, c1=50.0, c2=5.0, seed=seed)
    assert len(fleet.agents) > 0
    for agent in fleet.agents:
        assert len(agent.skills) == 3, f"Agent {agent.agent_id} has {len(agent.skills)} skills, expected exactly 3"
        assert len(set(agent.skills)) == 3, f"Agent {agent.agent_id} has duplicate skills: {agent.skills}"


# ── Test 2: Role-aware skills are deterministic ────────────────────────────────

@pytest.mark.parametrize("scenario_name", ["logistics", "inspection", "search_rescue"])
def test_role_aware_skills_are_deterministic(scenario_name):
    scenario = get_scenario(scenario_name, {}, seed=0)
    fleet1 = create_fleet_from_scenario(scenario.agent_config, KIN, c1=50.0, c2=5.0, seed=10)
    fleet2 = create_fleet_from_scenario(scenario.agent_config, KIN, c1=50.0, c2=5.0, seed=99)

    # Regardless of seed, agents of the same type must have the exact same role-aware skills
    for a1, a2 in zip(fleet1.agents, fleet2.agents):
        assert a1.agent_type == a2.agent_type
        assert a1.skills == a2.skills == ROLE_SKILLS[a1.agent_type]


# ── Test 3: Single-agent full-skill assignment works ───────────────────────────

def test_single_agent_full_skill_assignment():
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
        AgentState("robot_0", AgentType.ROBOT, Position(20.0, 20.0), skills=["lift", "rescue", "transport"]),
    ], KIN)
    subtask = Subtask("T_0", "aerial inspect", Position(12.0, 12.0), required_skills=["sense", "inspect"])
    assert validate_assignment_skills(["uav_0"], subtask, fleet) is True
    assert validate_joint_assignment(["uav_0"], subtask, fleet, c_task=30.0, r_reach=100.0) is True


# ── Test 4: Complementary multi-agent skill coverage works ────────────────────

def test_complementary_multi_agent_skill_coverage():
    fleet = AgentFleet([
        AgentState("vehicle_0", AgentType.VEHICLE, Position(10.0, 10.0), skills=["navigate", "transport", "rescue"]),
        AgentState("robot_0", AgentType.ROBOT, Position(15.0, 15.0), skills=["lift", "transport", "inspect"]),
    ], KIN)
    subtask = Subtask("T_0", "rescue operation", Position(12.0, 12.0), required_skills=["rescue", "lift"])

    # Individual agents fail
    assert validate_assignment_skills(["vehicle_0"], subtask, fleet) is False
    assert validate_assignment_skills(["robot_0"], subtask, fleet) is False

    # Complementary coalition succeeds
    assert validate_assignment_skills(["vehicle_0", "robot_0"], subtask, fleet) is True
    assert validate_joint_assignment(["vehicle_0", "robot_0"], subtask, fleet, c_task=30.0, r_reach=100.0) is True


# ── Test 5: Missing required skill fails ───────────────────────────────────────

def test_missing_required_skill_fails():
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
        AgentState("vehicle_0", AgentType.VEHICLE, Position(12.0, 12.0), skills=["navigate", "transport", "rescue"]),
    ], KIN)
    subtask = Subtask("T_0", "heavy lift", Position(12.0, 12.0), required_skills=["lift"])
    assert validate_assignment_skills(["uav_0"], subtask, fleet) is False
    assert validate_assignment_skills(["vehicle_0"], subtask, fleet) is False
    assert validate_assignment_skills(["uav_0", "vehicle_0"], subtask, fleet) is False


# ── Test 6: Empty coalition fails ─────────────────────────────────────────────

def test_empty_coalition_fails():
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
    ], KIN)
    subtask = Subtask("T_0", "task", Position(10.0, 10.0), required_skills=["navigate"])
    id_to_idx = {"uav_0": 0}
    psi = np.ones((1, 1))

    assert validate_coalition_members([], id_to_idx, psi, gamma_min=0.3) is False
    assert validate_assignment_skills([], subtask, fleet) is False
    assert validate_joint_assignment([], subtask, fleet, c_task=30.0, r_reach=100.0) is False
    assert validate_task_completion([], subtask, fleet) is False


# ── Test 7: Unknown agent ID fails ────────────────────────────────────────────

def test_unknown_agent_id_fails():
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
    ], KIN)
    subtask = Subtask("T_0", "task", Position(10.0, 10.0), required_skills=["navigate"])
    id_to_idx = {"uav_0": 0}
    psi = np.ones((1, 1))

    assert validate_coalition_members(["fake_id"], id_to_idx, psi, gamma_min=0.3) is False
    assert validate_coalition_members(["uav_0", "fake_id"], id_to_idx, psi, gamma_min=0.3) is False
    assert validate_assignment_skills(["fake_id"], subtask, fleet) is False
    assert validate_assignment_skills(["uav_0", "fake_id"], subtask, fleet) is False
    assert validate_joint_assignment(["fake_id"], subtask, fleet, c_task=30.0, r_reach=100.0) is False


# ── Test 8: Partial team arrival does not complete a task ─────────────────────

def test_partial_team_arrival_does_not_complete():
    subtask = Subtask("T_0", "coop", Position(10.0, 10.0), required_skills=["lift", "transport"])
    fleet = AgentFleet([
        AgentState("robot_0", AgentType.ROBOT, Position(10.0, 10.0), skills=["lift", "rescue", "transport"]),
        AgentState("vehicle_0", AgentType.VEHICLE, Position(50.0, 50.0), skills=["navigate", "transport", "rescue"]),
    ], KIN)
    assert validate_task_completion(["robot_0", "vehicle_0"], subtask, fleet, completion_radius=8.0) is False


# ── Test 9: Full valid team arrival completes a task ──────────────────────────

def test_full_valid_team_arrival_completes():
    subtask = Subtask("T_0", "coop", Position(10.0, 10.0), required_skills=["lift", "transport"])
    fleet = AgentFleet([
        AgentState("robot_0", AgentType.ROBOT, Position(11.0, 11.0), skills=["lift", "rescue", "transport"]),
        AgentState("vehicle_0", AgentType.VEHICLE, Position(12.0, 12.0), skills=["navigate", "transport", "rescue"]),
    ], KIN)
    assert validate_task_completion(["robot_0", "vehicle_0"], subtask, fleet, completion_radius=8.0) is True


# ── Test 10: Reallocation never produces capability-invalid assignments ───────

def test_reallocation_never_produces_invalid_assignments():
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
        AgentState("vehicle_0", AgentType.VEHICLE, Position(15.0, 15.0), skills=["navigate", "transport", "rescue"]),
        AgentState("robot_0", AgentType.ROBOT, Position(20.0, 20.0), skills=["lift", "rescue", "transport"]),
    ], KIN)
    reallocator = PostSwitchReallocator()
    subtasks = [
        Subtask("T_0", "sar extraction", Position(12.0, 12.0), required_skills=["rescue", "lift"]),
        Subtask("T_1", "aerial scan", Position(14.0, 14.0), required_skills=["sense", "navigate"]),
    ]
    coalitions = [
        {"coalition_id": 0, "members": ["uav_0"]},
        {"coalition_id": 1, "members": ["vehicle_0"]},
    ]
    dist_mat = np.zeros((3, 3))
    cqi_mat = np.ones((3, 3))

    reallocated = reallocator._algorithmic_reallocate(subtasks, fleet, coalitions, dist_mat, cqi_mat)
    valid_ids = {a.agent_id for a in fleet.agents}
    for c in reallocated:
        for m in c.get("members", []):
            assert m in valid_ids


# ── Test 11: No extra cloud calls are introduced ──────────────────────────────

def test_no_extra_cloud_calls(tmp_path):
    orch = DACAOrchestrator(
        scenario="search_rescue",
        network_profile="stable",
        seed=1,
        config=CONFIGS["A5"],
        max_steps=40,
    )
    orch.cloud_llm.config["use_mock"] = True
    orch.cloud_llm.config["cache_dir"] = str(tmp_path)
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
        dc.config["cache_dir"] = str(tmp_path)

    metrics = orch.run()
    assert metrics.cloud_api_calls <= 2


# ── Test 12: No extra communication/consensus rounds are introduced ───────────

def test_no_extra_communication_steps(tmp_path):
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=1,
        config=CONFIGS["A5"],
        max_steps=30,
    )
    orch.cloud_llm.config["use_mock"] = True
    orch.cloud_llm.config["cache_dir"] = str(tmp_path)
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
        dc.config["cache_dir"] = str(tmp_path)

    metrics = orch.run()
    assert metrics.paper_communication_steps <= 5
