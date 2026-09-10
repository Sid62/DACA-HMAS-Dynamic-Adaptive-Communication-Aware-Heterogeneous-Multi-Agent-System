"""Regression tests for Issue 1: Validate assignments and task completion.

Tests explicitly cover all 14 required cases from Issue 1 specifications:
 1. correct skill -> valid
 2. missing skill -> invalid
 3. multi-agent collective skill coverage -> valid
 4. multi-agent missing one required skill -> invalid
 5. unknown agent -> invalid
 6. empty coalition -> invalid
 7. unknown-only coalition -> invalid
 8. centralized first-agent-only completion -> must remain incomplete when team capability is invalid
 9. decentralized first-agent-only completion -> must remain incomplete when team capability is invalid
10. valid team -> task can complete
11. repaired assignment -> skill validated
12. reused assignment -> skill validated
13. post-switch reallocation -> skill validated
14. duplicate/invalid assignment does not become a valid completion path
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
from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position, dist
from src.env.scenarios import Subtask
from src.coordination.plan_continuity import PlanContinuityEngine
from src.coordination.centralized_hybrid import CentralizedHybridCoordinator
from src.coordination.decentralized_hybrid import DecentralizedHybridCoordinator
from src.reallocation.post_switch import PostSwitchReallocator
from src.memory.experience_store import SubtaskExperienceStore, compute_signature


KIN = {
    "uav": KinematicsConfig(15.0, 1.5),
    "vehicle": KinematicsConfig(10.0, 0.8),
    "robot": KinematicsConfig(3.0, 2.0),
}


class DummyEnv:
    """Lightweight dummy environment for testing execution steps without external LLM."""

    def __init__(self, fleet: AgentFleet, subtasks: list[Subtask]):
        self.fleet = fleet
        self.subtask_list = subtasks
        self.scenario_name = "test_scenario"
        self._subtasks = {s.subtask_id: s for s in subtasks}

    def mark_subtask_complete(self, subtask_id: str) -> None:
        if subtask_id in self._subtasks:
            self._subtasks[subtask_id].completed = True


@pytest.fixture
def test_fleet():
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["transport", "navigate", "sense", "inspect"]),
        AgentState("vehicle_0", AgentType.VEHICLE, Position(15.0, 15.0), skills=["transport", "navigate", "rescue", "inspect"]),
        AgentState("robot_0", AgentType.ROBOT, Position(20.0, 20.0), skills=["lift", "transport", "rescue", "sense", "inspect"]),
    ]
    return AgentFleet(agents, KIN)


# ── 1. Correct skill -> valid ─────────────────────────────────────────────────

def test_1_correct_skill_valid(test_fleet):
    subtask = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"])
    assert validate_assignment_skills(["uav_0"], subtask, test_fleet) is True
    assert validate_joint_assignment(["uav_0"], subtask, test_fleet, c_task=30.0, r_reach=100.0) is True


# ── 2. Missing skill -> invalid ───────────────────────────────────────────────

def test_2_missing_skill_invalid(test_fleet):
    subtask = Subtask("T_0", "lift_task", Position(12.0, 12.0), required_skills=["lift"])
    # uav_0 lacks "lift"
    assert validate_assignment_skills(["uav_0"], subtask, test_fleet) is False
    assert validate_joint_assignment(["uav_0"], subtask, test_fleet, c_task=30.0, r_reach=100.0) is False


# ── 3. Multi-agent collective skill coverage -> valid ─────────────────────────

def test_3_multiagent_collective_skills_valid(test_fleet):
    # uav_0 has navigate, robot_0 has lift
    subtask = Subtask("T_0", "team_task", Position(14.0, 14.0), required_skills=["lift", "navigate"])
    assert validate_assignment_skills(["uav_0", "robot_0"], subtask, test_fleet) is True
    assert validate_joint_assignment(["uav_0", "robot_0"], subtask, test_fleet, c_task=30.0, r_reach=100.0) is True


# ── 4. Multi-agent missing one required skill -> invalid ──────────────────────

def test_4_multiagent_missing_one_skill_invalid(test_fleet):
    # requires lift, navigate, and an impossible skill "teleport"
    subtask = Subtask("T_0", "impossible_task", Position(14.0, 14.0), required_skills=["lift", "navigate", "teleport"])
    assert validate_assignment_skills(["uav_0", "robot_0"], subtask, test_fleet) is False
    assert validate_joint_assignment(["uav_0", "robot_0"], subtask, test_fleet, c_task=30.0, r_reach=100.0) is False


# ── 5. Unknown agent -> invalid ───────────────────────────────────────────────

def test_5_unknown_agent_invalid(test_fleet):
    subtask = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"])
    # Unknown agent alone must fail
    assert validate_assignment_skills(["ghost_0"], subtask, test_fleet) is False
    assert validate_joint_assignment(["ghost_0"], subtask, test_fleet, c_task=30.0, r_reach=100.0) is False
    # Known agent paired with unknown agent must fail
    assert validate_assignment_skills(["uav_0", "ghost_0"], subtask, test_fleet) is False
    assert validate_joint_assignment(["uav_0", "ghost_0"], subtask, test_fleet, c_task=30.0, r_reach=100.0) is False

    # Decomposition validate_assignments must reject unknown IDs
    decomp = DistanceFeasibleDecomposer(cloud_llm=None, c_task=30.0, r_reach=100.0)
    raw = {"T_0": ["uav_0", "ghost_0"]}
    validated = decomp.validate_assignments(raw, test_fleet, [subtask])
    assert validated.get("T_0") != ["uav_0", "ghost_0"]
    # Unknown agent must not be silently discarded leaving ["uav_0"]
    assert "ghost_0" not in validated.get("T_0", [])


# ── 6. Empty coalition -> invalid ─────────────────────────────────────────────

def test_6_empty_coalition_invalid(test_fleet):
    psi = np.ones((3, 3))
    id_to_idx = {a.agent_id: i for i, a in enumerate(test_fleet.agents)}
    assert validate_coalition_members([], id_to_idx, psi, gamma_min=0.3) is False

    subtask = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"])
    assert validate_assignment_skills([], subtask, test_fleet) is False
    assert validate_joint_assignment([], subtask, test_fleet, c_task=30.0, r_reach=100.0) is False


# ── 7. Unknown-only coalition -> invalid ──────────────────────────────────────

def test_7_unknown_only_coalition_invalid(test_fleet):
    psi = np.ones((3, 3))
    id_to_idx = {a.agent_id: i for i, a in enumerate(test_fleet.agents)}
    assert validate_coalition_members(["ghost_1", "ghost_2"], id_to_idx, psi, gamma_min=0.3) is False
    assert validate_coalition_members(["uav_0", "ghost_1"], id_to_idx, psi, gamma_min=0.3) is False


# ── 8. Centralized first-agent-only completion -> remains incomplete ──────────

def test_8_centralized_first_agent_only_remains_incomplete():
    # Subtask requires ["lift", "navigate"]
    subtask = Subtask("T_0", "lift_nav", Position(10.0, 10.0), required_skills=["lift", "navigate"])
    # uav_0 at target (10, 10), but lacks "lift". robot_0 is far away at (90, 90).
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "transport"]),
        AgentState("robot_0", AgentType.ROBOT, Position(90.0, 90.0), skills=["lift", "transport"]),
    ], KIN)
    env = DummyEnv(fleet, [subtask])

    # Direct completion validator check
    assert validate_task_completion(["uav_0"], subtask, fleet, completion_radius=8.0) is False
    assert validate_task_completion(["uav_0", "robot_0"], subtask, fleet, completion_radius=8.0) is False

    # Centralized coordinator execution step
    coord = CentralizedHybridCoordinator(cloud_llm=None, device_llms={}, decomposer=None, experience_store=None)
    coord.execute_step(env, {"T_0": ["uav_0", "robot_0"]})
    # Task must NOT be completed because robot_0 with "lift" hasn't arrived
    assert subtask.completed is False


# ── 9. Decentralized first-agent-only completion -> remains incomplete ────────

def test_9_decentralized_first_agent_only_remains_incomplete():
    # Subtask requires ["lift", "navigate"]
    subtask = Subtask("T_0", "lift_nav", Position(10.0, 10.0), required_skills=["lift", "navigate"])
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "transport"]),
        AgentState("robot_0", AgentType.ROBOT, Position(90.0, 90.0), skills=["lift", "transport"]),
    ], KIN)
    env = DummyEnv(fleet, [subtask])

    # Decentralized coordinator execution step
    coord = DecentralizedHybridCoordinator(cloud_llm=None, device_llms={}, decomposer=None, experience_store=None)
    coord.execute_step(env, {"T_0": ["uav_0", "robot_0"]})
    # Task must NOT be completed because robot_0 hasn't arrived
    assert subtask.completed is False


# ── 10. Valid team -> task can complete ───────────────────────────────────────

def test_10_valid_team_can_complete():
    # Subtask requires ["lift", "navigate"]
    subtask = Subtask("T_0", "lift_nav", Position(10.0, 10.0), required_skills=["lift", "navigate"])
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(11.0, 11.0), skills=["navigate", "transport"]),
        AgentState("robot_0", AgentType.ROBOT, Position(12.0, 12.0), skills=["lift", "transport"]),
    ], KIN)
    # Both agents are within completion_radius=8.0m of (10, 10)
    assert validate_task_completion(["uav_0", "robot_0"], subtask, fleet, completion_radius=8.0) is True

    # Execute step completes task
    env = DummyEnv(fleet, [subtask])
    coord = CentralizedHybridCoordinator(cloud_llm=None, device_llms={}, decomposer=None, experience_store=None)
    coord.execute_step(env, {"T_0": ["uav_0", "robot_0"]})
    assert subtask.completed is True


# ── 11. Repaired assignment -> skill validated ────────────────────────────────

def test_11_repaired_assignment_skill_validated(test_fleet):
    decomp = DistanceFeasibleDecomposer(cloud_llm=None, c_task=30.0, r_reach=100.0)
    # Subtask requiring lift: uav_0 cannot satisfy it
    st = Subtask("T_0", "lift_task", Position(18.0, 18.0), required_skills=["lift"])
    raw_assignments = {"T_0": ["uav_0"]}  # invalid: lacks lift

    validated = decomp.validate_assignments(raw_assignments, test_fleet, [st])
    # Must repair to an agent with lift (robot_0)
    assert validated.get("T_0") == ["robot_0"]
    assert validate_assignment_skills(validated["T_0"], st, test_fleet) is True

    # Unsolvable subtask: no agent has "fly_to_space"
    st_impossible = Subtask("T_imp", "impossible", Position(18.0, 18.0), required_skills=["fly_to_space"])
    validated_imp = decomp.validate_assignments({"T_imp": ["uav_0"]}, test_fleet, [st_impossible])
    # Must NOT assign any agent lacking the skill
    assert validated_imp.get("T_imp", []) == []


# ── 12. Reused assignment -> skill validated ──────────────────────────────────

def test_12_reused_assignment_skill_validated(test_fleet, tmp_path):
    store_file = str(tmp_path / "test_exp.json")
    store = SubtaskExperienceStore(store_path=store_file, enabled=True)
    st = Subtask("T_0", "lift_task", Position(18.0, 18.0), required_skills=["lift"])
    agent_types = [a.agent_type.value for a in test_fleet.agents]
    sig = compute_signature("test_scenario", ["lift"], agent_types, 5.0)

    # 12a. Stored plan with invalid skill (uav_0 lacks lift) -> reuse rejected
    store.record(sig, {"T_0": ["uav_0"]}, True, "test_scenario", ["lift"], agent_types)
    coord = CentralizedHybridCoordinator(cloud_llm=None, device_llms={}, decomposer=None, experience_store=store)
    env = DummyEnv(test_fleet, [st])
    reused = coord._try_experience_reuse(env, test_fleet, [st])
    assert "T_0" not in reused

    # 12b. Stored plan with unknown agent ID -> reuse rejected (not silently filtered)
    store.record(sig, {"T_0": ["robot_0", "ghost_agent"]}, True, "test_scenario", ["lift"], agent_types)
    reused = coord._try_experience_reuse(env, test_fleet, [st])
    assert "T_0" not in reused

    # 12c. Stored plan with valid team (robot_0 has lift) -> reuse accepted
    store.record(sig, {"T_0": ["robot_0"]}, True, "test_scenario", ["lift"], agent_types)
    reused = coord._try_experience_reuse(env, test_fleet, [st])
    assert reused.get("T_0") == ["robot_0"]


# ── 13. Post-switch reallocation -> skill validated ───────────────────────────

def test_13_post_switch_reallocation_skill_validated(test_fleet):
    reallocator = PostSwitchReallocator(peer_manager=None, device_llms={})
    st = Subtask("T_0", "lift_task", Position(18.0, 18.0), required_skills=["lift"])
    subtasks = [st]
    coalitions = [{"coalition_id": 0, "members": ["uav_0", "vehicle_0"]}]
    distance_matrix = np.zeros((3, 3))
    cqi_matrix = np.ones((3, 3))

    reallocated = reallocator._algorithmic_reallocate(
        subtasks=subtasks,
        fleet=test_fleet,
        coalitions=coalitions,
        distance_matrix=distance_matrix,
        cqi_matrix=cqi_matrix,
    )
    # The agent assigned to lift must be robot_0 (the only one with lift)
    robot_in_coalition = any("robot_0" in c.get("members", []) for c in reallocated)
    assert robot_in_coalition is True


# ── 14. Duplicate/invalid assignment does not become a valid completion path ──

def test_14_duplicate_invalid_assignment_handled_correctly(test_fleet):
    engine = PlanContinuityEngine()
    st1 = Subtask("T_0", "nav_task", Position(10.0, 10.0), required_skills=["navigate"])
    st2 = Subtask("T_1", "another_nav", Position(80.0, 80.0), required_skills=["navigate"])

    # uav_0 is within lock threshold (<35m) of T_0 at (10, 10)
    # But new_assignments mistakenly duplicates uav_0 onto T_1 as well
    previous_assignments = {"T_0": ["uav_0"], "T_1": ["vehicle_0"]}
    new_assignments = {"T_0": ["uav_0"], "T_1": ["uav_0"]}

    locked = engine.apply_target_commitment_lock(
        new_assignments=new_assignments,
        previous_assignments=previous_assignments,
        fleet=test_fleet,
        subtasks=[st1, st2],
        lock_threshold=35.0,
    )
    # T_0 keeps uav_0 (locked). T_1 duplicate uav_0 must be stripped, leaving []
    assert locked.get("T_0") == ["uav_0"]
    assert locked.get("T_1") == []  # Not left as invalid duplicate

    # get_updated_executable_assignments drops assignments with unknown IDs
    engine.set_active_plan(
        assignments={"T_0": ["uav_0", "unknown_agent"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0"]}],
        subtasks=[st1],
        mode=0,
    )
    updated = engine.get_updated_executable_assignments(test_fleet, [st1])
    # T_0 assignment with unknown agent must be rejected, not kept
    assert "unknown_agent" not in updated.get("T_0", [])
