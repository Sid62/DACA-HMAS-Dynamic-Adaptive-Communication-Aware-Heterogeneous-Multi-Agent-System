"""Regression tests for Issue 2: Repair plan reuse and assignment updates.

Tests explicitly cover:
 1. Invalid skills fail despite high weighted score (hard feasibility gate).
 2. Unknown agent IDs fail evaluate_plan_validity().
 3. Invalid distance/joint assignment fails evaluate_plan_validity().
 4. Valid complementary team passes hard feasibility and reaches weighted scoring.
 5. Invalid repaired assignment is rejected (not accepted).
 6. Valid repaired assignment with collective skills is accepted.
 7. Repaired/reused assignments are propagated to execution and dispatch state.
 8. Stale assignments are removed after reassignment or task completion.
 9. Completed task IDs remain excluded from future plan reuse.
10. Duplicate assignments are cleaned even when another task is unassigned [].
11. A task with no feasible team remains pending/unassigned ([]) rather than receiving an arbitrary agent.
12. Agent assigned to multiple incomplete tasks fails plan validity.
"""

import numpy as np
import pytest

from src.coordination.plan_continuity import PlanContinuityEngine, ActivePlanContext
from src.coordination.centralized_hybrid import CentralizedHybridCoordinator
from src.coordination.decentralized_hybrid import DecentralizedHybridCoordinator
from src.coordination.orchestrator import DACAOrchestrator, CONFIGS
from src.decomposition.distance_feasible_decomp import validate_assignment_skills, validate_joint_assignment
from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position
from src.env.scenarios import Subtask


KIN = {
    "uav": KinematicsConfig(15.0, 1.5),
    "vehicle": KinematicsConfig(10.0, 0.8),
    "robot": KinematicsConfig(4.00, 2.0),
}


class DummyEnv:
    """Lightweight mock environment for testing execution without full simulation overhead."""

    def __init__(self, fleet: AgentFleet, subtasks: list[Subtask]):
        self.fleet = fleet
        self.subtask_list = subtasks
        self.scenario_name = "test_scenario"
        self._subtasks = {s.subtask_id: s for s in subtasks}

    def mark_subtask_complete(self, subtask_id: str) -> None:
        if subtask_id in self._subtasks:
            self._subtasks[subtask_id].completed = True


@pytest.fixture
def fleet():
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
        AgentState("vehicle_0", AgentType.VEHICLE, Position(15.0, 15.0), skills=["navigate", "transport", "rescue"]),
        AgentState("robot_0", AgentType.ROBOT, Position(20.0, 20.0), skills=["lift", "rescue", "transport"]),
    ]
    return AgentFleet(agents, KIN)


# ── 1. Invalid skills fail despite high weighted score ────────────────────────

def test_invalid_skills_fail_despite_high_score(fleet):
    engine = PlanContinuityEngine(validity_threshold=0.75, r_reach=100.0, c_task=30.0)
    # T_0 requires lift, but uav_0 has navigate/sense/inspect (missing lift)
    st = Subtask("T_0", "lift_task", Position(12.0, 12.0), required_skills=["lift"])
    
    # Set active plan with uav_0 assigned
    engine.set_active_plan(
        assignments={"T_0": ["uav_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0"]}],
        subtasks=[st],
        mode=0,
    )

    # Perfect communication (CQI=1.0), zero packet loss, zero latency, close distance (<3m)
    cqi = np.ones((3, 3))
    score = engine.evaluate_plan_validity(fleet, [st], cqi_matrix=cqi, sys_cqi=1.0, packet_loss=0.0, latency=0.0)

    # Hard feasibility gate MUST reject this plan despite perfect network & distance scores
    assert score.total_validity_score == 0.0
    assert score.is_valid is False
    assert engine.can_continue_plan(fleet, [st], cqi_matrix=cqi) is False


# ── 2. Unknown agent IDs fail evaluate_plan_validity ──────────────────────────

def test_unknown_agent_fails_plan_validity(fleet):
    engine = PlanContinuityEngine()
    st = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"])

    engine.set_active_plan(
        assignments={"T_0": ["ghost_agent_999"]},
        coalitions=[{"coalition_id": 0, "members": ["ghost_agent_999"]}],
        subtasks=[st],
        mode=0,
    )
    score = engine.evaluate_plan_validity(fleet, [st])
    assert score.total_validity_score == 0.0
    assert score.is_valid is False


# ── 3. Invalid distance/joint assignment fails evaluate_plan_validity ─────────

def test_invalid_distance_joint_assignment_fails(fleet):
    engine = PlanContinuityEngine(r_reach=50.0)
    # Target is at (200, 200), but uav_0 is at (10, 10) -> distance > 260m >> r_reach
    st = Subtask("T_0", "far_task", Position(200.0, 200.0), required_skills=["navigate"])

    engine.set_active_plan(
        assignments={"T_0": ["uav_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0"]}],
        subtasks=[st],
        mode=0,
    )
    score = engine.evaluate_plan_validity(fleet, [st])
    assert score.total_validity_score == 0.0
    assert score.is_valid is False


# ── 4. Valid complementary team passes hard feasibility and reaches score ────

def test_valid_complementary_team_passes(fleet):
    engine = PlanContinuityEngine(validity_threshold=0.75, r_reach=100.0, c_task=30.0)
    # T_0 requires navigate and lift: uav_0 has navigate, robot_0 has lift
    st = Subtask("T_0", "coop_task", Position(15.0, 15.0), required_skills=["navigate", "lift"])

    engine.set_active_plan(
        assignments={"T_0": ["uav_0", "robot_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0", "robot_0"]}],
        subtasks=[st],
        mode=0,
    )
    cqi = np.ones((3, 3))
    score = engine.evaluate_plan_validity(fleet, [st], cqi_matrix=cqi, sys_cqi=1.0)
    assert score.is_valid is True
    assert score.total_validity_score >= 0.75


# ── 5. Invalid repaired assignment is rejected ────────────────────────────────

def test_invalid_repaired_assignment_rejected(fleet):
    engine = PlanContinuityEngine(r_reach=100.0)
    # Impossible task: requires "teleport" which no agent has
    st = Subtask("T_0", "impossible", Position(12.0, 12.0), required_skills=["teleport"])
    
    # Active plan had uav_0 assigned (invalid)
    engine.set_active_plan(
        assignments={"T_0": ["uav_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0"]}],
        subtasks=[st],
        mode=0,
    )
    updated = engine.get_updated_executable_assignments(fleet, [st])
    # Must reject and leave T_0 as []
    assert updated.get("T_0") == []


# ── 6. Valid repaired assignment with collective skills is accepted ───────────

def test_valid_repaired_assignment_accepted(fleet):
    engine = PlanContinuityEngine(r_reach=100.0)
    # T_0 requires lift. Active context erroneously assigned uav_0 (lacks lift)
    st = Subtask("T_0", "lift_task", Position(18.0, 18.0), required_skills=["lift"])

    engine.set_active_plan(
        assignments={"T_0": ["uav_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0"]}],
        subtasks=[st],
        mode=0,
    )
    # get_updated_executable_assignments should drop uav_0 and repair with freed robot_0 (which has lift)
    updated = engine.get_updated_executable_assignments(fleet, [st])
    assert updated.get("T_0") == ["robot_0"]
    assert validate_assignment_skills(updated["T_0"], st, fleet) is True


# ── 7. Repaired/reused assignments reach execution and dispatch ───────────────

def test_repaired_reused_assignment_reaches_execution_and_dispatch(fleet):
    engine = PlanContinuityEngine(r_reach=100.0)
    st = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"])
    engine.set_active_plan(
        assignments={"T_0": ["uav_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0"]}],
        subtasks=[st],
        mode=0,
    )
    coord = CentralizedHybridCoordinator(cloud_llm=None, device_llms={}, decomposer=None, continuity_engine=engine)
    env = DummyEnv(fleet, [st])
    
    # plan() reuses valid active plan
    assignments, coalitions, cloud_reasoned, dispatch_occurred = coord.plan(env)
    assert cloud_reasoned is False
    assert assignments == {"T_0": ["uav_0"]}
    assert coord._last_dispatched_assignments == {"T_0": ["uav_0"]}


# ── 8. Stale assignments removed after reassignment or completion ─────────────

def test_stale_assignments_removed_after_completion(fleet):
    engine = PlanContinuityEngine()
    st = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"])
    engine.set_active_plan(
        assignments={"T_0": ["uav_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0"]}],
        subtasks=[st],
        mode=0,
    )
    # Mark task completed
    engine.mark_subtask_completed("T_0")

    # Authoritative active assignments must no longer hold T_0
    assert "T_0" not in engine.active_context.assignments
    assert "T_0" in engine.active_context.completed_subtask_ids

    # get_updated_executable_assignments must not reintroduce T_0
    st.completed = True
    updated = engine.get_updated_executable_assignments(fleet, [st])
    assert "T_0" not in updated


# ── 9. Completed task IDs remain excluded from future plan reuse ───────────────

def test_completed_task_ids_excluded_from_reuse(fleet):
    engine = PlanContinuityEngine()
    st1 = Subtask("T_0", "done_task", Position(12.0, 12.0), required_skills=["navigate"], completed=True)
    st2 = Subtask("T_1", "active_task", Position(18.0, 18.0), required_skills=["lift"])

    engine.set_active_plan(
        assignments={"T_0": ["uav_0"], "T_1": ["robot_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0", "robot_0"]}],
        subtasks=[st1, st2],
        mode=0,
    )
    # Only incomplete task T_1 should be in updated assignments
    updated = engine.get_updated_executable_assignments(fleet, [st1, st2])
    assert "T_0" not in updated
    assert updated.get("T_1") == ["robot_0"]


# ── 10. Duplicate assignments cleaned even when another task is [] ────────────

def test_duplicate_cleanup_works_with_empty_tasks(fleet):
    st0 = Subtask("T_0", "empty_task", Position(50.0, 50.0), required_skills=["inspect"])
    st1 = Subtask("T_1", "near_task", Position(12.0, 12.0), required_skills=["navigate"])
    st2 = Subtask("T_2", "far_task", Position(90.0, 90.0), required_skills=["navigate"])

    # T_0 is unassigned ([]). T_1 and T_2 mistakenly duplicate uav_0 (which is at (10, 10))
    raw = {"T_0": [], "T_1": ["uav_0"], "T_2": ["uav_0"]}

    cleaned = PlanContinuityEngine.clean_duplicate_assignments(raw, [st0, st1, st2], fleet)
    # T_0 remains []
    assert cleaned["T_0"] == []
    # uav_0 is much closer to T_1 (dist=2.83m) than T_2 (dist=113m) -> keeps T_1
    assert cleaned["T_1"] == ["uav_0"]
    # T_2 duplicate must be removed, leaving []
    assert cleaned["T_2"] == []


# ── 11. No feasible team stays pending [] ─────────────────────────────────────

def test_no_feasible_team_stays_pending(fleet):
    engine = PlanContinuityEngine()
    st = Subtask("T_0", "unsolvable", Position(12.0, 12.0), required_skills=["deep_ocean_drill"])
    engine.set_active_plan(
        assignments={"T_0": []},
        coalitions=[],
        subtasks=[st],
        mode=0,
    )
    updated = engine.get_updated_executable_assignments(fleet, [st])
    # Must NOT fall back to arbitrary agent; stays []
    assert updated.get("T_0") == []


# ── 12. Agent assigned to multiple incomplete tasks fails plan validity ───────

def test_agent_assigned_to_multiple_tasks_fails_validity(fleet):
    engine = PlanContinuityEngine()
    st1 = Subtask("T_0", "nav1", Position(12.0, 12.0), required_skills=["navigate"])
    st2 = Subtask("T_1", "nav2", Position(14.0, 14.0), required_skills=["navigate"])

    # uav_0 assigned to two different incomplete tasks
    engine.set_active_plan(
        assignments={"T_0": ["uav_0"], "T_1": ["uav_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0"]}],
        subtasks=[st1, st2],
        mode=0,
    )
    score = engine.evaluate_plan_validity(fleet, [st1, st2])
    # Duplicate conflict must trigger hard gate failure
    assert score.total_validity_score == 0.0
    assert score.is_valid is False


# ── 13. Completion single task does not raise RuntimeError ───────────────────

def test_completion_single_task_does_not_raise_runtime_error(fleet):
    """Verify completing one task does not raise RuntimeError and removes assignment."""
    engine = PlanContinuityEngine()
    # T_0 matches uav_0 position (10.0, 10.0), so validate_task_completion passes immediately
    st0 = Subtask("T_0", "nav0", Position(10.0, 10.0), required_skills=["navigate"])
    # T_1 is far away, so it does not complete
    st1 = Subtask("T_1", "nav1", Position(100.0, 100.0), required_skills=["navigate"])

    engine.set_active_plan(
        assignments={"T_0": ["uav_0"], "T_1": ["vehicle_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0"]}],
        subtasks=[st0, st1],
        mode=0,
    )
    # Use exact same assignments dict reference as active context to test mutation safety
    assignments = engine.active_context.assignments

    env = DummyEnv(fleet, [st0, st1])
    coordinator = CentralizedHybridCoordinator(cloud_llm=None, device_llms={}, decomposer=None, continuity_engine=engine)

    # Executing step must not raise RuntimeError: dictionary changed size during iteration
    coordinator.execute_step(env, assignments)

    assert st0.completed is True
    assert st1.completed is False
    assert "T_0" in engine.active_context.completed_subtask_ids
    assert "T_0" not in engine.active_context.assignments
    assert "T_0" not in assignments
    assert "T_1" in assignments
    assert assignments["T_1"] == ["vehicle_0"]


# ── 14. Completion multiple tasks in one step does not raise ─────────────────

def test_completion_multiple_tasks_in_one_step_safe(fleet):
    """Verify multiple tasks completing in the same step do not crash and all are recorded."""
    engine = PlanContinuityEngine()
    # T_0 at uav_0 (10, 10), T_1 at vehicle_0 (15, 15) -> both complete in same step!
    st0 = Subtask("T_0", "nav0", Position(10.0, 10.0), required_skills=["navigate"])
    st1 = Subtask("T_1", "nav1", Position(15.0, 15.0), required_skills=["navigate"])
    st2 = Subtask("T_2", "nav2", Position(200.0, 200.0), required_skills=["lift"])

    engine.set_active_plan(
        assignments={"T_0": ["uav_0"], "T_1": ["vehicle_0"], "T_2": ["robot_0"]},
        coalitions=[{"coalition_id": 0, "members": ["uav_0", "vehicle_0", "robot_0"]}],
        subtasks=[st0, st1, st2],
        mode=1,
    )
    assignments = engine.active_context.assignments

    env = DummyEnv(fleet, [st0, st1, st2])
    coordinator = DecentralizedHybridCoordinator(cloud_llm=None, device_llms={}, decomposer=None, continuity_engine=engine)

    # Must process both completions without RuntimeError and process remaining assignments
    coordinator.execute_step(env, assignments)

    assert st0.completed is True
    assert st1.completed is True
    assert st2.completed is False
    assert "T_0" in engine.active_context.completed_subtask_ids
    assert "T_1" in engine.active_context.completed_subtask_ids
    assert "T_2" not in engine.active_context.completed_subtask_ids

    # Both completed tasks removed; uncompleted task preserved
    assert "T_0" not in assignments
    assert "T_1" not in assignments
    assert "T_2" in assignments
    assert assignments["T_2"] == ["robot_0"]


# ── 15. Both centralized and decentralized paths are safe ────────────────────

def test_centralized_and_decentralized_completion_paths_safe(fleet):
    """Verify both execution paths safely handle completion when assignments dictionary is shared."""
    for is_decentralized in [False, True]:
        engine = PlanContinuityEngine()
        st0 = Subtask("T_0", "nav0", Position(10.0, 10.0), required_skills=["navigate"])
        st1 = Subtask("T_1", "nav1", Position(80.0, 80.0), required_skills=["transport"])

        engine.set_active_plan(
            assignments={"T_0": ["uav_0"], "T_1": ["vehicle_0"]},
            coalitions=[{"coalition_id": 0, "members": ["uav_0", "vehicle_0"]}],
            subtasks=[st0, st1],
            mode=1 if is_decentralized else 0,
        )
        assignments = engine.active_context.assignments
        env = DummyEnv(fleet, [st0, st1])

        if is_decentralized:
            coord = DecentralizedHybridCoordinator(cloud_llm=None, device_llms={}, decomposer=None, continuity_engine=engine)
        else:
            coord = CentralizedHybridCoordinator(cloud_llm=None, device_llms={}, decomposer=None, continuity_engine=engine)

        coord.execute_step(env, assignments)

        assert st0.completed is True
        assert "T_0" not in assignments
        assert "T_1" in assignments


# ── 16. Orchestrator execution loop safety on task completion ────────────────

def test_orchestrator_execution_completion_loop_safety(fleet):
    """Verify orchestrator's completion loop safely iterates when tasks complete."""
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=42,
        config=CONFIGS["A5"],
        max_steps=5,
    )
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True

    # Artificially position agent at first incomplete task to trigger completion during run
    st = orch.env.subtask_list[0]
    lead_agent = orch.env.fleet.agents[0]
    lead_agent.position.x = st.target.x
    lead_agent.position.y = st.target.y

    # Running orchestrator must complete the task without dictionary iteration error
    metrics = orch.run()
    assert metrics is not None
    assert metrics.steps > 0


# ── 17. Orchestrator 100-step mission completion safe ─────────────────────────

def test_orchestrator_100_step_mission_completion_safe():
    """Verify orchestrator runs 100 steps with multiple task completions without crashing."""
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=42,
        config=CONFIGS["A5"],
        max_steps=100,
    )
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True

    metrics = orch.run()
    assert metrics is not None
    assert metrics.steps <= 100
    assert metrics.success_rate > 0.0

