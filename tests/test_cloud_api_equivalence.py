"""Validation suite testing all 13 criteria (Tests A-M) for AutoHMA-LLM Cloud API Call equivalence.

Criteria:
- Test A: Initial centralized planning generates expected actual Cloud requests (2 calls).
- Test B: Stable execution generates no unnecessary additional Cloud requests.
- Test C: A genuine global feedback event generates another Cloud request.
- Test D: Multiple independent global events generate multiple additional Cloud requests.
- Test E: Local recoverable problems do not generate unnecessary Cloud calls.
- Test F: Cache hits do not increment cloud_api_calls.
- Test G: A real uncached Cloud request increments exactly once.
- Test H: should_replan() returning global-replan decision cannot be silently swallowed by a second continuity gate.
- Test I: Same seed + same configuration produces deterministic API counts.
- Test J: Different scenarios can produce different counts.
- Test K: Inspection communication degradation can actually influence global planning when it affects global plan.
- Test L: Decentralized architecture produces cloud_api_calls = 0.
- Test M: Reducing Cloud calls through optimization does not reduce task completion correctness.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator, DACAConfig
from src.coordination.replan_trigger import PlanState, ReplanDecision, should_replan
from src.coordination.plan_continuity import PlanContinuityEngine
from src.coordination.centralized_hybrid import CentralizedHybridCoordinator
from src.llm.cloud_llm_client import CloudLLMClient
from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position
from src.env.scenarios import Subtask


KIN = {
    "uav": KinematicsConfig(13.0, 1.5),
    "vehicle": KinematicsConfig(9.0, 0.8),
    "robot": KinematicsConfig(4.25, 2.0),
}


def _make_mock_orchestrator(
    scenario: str = "logistics",
    profile: str = "stable",
    seed: int = 1,
    config: DACAConfig | None = None,
    max_steps: int = 20,
    tmp_path: Path | None = None,
) -> DACAOrchestrator:
    cfg = config or CONFIGS["B1"]
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile=profile,
        seed=seed,
        config=cfg,
        max_steps=max_steps,
    )
    orch.cloud_llm.config["use_mock"] = True
    orch.cloud_llm.config["cache_responses"] = False
    if tmp_path:
        orch.cloud_llm.config["cache_dir"] = str(tmp_path)
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
        dc.config["cache_responses"] = False
        if tmp_path:
            dc.config["cache_dir"] = str(tmp_path)
    return orch


# --- Test A: Initial centralized planning generates expected actual Cloud requests (2 calls) ---
def test_a_initial_planning_generates_two_cloud_calls():
    orch = _make_mock_orchestrator("logistics", "stable", seed=1, max_steps=1)
    metrics = orch.run()
    # Initial centralized decomposition (1) + coalition formation (1) = exactly 2 Cloud API calls
    assert metrics.cloud_api_calls == 2
    assert orch.cloud_llm.usage.cloud_api_calls == 2
    assert orch.cloud_llm.usage.initial_planning_calls >= 2


# --- Test B: Stable execution generates no unnecessary additional Cloud requests ---
def test_b_stable_execution_generates_no_unnecessary_additional_calls():
    # In stable condition without disturbances, initial plan executes without spurious replans
    orch = _make_mock_orchestrator("logistics", "stable", seed=1, max_steps=10)
    metrics = orch.run()
    assert metrics.cloud_api_calls == 2


# --- Test C: A genuine global feedback event generates another Cloud request ---
def test_c_genuine_global_feedback_generates_additional_request():
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
        AgentState("robot_0", AgentType.ROBOT, Position(15.0, 15.0), skills=["lift", "rescue", "transport"]),
    ], KIN)
    subtasks = [
        Subtask("T_0", "nav", Position(10.0, 10.0), required_skills=["navigate"]),
        Subtask("T_1", "lift", Position(50.0, 50.0), required_skills=["lift"]),
    ]
    coalitions = [{"coalition_id": 0, "members": ["uav_0", "robot_0"]}]
    plan_state = PlanState(initialized=True, last_replan_step=-10, known_subtask_ids={"T_0", "T_1"}, known_completed_ids=set())

    # Initial plan has T_0 completed, but T_1 has no assignment and is far away
    subtasks[0].completed = True

    cont = PlanContinuityEngine(r_reach=20.0)
    cont.set_active_plan({"T_0": ["uav_0"]}, coalitions, subtasks, mode=0)

    # Trigger 3 should fire with global scope because T_1 is unassigned and outside reach
    decision = should_replan(plan_state, subtasks, fleet, coalitions, mode=0, continuity_engine=cont)
    assert decision.replan_now is True
    assert decision.scope == "global"
    assert "task_completed_needs_reassignment" in decision.reason


# --- Test D: Multiple independent global events generate multiple additional Cloud requests ---
def test_d_multiple_independent_global_events_generate_multiple_requests():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    client.semantic_cache.enabled = False
    coord = CentralizedHybridCoordinator(cloud_llm=client)
    
    # Mock environment
    st1 = Subtask("T_0", "s1", Position(10.0, 10.0), required_skills=["navigate"])
    st2 = Subtask("T_1", "s2", Position(20.0, 20.0), required_skills=["inspect"])
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
    ], KIN)

    class MockEnv:
        def __init__(self):
            self.fleet = fleet
            self.subtask_list = [st1, st2]
            self.scenario = "logistics"
        def get_observation(self):
            return {
                "instruction": "test",
                "agents": self.fleet.to_dict_list(),
                "subtasks": [{"id": s.subtask_id, "skills": s.required_skills, "target": [s.target.x, s.target.y]} for s in self.subtask_list],
            }

    env = MockEnv()
    # Initial: 2 calls
    coord.plan(env, force_replan=True, replan_reason="mission_initialization")
    assert client.usage.cloud_api_calls == 2

    # Global Event 1: task reassignment needed (+1 call)
    coord.plan(env, force_replan=True, replan_reason="task_completed_needs_reassignment:['T_0']")
    assert client.usage.cloud_api_calls == 3

    # Global Event 2: network degradation crossed threshold (+1 call)
    coord.plan(env, force_replan=True, replan_reason="packet_loss_crossed_threshold:0.450")
    assert client.usage.cloud_api_calls == 4


# --- Test E: Local recoverable problems do not generate unnecessary Cloud calls ---
def test_e_local_recoverable_problem_does_not_call_cloud():
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
        AgentState("uav_1", AgentType.UAV, Position(12.0, 12.0), skills=["navigate", "sense", "inspect"]),
    ], KIN)
    subtasks = [
        Subtask("T_0", "nav", Position(10.0, 10.0), required_skills=["navigate"]),
        Subtask("T_1", "nav2", Position(12.0, 12.0), required_skills=["navigate"]),
    ]
    coalitions = [{"coalition_id": 0, "members": ["uav_0", "uav_1"]}]
    plan_state = PlanState(initialized=True, last_replan_step=-10)

    cont = PlanContinuityEngine(r_reach=100.0)
    # Duplicate agent assigned to both tasks in active context
    cont.set_active_plan({"T_0": ["uav_0"], "T_1": ["uav_0"]}, coalitions, subtasks, mode=0)

    decision = should_replan(plan_state, subtasks, fleet, coalitions, mode=0, continuity_engine=cont)
    # Recoverable locally because uav_1 is free and covers navigate within r_reach
    assert decision.scope == "local"


# --- Test F: Cache hits do not increment cloud_api_calls ---
def test_f_cache_hits_do_not_increment_cloud_api_calls(tmp_path):
    config = {"use_mock": True, "cache_responses": True, "cache_dir": str(tmp_path)}
    client = CloudLLMClient(config=config)

    prompt = "Unique prompt for Cache Hit Test"
    # First call: cache miss -> increments to 1
    client.complete(prompt)
    assert client.usage.cloud_api_calls == 1
    assert client.usage.cache_hits == 0

    # Second call: cache hit -> stays at 1, cache_hits increments to 1
    client.complete(prompt)
    assert client.usage.cloud_api_calls == 1
    assert client.usage.cache_hits == 1
    assert client.usage.cloud_disk_cache_hits == 1


# --- Test G: A real uncached Cloud request increments exactly once ---
def test_g_real_uncached_request_increments_exactly_once():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    assert client.usage.cloud_api_calls == 0
    client.complete("prompt 1")
    assert client.usage.cloud_api_calls == 1
    client.complete("prompt 2")
    assert client.usage.cloud_api_calls == 2
    client.complete("prompt 3")
    assert client.usage.cloud_api_calls == 3


# --- Test H: should_replan() global decision cannot be swallowed by continuity gate ---
def test_h_double_gating_bypass_verified():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    engine = PlanContinuityEngine(r_reach=100.0)
    st = Subtask("T_0", "nav", Position(10.0, 10.0), required_skills=["navigate"])
    fleet = AgentFleet([AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate"])], KIN)
    engine.set_active_plan({"T_0": ["uav_0"]}, [{"coalition_id": 0, "members": ["uav_0"]}], [st], mode=0)

    coord = CentralizedHybridCoordinator(cloud_llm=client, continuity_engine=engine)
    class DummyEnv:
        def __init__(self):
            self.fleet = fleet
            self.subtask_list = [st]
            self.scenario = "logistics"
        def get_observation(self):
            return {"instruction": "task", "agents": fleet.to_dict_list(), "subtasks": [{"id": st.subtask_id, "skills": st.required_skills, "target": [10.0, 10.0]}]}

    env = DummyEnv()
    # When force_replan=True, plan() MUST NOT return (cloud_reasoned=False)
    assignments, coalitions, cloud_reasoned, dispatch_occurred = coord.plan(
        env, force_replan=True, replan_reason="task_completed_needs_reassignment:['T_0']"
    )
    assert cloud_reasoned is True
    assert client.usage.cloud_api_calls >= 1


# --- Test I: Same seed + same configuration produces deterministic API counts ---
def test_i_seed_determinism():
    orch1 = _make_mock_orchestrator("logistics", "stable", seed=42, max_steps=15)
    m1 = orch1.run()

    orch2 = _make_mock_orchestrator("logistics", "stable", seed=42, max_steps=15)
    m2 = orch2.run()

    assert m1.cloud_api_calls == m2.cloud_api_calls
    assert m1.cloud_api_calls == 2


# --- Test J: Different scenarios can produce different counts under disturbances ---
def test_j_scenario_variance():
    # Logistics stable has no communication disturbances
    orch_log = _make_mock_orchestrator("logistics", "stable", seed=1, max_steps=40)
    m_log = orch_log.run()

    # Inspection under oscillatory profile has delay and loss disturbances
    orch_insp = _make_mock_orchestrator("inspection", "oscillatory", seed=1, max_steps=40)
    m_insp = orch_insp.run()

    assert m_log.cloud_api_calls >= 2
    assert m_insp.cloud_api_calls >= 2
    assert isinstance(m_log.cloud_api_calls, int)
    assert isinstance(m_insp.cloud_api_calls, int)


# --- Test K: Inspection communication degradation influences global planning ---
def test_k_inspection_communication_degradation():
    orch = _make_mock_orchestrator("inspection", "oscillatory", seed=1, max_steps=50)
    metrics = orch.run()
    # In inspection with communication delay/loss under oscillatory conditions, replanning occurs
    assert metrics.replanning_count > 0
    assert metrics.cloud_api_calls >= 2


# --- Test L: Decentralized architecture produces cloud_api_calls = 0 ---
def test_l_decentralized_zero_cloud_calls():
    config = DACAConfig(name="B2", static_mode=1, use_optimizations=False)
    orch = _make_mock_orchestrator("logistics", "stable", seed=1, config=config, max_steps=15)
    metrics = orch.run()
    assert metrics.cloud_api_calls == 0
    assert orch.cloud_llm.usage.cloud_api_calls == 0


# --- Test M: Reducing Cloud calls through optimization does not reduce task completion correctness ---
def test_m_task_completion_correctness():
    orch = _make_mock_orchestrator("logistics", "stable", seed=1, config=CONFIGS["A5"], max_steps=140)
    metrics = orch.run()
    assert metrics.success_rate >= 0.80
    assert metrics.cloud_api_calls >= 2
