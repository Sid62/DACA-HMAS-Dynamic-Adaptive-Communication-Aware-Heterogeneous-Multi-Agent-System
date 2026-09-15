"""Test suite for Cloud Planning Single Source of Truth (SSoT) and metric accounting.

Verifies:
- cloud_planning_calls is the authoritative mutable counter
- cloud_api_calls is an exact read-only property alias
- device_planning_calls is the authoritative mutable counter
- device_api_calls is an exact read-only property alias
- total_api_calls and api_calls equal cloud_planning_calls + device_planning_calls
- Cache hits do not increment physical call counters
- Retries track wire network calls without double counting planner operations
- Dynamic replanning naturally increments counters beyond 2 when genuine global events occur
- Decentralized runs produce naturally zero Cloud calls
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import numpy as np

from src.coordination.centralized_hybrid import CentralizedHybridCoordinator
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator, DACAConfig
from src.coordination.plan_continuity import PlanContinuityEngine
from src.coordination.replan_trigger import PlanState, should_replan
from src.env.agents import AgentFleet, AgentState, AgentType, Position, KinematicsConfig
from src.env.scenarios import Subtask
from src.llm.cloud_llm_client import CloudLLMClient
from src.llm.device_llm_client import DeviceLLMClient
from src.metrics.evaluation import ExperimentMetrics, MetricsCollector

KIN = {
    "uav": KinematicsConfig(13.0, 1.5),
    "vehicle": KinematicsConfig(9.0, 0.8),
    "robot": KinematicsConfig(4.25, 2.0),
}


def _create_mock_env():
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
                "instruction": "Decompose mission into task assignments.",
                "agents": self.fleet.to_dict_list(),
                "subtasks": [
                    {"id": s.subtask_id, "skills": s.required_skills, "target": [s.target.x, s.target.y]}
                    for s in self.subtask_list
                ],
            }

    return MockEnv()


# ── Test A: One uncached decompose() ──────────────────────────────────────────
def test_a_one_uncached_decompose():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    client.decompose(
        "Decompose mission",
        [{"id": "uav_0", "skills": ["navigate"]}],
        [{"id": "T_0", "skills": ["navigate"]}],
    )
    assert client.usage.cloud_planning_calls == 1
    assert client.usage.cloud_api_calls == 1


# ── Test B: One uncached form_coalitions() ─────────────────────────────────────
def test_b_one_uncached_form_coalitions():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    client.form_coalitions(
        [{"id": "T_0", "skills": ["navigate"]}],
        [{"id": "uav_0", "skills": ["navigate"]}],
        distance_matrix=[[0.0]],
        cqi_matrix=[[1.0]],
    )
    assert client.usage.cloud_planning_calls == 1
    assert client.usage.cloud_api_calls == 1


# ── Test C: Two actual Cloud planner operations ───────────────────────────────
def test_c_two_actual_cloud_planner_operations():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    client.decompose(
        "Decompose mission",
        [{"id": "uav_0", "skills": ["navigate"]}],
        [{"id": "T_0", "skills": ["navigate"]}],
    )
    client.form_coalitions(
        [{"id": "T_0", "skills": ["navigate"]}],
        [{"id": "uav_0", "skills": ["navigate"]}],
        distance_matrix=[[0.0]],
        cqi_matrix=[[1.0]],
    )
    assert client.usage.cloud_planning_calls == 2
    assert client.usage.cloud_api_calls == 2


# ── Test D: Cloud cache hit ───────────────────────────────────────────────────
def test_d_cloud_cache_hit(tmp_path):
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": True, "cache_dir": str(tmp_path)})
    client.semantic_cache.enabled = False

    # First call: cache miss
    res1 = client.complete("Test prompt for cloud cache")
    assert client.usage.logical_requests == 1
    assert client.usage.cache_hits == 0
    assert client.usage.cloud_planning_calls == 1
    assert client.usage.cloud_api_calls == 1

    # Second call: disk cache hit
    res2 = client.complete("Test prompt for cloud cache")
    assert res1 == res2
    assert client.usage.logical_requests == 2
    assert client.usage.cache_hits == 1
    assert client.usage.cloud_planning_calls == 1  # Unchanged
    assert client.usage.cloud_api_calls == 1       # Unchanged


# ── Test E: Device cache hit ──────────────────────────────────────────────────
def test_e_device_cache_hit(tmp_path):
    client = DeviceLLMClient(config={"use_mock": True, "cache_responses": True, "cache_dir": str(tmp_path)}, node_id="uav")

    # First call: miss
    client.complete("Local reasoning dispatch prompt")
    assert client.usage.device_planning_calls == 1
    assert client.usage.device_api_calls == 1
    assert client.usage.cache_hits == 0

    # Second call: hit
    client.complete("Local reasoning dispatch prompt")
    assert client.usage.device_planning_calls == 1  # Unchanged
    assert client.usage.device_api_calls == 1       # Unchanged
    assert client.usage.cache_hits == 1


# ── Test F: Device cache miss ─────────────────────────────────────────────────
def test_f_device_cache_miss():
    client = DeviceLLMClient(config={"use_mock": True, "cache_responses": False}, node_id="uav")
    client.complete("Prompt 1")
    assert client.usage.device_planning_calls == 1
    assert client.usage.device_api_calls == 1

    client.complete("Prompt 2")
    assert client.usage.device_planning_calls == 2
    assert client.usage.device_api_calls == 2


# ── Test G: Global replan with one invalid planning artifact ──────────────────
def test_g_global_replan_with_one_invalid_artifact():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    client.semantic_cache.enabled = False
    coord = CentralizedHybridCoordinator(cloud_llm=client)
    env = _create_mock_env()

    # Initial plan: 2 calls (decompose + form_coalitions)
    coord.plan(env, force_replan=True, replan_reason="mission_initialization")
    assert client.usage.cloud_planning_calls == 2

    # Global replan with decomposition invalid ONLY
    coord.plan(env, force_replan=True, replan_reason="task_reassignment", invalid_artifacts={"decomposition"})
    # Exactly +1 Cloud operation occurs
    assert client.usage.cloud_planning_calls == 3
    assert client.usage.cloud_api_calls == 3


# ── Test H: Global replan with multiple invalid artifacts ─────────────────────
def test_h_global_replan_with_multiple_invalid_artifacts():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    client.semantic_cache.enabled = False
    coord = CentralizedHybridCoordinator(cloud_llm=client)
    env = _create_mock_env()

    # Initial plan: 2 calls
    coord.plan(env, force_replan=True, replan_reason="mission_initialization")
    assert client.usage.cloud_planning_calls == 2

    # Global replan with BOTH decomposition and coalitions invalid
    coord.plan(env, force_replan=True, replan_reason="severe_global_disruption", invalid_artifacts={"decomposition", "coalitions"})
    # Exactly +2 Cloud operations occur
    assert client.usage.cloud_planning_calls == 4
    assert client.usage.cloud_api_calls == 4


# ── Test I: Repeated valid global replans (NOT locked at 2) ───────────────────
def test_i_repeated_valid_global_replans():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    client.semantic_cache.enabled = False
    coord = CentralizedHybridCoordinator(cloud_llm=client)
    env = _create_mock_env()

    coord.plan(env, force_replan=True, replan_reason="mission_initialization")
    assert client.usage.cloud_planning_calls == 2

    # Sequence of valid global replanning events
    coord.plan(env, force_replan=True, replan_reason="task_completed_needs_reassignment:['T_0']")
    assert client.usage.cloud_planning_calls == 3

    coord.plan(env, force_replan=True, replan_reason="packet_loss_crossed_threshold:0.50")
    assert client.usage.cloud_planning_calls == 4

    coord.plan(env, force_replan=True, replan_reason="cqi_changed_significantly:0.35")
    assert client.usage.cloud_planning_calls == 5

    # Proves the system is NOT permanently locked at 2
    assert client.usage.cloud_planning_calls > 2
    assert client.usage.cloud_api_calls == 5


# ── Test J: Local repair (0 additional Cloud calls) ───────────────────────────
def test_j_local_repair():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    client.semantic_cache.enabled = False
    cont = PlanContinuityEngine(r_reach=100.0)
    coord = CentralizedHybridCoordinator(cloud_llm=client, continuity_engine=cont)
    env = _create_mock_env()

    coord.plan(env, force_replan=True, replan_reason="mission_initialization")
    assert client.usage.cloud_planning_calls == 2

    # Standalone execution where plan continuity continues valid plan
    coord.plan(env, force_replan=False)
    # 0 additional Cloud calls
    assert client.usage.cloud_planning_calls == 2
    assert client.usage.cloud_api_calls == 2

    # Verify should_replan returns local scope for recoverable problem
    plan_state = PlanState(initialized=True, last_replan_step=-10)
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
        AgentState("uav_1", AgentType.UAV, Position(12.0, 12.0), skills=["navigate", "sense", "inspect"]),
    ], KIN)
    subtasks = [
        Subtask("T_0", "nav", Position(10.0, 10.0), required_skills=["navigate"]),
        Subtask("T_1", "nav2", Position(12.0, 12.0), required_skills=["navigate"]),
    ]
    coalitions = [{"coalition_id": 0, "members": ["uav_0", "uav_1"]}]
    cont.set_active_plan({"T_0": ["uav_0"], "T_1": ["uav_0"]}, coalitions, subtasks, mode=0)
    decision = should_replan(plan_state, subtasks, fleet, coalitions, mode=0, continuity_engine=cont)
    assert decision.scope == "local"
    # Local scope means 0 Cloud calls


# ── Test K: Decentralized-only run ────────────────────────────────────────────
def test_k_decentralized_only_run(tmp_path):
    config = DACAConfig(name="B2", static_mode=1, use_optimizations=False)
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=42,
        config=config,  # Decentralized config
        max_steps=10,
    )
    orch.cloud_llm.config["use_mock"] = True
    orch.cloud_llm.config["cache_dir"] = str(tmp_path)
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
        dc.config["cache_dir"] = str(tmp_path)

    metrics = orch.run()
    assert metrics.cloud_planning_calls == 0
    assert metrics.cloud_api_calls == 0


# ── Test L: Metric alias consistency across runs ──────────────────────────────
def test_l_metric_alias_consistency_across_runs():
    metrics = ExperimentMetrics(
        config_name="hybrid",
        scenario="logistics",
        network_profile="stable",
        seed=1,
        success_rate=1.0,
        steps=10,
        cloud_tokens=500,
        device_tokens=200,
        total_tokens=700,
        cloud_planning_calls=4,
        device_planning_calls=3,
    )
    assert metrics.cloud_api_calls == metrics.cloud_planning_calls == 4
    d = metrics.to_dict()
    assert d["cloud_api_calls"] == d["cloud_planning_calls"] == 4


# ── Test M: Device alias consistency ──────────────────────────────────────────
def test_m_device_alias_consistency():
    metrics = ExperimentMetrics(
        config_name="hybrid",
        scenario="logistics",
        network_profile="stable",
        seed=1,
        success_rate=1.0,
        steps=10,
        cloud_tokens=500,
        device_tokens=200,
        total_tokens=700,
        cloud_planning_calls=2,
        device_planning_calls=7,
    )
    assert metrics.device_api_calls == metrics.device_planning_calls == 7
    d = metrics.to_dict()
    assert d["device_api_calls"] == d["device_planning_calls"] == 7


# ── Test N: Total physical call consistency ───────────────────────────────────
def test_n_total_physical_call_consistency():
    metrics = ExperimentMetrics(
        config_name="hybrid",
        scenario="logistics",
        network_profile="stable",
        seed=1,
        success_rate=1.0,
        steps=10,
        cloud_tokens=500,
        device_tokens=200,
        total_tokens=700,
        cloud_planning_calls=3,
        device_planning_calls=5,
    )
    assert metrics.api_calls == 8
    assert metrics.total_api_calls == 8
    d = metrics.to_dict()
    assert d["api_calls"] == 8
    assert d["total_api_calls"] == 8


# ── Test O: Logical vs physical ───────────────────────────────────────────────
def test_o_logical_vs_physical():
    metrics = ExperimentMetrics(
        config_name="hybrid",
        scenario="logistics",
        network_profile="stable",
        seed=1,
        success_rate=1.0,
        steps=10,
        cloud_tokens=500,
        device_tokens=200,
        total_tokens=700,
        cloud_planning_calls=3,
        device_planning_calls=2,
        logical_llm_requests=8,
    )
    assert metrics.logical_llm_requests >= (metrics.cloud_planning_calls + metrics.device_planning_calls)
    assert metrics.logical_requests == metrics.logical_llm_requests == 8
    d = metrics.to_dict()
    assert d["logical_requests"] == d["logical_llm_requests"] == 8


# ── Test P: Cache disabled ────────────────────────────────────────────────────
def test_p_cache_disabled():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False})
    client.complete("prompt A")
    client.complete("prompt A")
    # Both execute because cache is disabled
    assert client.usage.cloud_planning_calls == 2
    assert client.usage.cloud_api_calls == 2


# ── Test Q: Retry case ────────────────────────────────────────────────────────
def test_q_retry_case():
    client = CloudLLMClient(config={"use_mock": False, "cache_responses": False})
    client.max_retries = 3

    # Simulate attempt 1 fails with transport error, attempt 2 succeeds
    attempts = [0]
    def mock_api_call(prompt, system):
        attempts[0] += 1
        if attempts[0] == 1:
            raise ConnectionError("Temporary connection reset")
        return '{"result": "success"}', 50, 20, 70, True

    client._api_call = mock_api_call

    client.complete("Plan prompt with retry")

    # Authoritative semantics:
    # 1 planner operation succeeded
    assert client.usage.cloud_planning_calls == 1
    assert client.usage.cloud_api_calls == 1
    # 2 network attempts across the wire
    assert client.usage.cloud_network_calls == 2
    # 1 failed network attempt
    assert client.usage.cloud_failed_attempts == 1
    # 1 retried planner operation
    assert client.usage.retried_calls == 1


# ── Test R: Different seeds and scenarios ─────────────────────────────────────
def test_r_different_seeds_scenarios(tmp_path):
    # Verify that different scenarios/seeds run naturally without hardcoded constraints
    orch1 = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=1,
        config=CONFIGS["A5"],
        max_steps=20,
    )
    orch1.cloud_llm.config["use_mock"] = True
    orch1.cloud_llm.config["cache_dir"] = str(tmp_path / "seed1")
    for dc in orch1.device_llms.values():
        dc.config["use_mock"] = True
        dc.config["cache_dir"] = str(tmp_path / "seed1")
    m1 = orch1.run()

    orch2 = DACAOrchestrator(
        scenario="search_rescue",
        network_profile="stable",
        seed=2,
        config=CONFIGS["A5"],
        max_steps=20,
    )
    orch2.cloud_llm.config["use_mock"] = True
    orch2.cloud_llm.config["cache_dir"] = str(tmp_path / "seed2")
    for dc in orch2.device_llms.values():
        dc.config["use_mock"] = True
        dc.config["cache_dir"] = str(tmp_path / "seed2")
    m2 = orch2.run()

    # In both runs, single-source-of-truth invariants hold unconditionally
    assert m1.cloud_api_calls == m1.cloud_planning_calls
    assert m2.cloud_api_calls == m2.cloud_planning_calls
    assert m1.device_api_calls == m1.device_planning_calls
    assert m2.device_api_calls == m2.device_planning_calls
    assert m1.api_calls == m1.cloud_planning_calls + m1.device_planning_calls
    assert m2.api_calls == m2.cloud_planning_calls + m2.device_planning_calls
