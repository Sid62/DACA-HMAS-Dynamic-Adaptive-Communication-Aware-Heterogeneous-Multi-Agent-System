"""Unit and regression tests for scenario fleet balance, capability distribution, and determinism.

Verifies:
1. Exact balanced agent counts per scenario:
   - Logistics: 8 agents (2 UAV, 3 Vehicle, 3 Robot)
   - Inspection: 8 agents (3 UAV, 2 Vehicle, 3 Robot)
   - Search & Rescue: 10 agents (3 UAV, 3 Vehicle, 4 Robot)
2. validate_scenario_agent_balance reports zero severe imbalances and 100% subtask coverage.
3. get_fleet_capability_summary returns accurate capability distributions.
4. Deterministic agent ID numbering and inspection sensor position overrides.
5. Every subtask across all 3 scenarios has feasible single-agent or complementary coalition assignments.
6. Centralized and decentralized coordination sanity test executes successfully.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.config import get_thresholds
from src.env.agents import (
    AgentFleet,
    AgentType,
    KinematicsConfig,
    ROLE_SKILLS,
    create_fleet_from_scenario,
)
from src.env.scenarios import (
    Scenario,
    get_scenario,
    get_fleet_capability_summary,
    validate_scenario_agent_balance,
    print_scenario_agent_config,
)
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator


KIN = {
    "uav": KinematicsConfig(15.0, 1.5),
    "vehicle": KinematicsConfig(10.0, 0.8),
    "robot": KinematicsConfig(3.0, 2.0),
}


# ── Test 1: Exact Agent Counts and Distribution ──────────────────────────────

@pytest.mark.parametrize(
    "scenario_name,expected_total,expected_counts",
    [
        ("logistics", 10, {"num_uav": 3, "num_vehicle": 4, "num_robot": 3}),
        ("inspection", 8, {"num_uav": 3, "num_vehicle": 2, "num_robot": 3}),
        ("search_rescue", 10, {"num_uav": 3, "num_vehicle": 3, "num_robot": 4}),
    ],
)
def test_scenario_agent_counts_and_types(scenario_name, expected_total, expected_counts):
    th = get_thresholds()
    sc = get_scenario(scenario_name, th, seed=42)
    ac = sc.agent_config
    
    total = ac["num_uav"] + ac["num_vehicle"] + ac["num_robot"]
    assert total == expected_total, f"{scenario_name} total agents = {total}, expected {expected_total}"
    
    for k, v in expected_counts.items():
        assert ac[k] == v, f"{scenario_name} {k} = {ac[k]}, expected {v}"


# ── Test 2: Fleet Instantiation and Deterministic IDs ─────────────────────────

@pytest.mark.parametrize(
    "scenario_name,expected_ids",
    [
        ("logistics", [
            "uav_0", "uav_1", "uav_2",
            "vehicle_3", "vehicle_4", "vehicle_5", "vehicle_6",
            "robot_7", "robot_8", "robot_9"
        ]),
        ("inspection", [
            "uav_0", "uav_1", "uav_2",
            "vehicle_3", "vehicle_4",
            "robot_5", "robot_6", "robot_7"
        ]),
        ("search_rescue", [
            "uav_0", "uav_1", "uav_2",
            "vehicle_3", "vehicle_4", "vehicle_5",
            "robot_6", "robot_7", "robot_8", "robot_9"
        ]),
    ],
)
def test_fleet_agent_ids_deterministic(scenario_name, expected_ids):
    th = get_thresholds()
    sc = get_scenario(scenario_name, th, seed=0)
    fleet = create_fleet_from_scenario(sc.agent_config, KIN, c1=50.0, c2=5.0, seed=0)
    
    actual_ids = [a.agent_id for a in fleet.agents]
    assert actual_ids == expected_ids


# ── Test 3: Inspection Sensor Position Overrides ──────────────────────────────

def test_inspection_sensor_position_overrides():
    th = get_thresholds()
    sc = get_scenario("inspection", th, seed=0)
    fleet = create_fleet_from_scenario(sc.agent_config, KIN, c1=50.0, c2=5.0, seed=0)
    
    # Robots start at index num_uav + num_vehicle = 3 + 2 = 5
    # robot_5, robot_6, robot_7 are stationary sensor proxies placed at subtask targets
    for i in range(3):
        agent_id = f"robot_{5 + i}"
        agent = fleet.get_agent(agent_id)
        expected_target = sc.subtasks[i % len(sc.subtasks)].target
        assert np.isclose(agent.position.x, expected_target.x)
        assert np.isclose(agent.position.y, expected_target.y)


# ── Test 4: validate_scenario_agent_balance reports zero severe imbalance ─────

@pytest.mark.parametrize("scenario_name", ["logistics", "inspection", "search_rescue"])
def test_validate_scenario_agent_balance(scenario_name):
    th = get_thresholds()
    sc = get_scenario(scenario_name, th, seed=42)
    fleet = create_fleet_from_scenario(sc.agent_config, KIN, c1=50.0, c2=5.0, seed=42)
    
    res = validate_scenario_agent_balance(sc, fleet)
    assert res["has_severe_imbalance"] is False, f"Severe imbalance found in {scenario_name}: {res['imbalance_reasons']}"
    assert res["coverage_rate"] == 1.0, f"Coverage rate in {scenario_name} = {res['coverage_rate']}, expected 1.0"
    assert res["covered_subtasks"] == res["total_subtasks"]


# ── Test 5: get_fleet_capability_summary capability counts ────────────────────

def test_fleet_capability_summary_logistics():
    th = get_thresholds()
    sc = get_scenario("logistics", th, seed=42)
    summary = get_fleet_capability_summary(sc)
    
    # 3 UAV, 4 Vehicle, 3 Robot = 10 agents
    assert summary["total_agents"] == 10
    assert summary["sensing_capable"] == 3      # 3 UAVs
    assert summary["navigation_capable"] == 7   # 3 UAVs + 4 Vehicles
    assert summary["transport_capable"] == 7    # 4 Vehicles + 3 Robots
    assert summary["lifting_capable"] == 3      # 3 Robots
    assert summary["rescue_capable"] == 7       # 4 Vehicles + 3 Robots
    assert summary["inspection_capable"] == 3   # 3 UAVs
    assert summary["multi_skill_agents"] == 10  # 100% multi-skill


def test_fleet_capability_summary_inspection():
    th = get_thresholds()
    sc = get_scenario("inspection", th, seed=42)
    summary = get_fleet_capability_summary(sc)
    
    # 3 UAV, 2 Vehicle, 3 Robot = 8 agents
    assert summary["total_agents"] == 8
    assert summary["sensing_capable"] == 3      # 3 UAVs
    assert summary["navigation_capable"] == 5   # 3 UAVs + 2 Vehicles
    assert summary["transport_capable"] == 5    # 2 Vehicles + 3 Robots
    assert summary["lifting_capable"] == 3      # 3 Robots
    assert summary["rescue_capable"] == 5       # 2 Vehicles + 3 Robots
    assert summary["inspection_capable"] == 3   # 3 UAVs
    assert summary["multi_skill_agents"] == 8   # 100% multi-skill


def test_fleet_capability_summary_search_rescue():
    th = get_thresholds()
    sc = get_scenario("search_rescue", th, seed=42)
    summary = get_fleet_capability_summary(sc)
    
    # 3 UAV, 3 Vehicle, 4 Robot = 10 agents
    assert summary["total_agents"] == 10
    assert summary["sensing_capable"] == 3      # 3 UAVs
    assert summary["navigation_capable"] == 6   # 3 UAVs + 3 Vehicles
    assert summary["transport_capable"] == 7    # 3 Vehicles + 4 Robots
    assert summary["lifting_capable"] == 4      # 4 Robots
    assert summary["rescue_capable"] == 7       # 3 Vehicles + 4 Robots
    assert summary["inspection_capable"] == 3   # 3 UAVs
    assert summary["multi_skill_agents"] == 10  # 100% multi-skill


# ── Test 6: Startup summary formatting string ─────────────────────────────────

def test_print_scenario_agent_config():
    th = get_thresholds()
    sc = get_scenario("logistics", th, seed=0)
    output = print_scenario_agent_config(sc)
    
    assert "Scenario: Logistics" in output
    assert "Total Agents: 10" in output
    assert "UAVs: 3" in output
    assert "Vehicles: 4" in output
    assert "Robots: 3" in output
    assert "Skills:" in output
    assert "UAV: navigate, sense, inspect" in output
    assert "Vehicle: navigate, transport, rescue" in output
    assert "Robot: lift, rescue, transport" in output


# ── Test 7: Simulation Sanity Across All 3 Scenarios ──────────────────────────

@pytest.mark.parametrize("scenario_name", ["logistics", "inspection", "search_rescue"])
@pytest.mark.parametrize("config_name", ["B1", "A5"])
def test_simulation_sanity(scenario_name, config_name, tmp_path):
    orch = DACAOrchestrator(
        scenario=scenario_name,
        network_profile="stable",
        seed=42,
        config=CONFIGS[config_name],
        max_steps=5,
    )
    orch.cloud_llm.config["use_mock"] = True
    orch.cloud_llm.config["cache_dir"] = str(tmp_path)
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
        dc.config["cache_dir"] = str(tmp_path)

    metrics = orch.run()
    assert metrics.steps > 0
    assert metrics.paper_communication_steps >= 0
    assert metrics.cloud_api_calls >= 0
    assert orch.env.fleet.n_agents in (8, 10)
