"""Regression tests for centralized dispatch accounting across normal and cached plan paths.

Verifies:
1. Normal planning dispatch records dispatch exactly once.
2. Reused plan with unchanged assignments records no dispatch.
3. Reused plan with changed assignments records direct dispatch exactly once.
4. Normal planning path is not double-counted.
5. paper_communication_steps dynamically equals global_planning + dispatch.
"""

import numpy as np
import pytest

from src.coordination.centralized_hybrid import CentralizedHybridCoordinator
from src.coordination.plan_continuity import PlanContinuityEngine
from src.env.agents import AgentFleet, AgentState, AgentType, Position
from src.env.scenarios import Subtask
from src.llm.device_llm_client import DeviceLLMClient
from src.metrics.communication_counter import CommunicationStepCounter


class MockDeviceClient:
    """Mock Device LLM client tracking dispatch calls."""

    def __init__(self, node_id: str, managed_agent_ids: list[str]) -> None:
        self.node_id = node_id
        self.managed_agent_ids = managed_agent_ids
        self.dispatch_calls: list[list[dict]] = []

    def dispatch(self, coalitions: list[dict], mode: int = 0) -> dict:
        self.dispatch_calls.append(coalitions)
        return {"dispatched": True}


class DummyEnv:
    """Minimal environment for testing coordinator dispatch paths."""

    def __init__(self, fleet: AgentFleet, subtasks: list[Subtask]) -> None:
        self.fleet = fleet
        self.subtask_list = subtasks
        self.scenario_name = "test_scenario"

    def get_observation(self) -> dict:
        return {
            "instruction": "test mission",
            "agents": [a.agent_id for a in self.fleet.agents],
            "subtasks": [s.subtask_id for s in self.subtask_list],
        }

    def mark_subtask_complete(self, subtask_id: str) -> None:
        for s in self.subtask_list:
            if s.subtask_id == subtask_id:
                s.completed = True


from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position

KIN = {
    "uav": KinematicsConfig(15.0, 1.5),
    "vehicle": KinematicsConfig(10.0, 0.8),
    "robot": KinematicsConfig(3.0, 2.0),
}


@pytest.fixture
def fleet():
    agents = [
        AgentState("uav_0", AgentType.UAV, Position(10.0, 10.0), skills=["navigate", "sense", "inspect"]),
        AgentState("vehicle_0", AgentType.VEHICLE, Position(15.0, 15.0), skills=["navigate", "transport", "rescue"]),
        AgentState("robot_0", AgentType.ROBOT, Position(20.0, 20.0), skills=["lift", "rescue", "transport"]),
    ]
    return AgentFleet(agents, KIN)


@pytest.fixture
def mock_device_clients():
    return {
        "uav": MockDeviceClient("uav", ["uav_0"]),
        "vehicle": MockDeviceClient("vehicle", ["vehicle_0"]),
        "robot": MockDeviceClient("robot", ["robot_0"]),
    }


def test_scenario_1_normal_planning_dispatch(fleet, mock_device_clients):
    """Scenario 1: Normal planning dispatch records dispatch counter exactly once."""
    comm_counter = CommunicationStepCounter()
    st = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"])
    env = DummyEnv(fleet, [st])

    engine = PlanContinuityEngine(r_reach=100.0)
    coord = CentralizedHybridCoordinator(
        cloud_llm=None,
        device_llms=mock_device_clients,
        decomposer=None,
        continuity_engine=engine,
    )

    # Mock decompose & form_coalitions
    coalitions = [{"coalition_id": 0, "members": ["uav_0"]}]
    assignments = {"T_0": ["uav_0"]}

    # Simulate normal planning path execution
    engine.set_active_plan(assignments, coalitions, [st], mode=0)
    dispatch_occurred = coord._dispatch_domains(coalitions)
    assert dispatch_occurred is True
    coord._last_dispatched_assignments = dict(assignments)

    # Caller accounting contract
    comm_counter.record_global_planning(1, "centralized_global_planning")
    if dispatch_occurred:
        comm_counter.record_dispatch(1, "centralized_domain_dispatch")

    assert comm_counter.breakdown["global_planning"] == 1
    assert comm_counter.breakdown["dispatch"] == 1
    assert comm_counter.paper_value == 2
    assert len(mock_device_clients["uav"].dispatch_calls) == 1


def test_scenario_2_reused_plan_assignment_unchanged(fleet, mock_device_clients):
    """Scenario 2: Reused plan with unchanged assignment performs no dispatch."""
    comm_counter = CommunicationStepCounter()
    st = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"])

    engine = PlanContinuityEngine(r_reach=100.0)
    coord = CentralizedHybridCoordinator(
        cloud_llm=None,
        device_llms=mock_device_clients,
        decomposer=None,
        continuity_engine=engine,
    )

    coalitions = [{"coalition_id": 0, "members": ["uav_0"]}]
    assignments = {"T_0": ["uav_0"]}
    engine.set_active_plan(assignments, coalitions, [st], mode=0)

    # Step 0 dispatch
    dispatch_occurred = coord._dispatch_domains(coalitions)
    coord._last_dispatched_assignments = dict(assignments)
    comm_counter.record_global_planning(1, "centralized_global_planning")
    if dispatch_occurred:
        comm_counter.record_dispatch(1, "centralized_domain_dispatch")

    assert comm_counter.breakdown["dispatch"] == 1
    assert len(mock_device_clients["uav"].dispatch_calls) == 1

    # Step 1: cached plan reuse with UNCHANGED assignment
    current_assignments = dict(assignments)
    if current_assignments != coord._last_dispatched_assignments:
        dispatch_occurred = coord._dispatch_domains(coalitions)
        if dispatch_occurred:
            comm_counter.record_dispatch(1, "centralized_domain_dispatch")
        coord._last_dispatched_assignments = dict(current_assignments)
    else:
        coord.dispatch_skipped_count += 1

    # Dispatch counter MUST NOT increase
    assert comm_counter.breakdown["dispatch"] == 1
    assert comm_counter.paper_value == 2
    assert len(mock_device_clients["uav"].dispatch_calls) == 1
    assert coord.dispatch_skipped_count == 1


def test_scenario_3_reused_plan_assignment_changed_direct_dispatch(fleet, mock_device_clients):
    """Scenario 3: Reused plan with changed assignment invokes _dispatch_domains directly and increments dispatch."""
    comm_counter = CommunicationStepCounter()
    st0 = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"], completed=True)
    st1 = Subtask("T_1", "lift_task", Position(22.0, 22.0), required_skills=["lift"])

    engine = PlanContinuityEngine(r_reach=100.0)
    coord = CentralizedHybridCoordinator(
        cloud_llm=None,
        device_llms=mock_device_clients,
        decomposer=None,
        continuity_engine=engine,
    )

    coalitions = [
        {"coalition_id": 0, "members": ["uav_0"]},
        {"coalition_id": 1, "members": ["robot_0"]},
    ]
    initial_assignments = {"T_0": ["uav_0"]}
    engine.set_active_plan(initial_assignments, coalitions, [st0, st1], mode=0)

    # Initial dispatch
    coord._dispatch_domains(coalitions)
    coord._last_dispatched_assignments = dict(initial_assignments)
    comm_counter.record_global_planning(1, "centralized_global_planning")
    comm_counter.record_dispatch(1, "centralized_domain_dispatch")
    assert comm_counter.breakdown["dispatch"] == 1
    assert comm_counter.paper_value == 2

    # Step 1: T_0 completed, robot_0 now assigned to T_1 (assignments changed without global replan)
    updated_assignments = {"T_1": ["robot_0"]}
    assert updated_assignments != coord._last_dispatched_assignments

    # Path B execution in orchestrator
    dispatch_occurred = coord._dispatch_domains(coalitions)
    assert dispatch_occurred is True
    if dispatch_occurred:
        comm_counter.record_dispatch(1, "centralized_domain_dispatch")
    coord._last_dispatched_assignments = dict(updated_assignments)

    # Dispatch counter MUST increase by 1 to reflect actual Device LLM dispatch
    assert comm_counter.breakdown["dispatch"] == 2
    assert comm_counter.breakdown["global_planning"] == 1
    # Dynamic paper communication steps naturally becomes 3
    assert comm_counter.paper_value == 3

    # Step 2: assignments remain {"T_1": ["robot_0"]}, no additional dispatch
    if updated_assignments != coord._last_dispatched_assignments:
        dispatch_occurred = coord._dispatch_domains(coalitions)
        if dispatch_occurred:
            comm_counter.record_dispatch(1, "centralized_domain_dispatch")
    else:
        coord.dispatch_skipped_count += 1

    assert comm_counter.breakdown["dispatch"] == 2
    assert comm_counter.paper_value == 3


def test_scenario_4_normal_planning_not_double_counted(fleet, mock_device_clients):
    """Scenario 4: Normal planning path records exactly one dispatch and does NOT double-count."""
    comm_counter = CommunicationStepCounter()
    st = Subtask("T_0", "nav_task", Position(12.0, 12.0), required_skills=["navigate"])
    env = DummyEnv(fleet, [st])

    engine = PlanContinuityEngine(r_reach=100.0)
    coord = CentralizedHybridCoordinator(
        cloud_llm=None,
        device_llms=mock_device_clients,
        decomposer=None,
        continuity_engine=engine,
    )

    # Centralized coordinator's _dispatch_domains itself must NOT call record_dispatch
    assert not hasattr(coord, "comm_counter")

    # When plan() runs with continuity engine reusing plan
    engine.set_active_plan({"T_0": ["uav_0"]}, [{"coalition_id": 0, "members": ["uav_0"]}], [st], mode=0)
    assignments, coalitions, cloud_reasoned, dispatch_occurred = coord.plan(env)

    assert dispatch_occurred is True
    assert cloud_reasoned is False

    # Orchestrator handles accounting
    if cloud_reasoned:
        comm_counter.record_global_planning(1, "centralized_global_planning")
    if dispatch_occurred:
        comm_counter.record_dispatch(1, "centralized_domain_dispatch")

    # Exactly 1 dispatch event recorded
    assert comm_counter.breakdown["dispatch"] == 1
    assert comm_counter.breakdown.get("global_planning", 0) == 0
    assert comm_counter.paper_value == 1


def test_scenario_5_dynamic_paper_metric_formula():
    """Scenario 5: paper_communication_steps strictly and dynamically equals global_planning + dispatch."""
    counter = CommunicationStepCounter()

    # Case: 0 planning, 0 dispatch -> 0
    assert counter.paper_value == 0

    # Case: 0 planning, 1 dispatch -> 1
    counter.record_dispatch(1, "dispatch_only")
    assert counter.paper_value == 1

    # Case: 1 planning, 1 dispatch -> 2
    counter.record_global_planning(1, "replan")
    assert counter.paper_value == 2

    # Case: 1 planning, 2 dispatches -> 3
    counter.record_dispatch(1, "additional_cached_dispatch")
    assert counter.paper_value == 3

    # Case: 2 planning, 2 dispatches -> 4
    counter.record_global_planning(1, "second_replan")
    assert counter.paper_value == 4
    assert counter.paper_value == counter.breakdown["global_planning"] + counter.breakdown["dispatch"]
