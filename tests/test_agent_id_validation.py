"""Unit tests for agent ID validation and hallucination protection."""

import pytest
from src.llm.cloud_llm_client import CloudLLMClient
from src.env.agents import AgentFleet, AgentState, AgentType, Position, KinematicsConfig


def test_parser_strips_hallucinated_agent_ids():
    """Verify that _normalize_coalitions filters out hallucinated agent IDs like 'uav_8'."""
    client = CloudLLMClient()
    agents = [
        {"id": "uav_0", "type": "uav"},
        {"id": "uav_1", "type": "uav"},
        {"id": "vehicle_2", "type": "vehicle"},
    ]
    raw_response = """
    {
      "coalitions": [
        {"coalition_id": 31, "members": ["uav_0", "uav_8"]},
        {"coalition_id": 32, "members": ["uav_99", "robot_8"]}
      ]
    }
    """

    coalitions, is_fallback, stripped_map = client._parse_coalitions_response(raw_response, agents)

    # Coalition 31 should have 'uav_8' stripped, keeping only 'uav_0'
    # Coalition 32 should have both 'uav_99' and 'robot_8' stripped, resulting in an empty coalition that is discarded
    assert not is_fallback
    assert len(coalitions) == 1
    assert coalitions[0]["coalition_id"] == 31
    assert coalitions[0]["members"] == ["uav_0"]


def test_fleet_has_agent_and_get_agent_error_handling():
    """Verify Fleet.has_agent and Fleet.get_agent error reporting."""
    agents = [
        AgentState(agent_id="uav_0", agent_type=AgentType.UAV, position=Position(0, 0)),
        AgentState(agent_id="uav_1", agent_type=AgentType.UAV, position=Position(10, 10)),
    ]
    kinematics = {"uav": KinematicsConfig(max_speed=5.0, max_turn_rate=1.0)}
    fleet = AgentFleet(agents, kinematics)

    assert fleet.has_agent("uav_0") is True
    assert fleet.has_agent("uav_1") is True
    assert fleet.has_agent("uav_8") is False

    # get_agent for valid ID succeeds
    a = fleet.get_agent("uav_0")
    assert a.agent_id == "uav_0"

    # get_agent for invalid ID raises ValueError with informative message
    with pytest.raises(ValueError, match="not found in fleet roster"):
        fleet.get_agent("uav_8")
