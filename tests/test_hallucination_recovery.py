"""Unit tests for Advanced Hallucination Recovery Pipeline (Re-query, Role Substitution, Escalation Logging)."""

from src.llm.cloud_llm_client import CloudLLMClient


def test_requery_and_role_substitution_recovery():
    """Verify that CloudLLMClient recovers from hallucinated agent IDs via re-query and role substitution."""
    client = CloudLLMClient()
    agents = [
        {"id": "uav_0", "type": "uav"},
        {"id": "uav_1", "type": "uav"},
        {"id": "vehicle_2", "type": "vehicle"},
        {"id": "robot_3", "type": "robot"},
    ]
    subtasks = [{"id": "T_0", "required_skills": ["navigate", "inspect"]}]

    # Mock response with hallucinated ID 'uav_8'
    raw_response = """
    {
      "coalitions": [
        {"coalition_id": 10, "members": ["uav_0", "uav_8"]},
        {"coalition_id": 11, "members": ["vehicle_2", "robot_3"]}
      ]
    }
    """

    # Mock complete to handle re-query
    def mock_complete(prompt, system="", caller=""):
        if "CORRECTIVE RE-QUERY" in prompt:
            # Corrective re-query returns valid coalition without 'uav_8'
            return """
            {
              "coalitions": [
                {"coalition_id": 10, "members": ["uav_0", "uav_1"]},
                {"coalition_id": 11, "members": ["vehicle_2", "robot_3"]}
              ]
            }
            """
        return raw_response

    client.complete = mock_complete

    coalitions = client.form_coalitions(subtasks, agents)

    # Check recovery stats
    stats = client.hallucination_stats
    assert stats["total_events"] == 1
    assert stats["retry_attempts"] == 1
    assert stats["retry_successes"] == 1

    # Verify returned coalitions are valid and contain valid IDs only
    assert len(coalitions) == 2
    all_members = [m for c in coalitions for m in c["members"]]
    assert "uav_8" not in all_members
    assert "uav_0" in all_members
    assert "uav_1" in all_members


def test_role_substitution_fallback_when_requery_fails():
    """Verify role substitution when re-query does not eliminate invalid IDs."""
    client = CloudLLMClient()
    agents = [
        {"id": "uav_0", "type": "uav"},
        {"id": "uav_1", "type": "uav"},
        {"id": "vehicle_2", "type": "vehicle"},
    ]
    subtasks = [{"id": "T_0", "required_skills": ["navigate"]}]

    raw_response = """
    {
      "coalitions": [
        {"coalition_id": 1, "members": ["uav_0", "uav_8"]}
      ]
    }
    """

    # Re-query also fails to fix it
    def mock_complete(prompt, system="", caller=""):
        return raw_response

    client.complete = mock_complete

    coalitions = client.form_coalitions(subtasks, agents)

    stats = client.hallucination_stats
    assert stats["total_events"] == 1
    assert stats["retry_attempts"] == 1
    assert stats["retry_successes"] == 0
    assert stats["substitutions"] == 1, "Should substitute missing uav_8 with idle uav_1"

    # Verify that uav_1 was substituted in place of uav_8
    all_members = [m for c in coalitions for m in c["members"]]
    assert "uav_8" not in all_members
    assert "uav_0" in all_members
    assert "uav_1" in all_members
