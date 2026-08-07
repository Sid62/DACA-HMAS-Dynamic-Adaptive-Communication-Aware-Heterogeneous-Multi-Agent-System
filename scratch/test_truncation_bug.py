"""Reproduce exact truncation bug where text.rindex('}') grabs last nested object."""

from src.llm.cloud_llm_client import CloudLLMClient

def reproduce_truncation_bug():
    client = CloudLLMClient()
    agents = [
        {"id": "uav_0", "type": "uav"},
        {"id": "uav_1", "type": "uav"},
        {"id": "vehicle_4", "type": "vehicle"},
        {"id": "robot_6", "type": "robot"},
    ]

    # Simulating an LLM response truncated at max_tokens:
    # Outer JSON was {"coalitions": [{"coalition_id": 0, ...}, {"coalition_id": 1, ...
    # But got cut off mid-way!
    truncated_raw = '{"coalitions": [{"coalition_id": 0, "members": ["uav_0", "uav_1"]}, {"coalition_id": 1, "members": ["vehicle_4"'

    print("RAW TRUNCATED RESPONSE:\n", truncated_raw)
    
    parsed_json = client._parse_json(truncated_raw)
    print("\nparsed_json from _parse_json():\n", parsed_json)

    norm, fallback_used, stripped = client._parse_coalitions_response(truncated_raw, agents)
    print("\n_parse_coalitions_response() -> Fallback Used:", fallback_used)

if __name__ == "__main__":
    reproduce_truncation_bug()
