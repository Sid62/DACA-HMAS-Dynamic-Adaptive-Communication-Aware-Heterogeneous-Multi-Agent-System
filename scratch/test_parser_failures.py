"""Test _parse_json against multiple brace blocks / thinking tags / trailing prose."""

from src.llm.cloud_llm_client import CloudLLMClient

def test_multiple_braces():
    client = CloudLLMClient()
    agents = [
        {"id": "uav_0", "type": "uav"},
        {"id": "uav_1", "type": "uav"},
    ]

    # Test Case 6: Thinking / reasoning block before markdown code fence
    case6 = """Here is my reasoning: { "step": "evaluate requirements" }

```json
{
  "coalitions": [
    {"coalition_id": 0, "members": ["uav_0", "uav_1"]}
  ]
}
```"""

    # Test Case 7: Trailing reasoning block after markdown code fence
    case7 = """```json
{
  "coalitions": [
    {"coalition_id": 0, "members": ["uav_0", "uav_1"]}
  ]
}
```

Summary: Coalition 0 created with {uav_0, uav_1} for safety."""

    # Test Case 8: Code fence without outer braces matching
    case8 = """I have formed the following coalitions:

```json
{
  "coalitions": [
    {"coalition_id": 0, "members": ["uav_0", "uav_1"]}
  ]
}
```

Let me know if you want {changes}."""

    for name, raw in [("Reasoning before JSON block", case6),
                     ("Reasoning after JSON block", case7),
                     ("Braces in trailing text", case8)]:
        norm, fallback_used, stripped = client._parse_coalitions_response(raw, agents)
        print(f"[{name}] -> Fallback Used: {fallback_used} | Result: {norm}")

if __name__ == "__main__":
    test_multiple_braces()
