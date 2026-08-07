"""Test _parse_json_robust auto-repair algorithm on severely truncated LLM outputs."""

import json

def parse_json_robust(text: str) -> dict:
    if not text:
        return {}
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    if "```" in text:
        import re
        matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        for m in matches:
            try:
                start = m.index("{")
                end = m.rindex("}") + 1
                return json.loads(m[start:end])
            except (ValueError, json.JSONDecodeError):
                pass

    try:
        start = text.index("{")
        sub = text[start:].strip()
        stack = []
        in_string = False
        escape = False
        repaired_chars = []
        for ch in sub:
            repaired_chars.append(ch)
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if stack and stack[-1] == ch:
                        stack.pop()
        
        if in_string:
            repaired_chars.append('"')
        while stack:
            repaired_chars.append(stack.pop())
        
        repaired_text = "".join(repaired_chars)
        return json.loads(repaired_text)
    except (ValueError, json.JSONDecodeError):
        pass

    return {}

def test_robust_repair():
    truncated_1 = '{"coalitions": [{"coalition_id": 0, "members": ["uav_0", "uav_1"]}, {"coalition_id": 1, "members": ["vehicle_4"'
    truncated_2 = '{"coalitions": [{"coalition_id": 0, "members": ["uav_0", "uav_1"]}'
    truncated_3 = 'Here is the response:\n```json\n{"coalitions": [{"coalition_id": 0, "members": ["uav_0", "uav_1"]}, {"coalition_id": 1, "members": ['

    for i, t in enumerate([truncated_1, truncated_2, truncated_3], 1):
        res = parse_json_robust(t)
        print(f"[Truncated Test {i}] -> Extracted: {res}")

if __name__ == "__main__":
    test_robust_repair()
