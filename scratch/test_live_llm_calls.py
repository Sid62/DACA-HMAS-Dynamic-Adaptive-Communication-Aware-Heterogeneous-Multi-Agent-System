"""Task 1 & Task 2 Test Harness: Run form_coalitions multiple times with live/mock LLM and log full raw text."""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.llm.cloud_llm_client import CloudLLMClient

def run_multi_call_test(n_calls=10):
    client = CloudLLMClient()
    print(f"=== TESTING {n_calls} CONSECUTIVE form_coalitions CALLS ===")
    
    subtasks = [
        {"id": "T_0", "description": "inspect tower 1", "required_skills": ["inspect", "sense"]},
        {"id": "T_1", "description": "inspect tower 2", "required_skills": ["navigate", "inspect"]},
        {"id": "T_2", "description": "transport goods", "required_skills": ["sense", "lift"]},
    ]
    agents = [
        {"id": "uav_0", "type": "uav", "skills": ["inspect", "sense"]},
        {"id": "uav_1", "type": "uav", "skills": ["navigate", "inspect"]},
        {"id": "vehicle_4", "type": "vehicle", "skills": ["sense", "lift"]},
        {"id": "robot_6", "type": "robot", "skills": ["inspect", "sense"]},
    ]

    records = []
    for i in range(1, n_calls + 1):
        client.current_step = i * 10
        # Call form_coalitions
        res = client.form_coalitions(subtasks, agents)
        
        # Check success: fallback used if singletons generated or raw response failed
        is_singleton_fallback = (len(res) == len(agents) and all(len(c.get("members", [])) == 1 for c in res))
        success = not is_singleton_fallback
        
        records.append({
            "call_index": i,
            "success": "Y" if success else "N",
            "coalitions_returned": res,
        })
        print(f"Call {i:2d} | Success: {'Y' if success else 'N'} | Coalitions: {res}")

if __name__ == "__main__":
    run_multi_call_test(10)
