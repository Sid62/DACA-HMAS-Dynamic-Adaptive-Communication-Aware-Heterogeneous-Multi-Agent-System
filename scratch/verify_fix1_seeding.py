"""Verification script for Fix 1: Non-deterministic seeding (R1).

Runs scenario creation across 3 independent subprocesses with PYTHONHASHSEED=0, 1, 2
and verifies that subtask targets and priorities are 100% byte-identical.
"""

import os
import subprocess
import sys
import json
from pathlib import Path

# Ensure local workspace root is at sys.path[0]
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

SUBPROCESS_CODE = f"""
import sys, os
sys.path.insert(0, r"{WORKSPACE_ROOT}")
import json
from src.env.scenarios import get_scenario

thresholds = {{}}
scenarios_to_check = ["logistics", "inspection", "search_rescue"]
result = {{}}

for sc_name in scenarios_to_check:
    scenario = get_scenario(sc_name, thresholds, seed=42)
    subtasks_data = []
    for st in scenario.subtasks:
        subtasks_data.append({{
            "id": st.subtask_id,
            "target": {{"x": st.target.x, "y": st.target.y}},
            "priority": st.priority,
            "required_skills": st.required_skills
        }})
    result[sc_name] = {{
        "subtasks": subtasks_data,
        "metadata": {{k: v for k, v in scenario.metadata.items() if isinstance(v, (int, float, str))}}
    }}

print(json.dumps(result))
"""

def main():
    outputs = []
    seeds = [0, 1, 2]
    
    print("--- Running Verification for Fix 1 (R1 Seeding) ---")
    for hash_seed in seeds:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(hash_seed)
        env["PYTHONPATH"] = WORKSPACE_ROOT
        
        proc = subprocess.run(
            [sys.executable, "-c", SUBPROCESS_CODE],
            env=env,
            capture_output=True,
            text=True,
            cwd=WORKSPACE_ROOT
        )
        
        if proc.returncode != 0:
            print(f"[FAIL] Subprocess with PYTHONHASHSEED={hash_seed} exited with code {proc.returncode}")
            print(f"Error output:\n{proc.stderr}")
            sys.exit(1)
            
        outputs.append(proc.stdout.strip())
        print(f"PYTHONHASHSEED={hash_seed}: Generated scenario output ({len(proc.stdout.strip())} bytes)")

    # Assert byte-identical equality across all 3 runs
    if outputs[0] == outputs[1] == outputs[2]:
        print("\n[PASS] Scenario generation is 100% byte-identical across PYTHONHASHSEED=0, 1, 2!")
    else:
        print("\n[FAIL] Non-determinism detected! Outputs differed across PYTHONHASHSEED values.")
        if outputs[0] != outputs[1]:
            print(f"Difference between 0 and 1:\nSeed 0: {outputs[0]}\nSeed 1: {outputs[1]}")
        sys.exit(1)

if __name__ == "__main__":
    main()
