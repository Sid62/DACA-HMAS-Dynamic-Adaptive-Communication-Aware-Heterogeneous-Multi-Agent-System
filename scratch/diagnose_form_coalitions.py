"""Task 1 Diagnostic Script: Capture and characterize form_coalitions calls across 3 seeds."""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator

def run_seed_and_capture(seed, scenario="inspection", profile="oscillatory"):
    print(f"\n=================== RUN SEED {seed} ({scenario}, {profile}) ===================")
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile=profile,
        seed=seed,
        config=CONFIGS["A5"],
        max_steps=200,
    )

    client = orch.cloud_llm
    orig_form_coalitions = client.form_coalitions
    call_records = []
    call_counter = 0

    def monitored_form_coalitions(subtasks, agents, distance_matrix=None, cqi_matrix=None):
        nonlocal call_counter
        call_counter += 1
        current_idx = call_counter

        # Calculate prompt_chars
        from src.config import get_thresholds
        from src.llm.prompts import format_prompt
        th = get_thresholds()
        try:
            prompt = format_prompt(
                "coalition",
                subtasks=json.dumps(subtasks),
                agents=json.dumps(agents),
                distance_matrix=json.dumps(distance_matrix),
                cqi_matrix=json.dumps(cqi_matrix),
                c1=str(th.get("C1", 50.0)),
                gamma_min=str(th.get("gamma_min", 0.3)),
            )
        except Exception:
            prompt = f"subtasks: {json.dumps(subtasks)}"
        
        prompt_chars = len(prompt)

        # Call original form_coalitions
        res = orig_form_coalitions(subtasks, agents, distance_matrix, cqi_matrix)

        # Determine if parsed successfully or used singletons fallback
        # Singletons fallback creates 1-agent coalitions for every agent
        is_singleton = (len(res) == len(agents) and all(len(c.get("members", [])) == 1 for c in res))
        success = not is_singleton

        call_records.append({
            "call_index": current_idx,
            "step": client.current_step,
            "prompt_chars": prompt_chars,
            "success": "Y" if success else "N",
            "coalition_count": len(res),
        })
        return res

    client.form_coalitions = monitored_form_coalitions
    orch.run()

    print(f"\nSEED {seed} TABLE:")
    print("call_index | step | prompt_chars | success | coalition_count")
    print("-" * 55)
    for r in call_records:
        print(f"{r['call_index']:^10} | {r['step']:^4} | {r['prompt_chars']:^12} | {r['success']:^7} | {r['coalition_count']:^15}")
    return call_records

if __name__ == "__main__":
    records_s1 = run_seed_and_capture(1, "inspection", "oscillatory")
    records_s2 = run_seed_and_capture(2, "inspection", "oscillatory")
    records_s3 = run_seed_and_capture(3, "inspection", "oscillatory")
