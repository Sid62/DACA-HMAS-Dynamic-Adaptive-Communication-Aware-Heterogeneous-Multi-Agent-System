import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS

def run_test(scenario, profile, seed, max_steps, name):
    print(f"--- Running {name}: scenario={scenario}, profile={profile}, seed={seed}, max_steps={max_steps} ---")
    config = CONFIGS["A5"]
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile=profile,
        seed=seed,
        config=config,
        max_steps=max_steps,
    )
    # Enable mock mode for offline empirical probing
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True

    metrics = orch.run()
    d = metrics.to_dict()
    return d

if __name__ == "__main__":
    r1 = run_test("inspection", "oscillatory", 1, 30, "Run 1")
    r2 = run_test("logistics", "oscillatory", 2, 60, "Run 2")
    r3 = run_test("search_rescue", "oscillatory", 3, 100, "Run 3")

    print("\n=== SUMMARY OF EMPIRICAL VERIFICATION ===")
    for idx, r in enumerate([r1, r2, r3], 1):
        print(f"\nRun {idx} ({r['scenario']}):")
        print(f"  memory_mb: {r['memory_mb']}")
        print(f"  computation_s: {r['computation_s']}")
        print(f"  total_wall_clock_s: {r['total_wall_clock_s']}")
        print(f"  api_calls (cloud/device/total): {r['cloud_planning_calls']}/{r['device_planning_calls']}/{r['api_calls']}")
        print(f"  communication_steps: {r['communication_steps']}")
        print(f"  communication_step_breakdown: {json.dumps(r['communication_step_breakdown'])}")
        print(f"  switch_count: {r['switch_count']}")
        print(f"  success_rate: {r['success_rate']}")

    print("\n=== FULL JSON EXCERPT RUN 1 ===")
    print(json.dumps(r1, indent=2))

    print("\n=== FULL JSON EXCERPT RUN 2 ===")
    print(json.dumps(r2, indent=2))
