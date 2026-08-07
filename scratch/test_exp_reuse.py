import json
import sys
sys.path.insert(0, ".")

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS

def run_exp(reuse_enabled: bool, scenario: str = "inspection"):
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile="stable",
        seed=42,
        config=CONFIGS["B1"],
        max_steps=50,
    )
    orch.cloud_llm.config["use_mock"] = True
    orch.cloud_llm.config["experience_reuse"] = {"enabled": reuse_enabled, "store_path": "scratch/exp_test.json"}
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
        dc.config["experience_reuse"] = {"enabled": reuse_enabled, "store_path": "scratch/exp_test.json"}
    return orch.run().to_dict()

r_disabled = run_exp(False, "inspection")
r_enabled = run_exp(True, "inspection")

print("--- EXPERIENCE REUSE COMPARISON (Inspection) ---")
print(f"Disabled: success_rate={r_disabled['success_rate']}, tokens={r_disabled['tokens']}, api_calls={r_disabled['api_calls']}, replans={r_disabled['replanning_count']}, comm_steps={r_disabled['communication_steps']}")
print(f"Enabled:  success_rate={r_enabled['success_rate']}, tokens={r_enabled['tokens']}, api_calls={r_enabled['api_calls']}, replans={r_enabled['replanning_count']}, comm_steps={r_enabled['communication_steps']}")
