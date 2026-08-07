import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS

def run_experiment(scenario: str, seed: int, max_steps: int = 50) -> dict:
    cfg = CONFIGS["A5"]
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile="oscillatory",
        seed=seed,
        config=cfg,
        max_steps=max_steps,
    )
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
    return orch.run().to_dict()

if __name__ == "__main__":
    insp_run = run_experiment("inspection", seed=1, max_steps=30)
    logi_run = run_experiment("logistics", seed=1, max_steps=30)
    sr_run   = run_experiment("search_rescue", seed=1, max_steps=30)

    print("=== SCENARIO COMPARISON (CONSTRAINT 4) ===")
    print(f"Inspection memory_mb:    {insp_run['memory_mb']}")
    print(f"Logistics memory_mb:     {logi_run['memory_mb']}")
    print(f"Search & Rescue memory_mb: {sr_run['memory_mb']}")
    print(f"Values differ: {insp_run['memory_mb'] != logi_run['memory_mb'] != sr_run['memory_mb']}")

    print("\n=== FULL JSON EXCERPT AFTER FIX (INSPECTION SEED 1) ===")
    print(json.dumps(insp_run, indent=2))
