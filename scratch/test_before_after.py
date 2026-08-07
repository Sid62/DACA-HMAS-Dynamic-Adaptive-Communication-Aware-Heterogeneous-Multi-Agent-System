import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS

def run_seed(scenario: str, seed: int, max_steps: int = 50) -> dict:
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
    result = orch.run()
    return result.to_dict()

if __name__ == "__main__":
    r_insp = run_seed("inspection", seed=1, max_steps=50)
    r_logi = run_seed("logistics", seed=1, max_steps=50)
    print("=== INSPECTION SEED 1 ===")
    print(json.dumps(r_insp, indent=2))
    print("=== LOGISTICS SEED 1 ===")
    print(json.dumps(r_logi, indent=2))
