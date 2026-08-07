import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS

def run_scenario(scenario: str, profile: str, seed: int, max_steps: int = 60) -> dict:
    cfg = CONFIGS["A5"]
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile=profile,
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
    r1 = run_scenario("inspection", "oscillatory", seed=1, max_steps=60)
    r2 = run_scenario("logistics", "oscillatory", seed=2, max_steps=60)

    print("=== VALIDATION CHECK RUN 1 (INSPECTION) ===")
    bd1 = r1["communication_step_breakdown"]
    p_calc1 = bd1.get("global_planning", 0) + bd1.get("dispatch", 0)
    t_calc1 = sum(bd1.values())
    print(f"Breakdown: {json.dumps(bd1)}")
    print(f"communication_steps: {r1['communication_steps']} (Sum={t_calc1})")
    print(f"paper_communication_steps: {r1['paper_communication_steps']} (Calculated={p_calc1})")
    print(f"Matches: {r1['paper_communication_steps'] == p_calc1 and r1['communication_steps'] == t_calc1}")

    print("\n=== VALIDATION CHECK RUN 2 (LOGISTICS) ===")
    bd2 = r2["communication_step_breakdown"]
    p_calc2 = bd2.get("global_planning", 0) + bd2.get("dispatch", 0)
    t_calc2 = sum(bd2.values())
    print(f"Breakdown: {json.dumps(bd2)}")
    print(f"communication_steps: {r2['communication_steps']} (Sum={t_calc2})")
    print(f"paper_communication_steps: {r2['paper_communication_steps']} (Calculated={p_calc2})")
    print(f"Matches: {r2['paper_communication_steps'] == p_calc2 and r2['communication_steps'] == t_calc2}")

    print("\n=== JSON OUTPUT EXCERPT ===")
    excerpt = {
        "communication_steps": r1["communication_steps"],
        "paper_communication_steps": r1["paper_communication_steps"],
        "communication_step_breakdown": r1["communication_step_breakdown"]
    }
    print(json.dumps(excerpt, indent=2))
