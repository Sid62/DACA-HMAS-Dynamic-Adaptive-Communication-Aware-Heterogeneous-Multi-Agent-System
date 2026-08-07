"""Capture baseline metrics for all 3 scenarios before scenario fidelity changes."""
import json
import sys
sys.path.insert(0, ".")

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS
from src.env.scenarios import get_scenario
from src.config import get_thresholds


def capture_subtask_targets(scenario_name: str, seed: int) -> list:
    """Capture subtask target coordinates for RNG-ordering verification."""
    thresholds = get_thresholds()
    scenario = get_scenario(scenario_name, thresholds, seed)
    return [(s.subtask_id, s.target.x, s.target.y) for s in scenario.subtasks]


def run_scenario(scenario_name: str, seed: int = 42, max_steps: int = 50) -> dict:
    """Run a scenario with forced mock mode."""
    cfg = CONFIGS["B1"]
    orch = DACAOrchestrator(
        scenario=scenario_name,
        network_profile="stable",
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
    results = {}
    targets = {}
    
    for scenario in ["logistics", "inspection", "search_rescue"]:
        print(f"\n{'='*60}")
        print(f"Running {scenario} seed=42...")
        print(f"{'='*60}")
        
        # Capture subtask targets for RNG verification
        targets[scenario] = capture_subtask_targets(scenario, 42)
        
        # Run scenario
        results[scenario] = run_scenario(scenario, seed=42)
        print(f"  success_rate = {results[scenario]['success_rate']}")
        print(f"  memory_mb = {results[scenario]['memory_mb']}")

    # Write baseline
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    
    for scenario in ["logistics", "inspection", "search_rescue"]:
        r = results[scenario]
        print(f"\n--- {scenario} ---")
        print(json.dumps(r, indent=2))
    
    # Write subtask targets for RNG verification
    print(f"\n{'='*60}")
    print("SUBTASK TARGETS (for RNG ordering check)")
    print(f"{'='*60}")
    for scenario, tgts in targets.items():
        print(f"\n{scenario}:")
        for tid, x, y in tgts:
            print(f"  {tid}: ({x:.15f}, {y:.15f})")
    
    # Save to file
    with open("scratch/baseline_metrics.json", "w") as f:
        json.dump({"results": results, "targets": {
            k: [(t[0], t[1], t[2]) for t in v] for k, v in targets.items()
        }}, f, indent=2)
    print("\nSaved to scratch/baseline_metrics.json")
