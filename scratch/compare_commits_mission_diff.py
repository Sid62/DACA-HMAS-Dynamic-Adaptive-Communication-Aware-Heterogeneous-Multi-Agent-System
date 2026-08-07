"""Compare commit 64612ca baseline vs current HEAD seed-by-seed across all scenarios."""

import json
from pathlib import Path
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator

ROOT = Path(__file__).resolve().parent.parent

SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILES = ["oscillatory"]
SEEDS = [1, 2, 3, 4, 5]

def run_current_head():
    results = {}
    for scen in SCENARIOS:
        for prof in PROFILES:
            key = f"{scen}_{prof}"
            results[key] = []
            for seed in SEEDS:
                orch = DACAOrchestrator(
                    scenario=scen,
                    network_profile=prof,
                    seed=seed,
                    config=CONFIGS["A5"],
                    max_steps=200,
                )
                metrics = orch.run().to_dict()
                results[key].append(metrics)
    return results

if __name__ == "__main__":
    head_results = run_current_head()
    print("=== CURRENT HEAD RESULTS (max_steps=200, Seeds 1..5) ===")
    for key, metrics_list in head_results.items():
        rates = [m["success_rate"] for m in metrics_list]
        avg_rate = sum(rates) / len(rates)
        print(f"Scenario {key}: rates={rates} -> Mean={avg_rate:.2f}%")
