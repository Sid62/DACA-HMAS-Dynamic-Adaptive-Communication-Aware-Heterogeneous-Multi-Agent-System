"""Run 30 A5 validation experiments for Optimization 1."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator

SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILES = ["stable", "oscillatory"]
SEEDS = 5
MAX_STEPS = 200

def main():
    out_dir = ROOT / "experiments/results/opt1_cqi"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    total = len(SCENARIOS) * len(PROFILES) * SEEDS
    done = 0
    
    print(f"Running 30 A5 Validation Experiments for Optimization #1...")
    
    for scenario in SCENARIOS:
        for profile in PROFILES:
            for seed in range(SEEDS):
                orch = DACAOrchestrator(
                    scenario=scenario,
                    network_profile=profile,
                    seed=seed,
                    config=CONFIGS["A5"],
                    max_steps=MAX_STEPS,
                )
                metrics = orch.run()
                d = metrics.to_dict()
                records.append(d)
                done += 1
                print(
                    f"[{done:2d}/{total}] A5/{scenario}/{profile}/s{seed+1} "
                    f"success={metrics.success_rate:.1f}% calls={metrics.total_api_calls} "
                    f"tokens={metrics.total_tokens} replans={metrics.replanning_count} "
                    f"switches={metrics.switch_count}"
                )

    with open(out_dir / "a5_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nA5 Validation complete! Saved {len(records)} runs to {out_dir / 'a5_results.json'}")

if __name__ == "__main__":
    main()
