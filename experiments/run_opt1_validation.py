"""Targeted runner for Optimization 1 validation (60 runs total: 30 B1, 30 A5)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.metrics.evaluation import MetricsCollector

CONFIGS_TO_RUN = ["B1", "A5"]
SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILES = ["stable", "oscillatory"]
SEEDS = 5
MAX_STEPS = 200

def main():
    collector = MetricsCollector()
    out_dir = ROOT / "experiments/results/opt1_cqi"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(CONFIGS_TO_RUN) * len(SCENARIOS) * len(PROFILES) * SEEDS
    done = 0
    
    print(f"Starting Optimization 1 Validation Sweep ({total} runs)...")
    
    for cfg_name in CONFIGS_TO_RUN:
        for scenario in SCENARIOS:
            for profile in PROFILES:
                for seed in range(SEEDS):
                    orch = DACAOrchestrator(
                        scenario=scenario,
                        network_profile=profile,
                        seed=seed,
                        config=CONFIGS[cfg_name],
                        max_steps=MAX_STEPS,
                    )
                    metrics = orch.run()
                    collector.records.append(metrics)
                    done += 1
                    print(
                        f"[{done:2d}/{total}] {cfg_name}/{scenario}/{profile}/s{seed+1} "
                        f"success={metrics.success_rate:.1f}% calls={metrics.total_api_calls} "
                        f"tokens={metrics.total_tokens} replans={metrics.replanning_count} "
                        f"switches={metrics.switch_count}"
                    )

    all_results = [r.to_dict() for r in collector.records]
    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    agg = MetricsCollector.aggregate_by_config(collector.records)
    with open(out_dir / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(f"\nValidation sweep complete! Saved {len(all_results)} runs to {out_dir / 'all_results.json'}")

if __name__ == "__main__":
    main()
