#!/usr/bin/env python3
"""Full experimental sweep: B1/B2, A1-A5 across scenarios, profiles, seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from src.llm.exceptions import ExperimentFailed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.metrics.evaluation import MetricsCollector


CONFIGS_TO_RUN = ["B1", "B2", "A1", "A2", "A3", "A4", "A5"]
SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILES = ["stable", "gradual", "sudden", "oscillatory"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full DACA-HMAS sweep")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Seeds per cell (use 10+ for paper)")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output-dir", default="experiments/results/full_sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced matrix for CI/dev")
    args = parser.parse_args()

    configs = ["B1", "B2", "A5"] if args.quick else CONFIGS_TO_RUN
    scenarios = ["logistics", "inspection"] if args.quick else SCENARIOS
    profiles = ["stable", "gradual"] if args.quick else PROFILES
    seeds = min(args.seeds, 2) if args.quick else args.seeds

    collector = MetricsCollector()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(configs) * len(scenarios) * len(profiles) * seeds
    done = 0

    for cfg_name in configs:
        for scenario in scenarios:
            for profile in profiles:
                for seed in range(seeds):
                    orch = DACAOrchestrator(
                        scenario=scenario,
                        network_profile=profile,
                        seed=seed,
                        config=CONFIGS[cfg_name],
                        max_steps=args.max_steps,
                    )
                    try:
                       metrics = orch.run()
                    except ExperimentFailed as e:
                       print(
                           f"[FAILED] "
                           f"{cfg_name}/{scenario}/{profile}/s{seed}: {e}"
                       )
                       continue

                    collector.records.append(metrics)
                    done += 1
                    print(
                        f"[{done}/{total}] {cfg_name}/{scenario}/{profile}/s{seed} "
                        f"success={metrics.success_rate:.1%} CFR={metrics.cfr:.3f} SC={metrics.switch_count}"
                    )

    all_results = [r.to_dict() for r in collector.records]
    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    agg = MetricsCollector.aggregate_by_config(collector.records)
    with open(out_dir / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    b1_success = [r.success_rate for r in collector.records
                  if r.config_name == "B1" and r.scenario == "logistics"]
    b2_success = [r.success_rate for r in collector.records
                  if r.config_name == "B2" and r.scenario == "inspection"]
    if b1_success and b2_success:
        sig = MetricsCollector.significance_test(b1_success, b2_success)
        with open(out_dir / "significance.json", "w", encoding="utf-8") as f:
            json.dump(sig, f, indent=2)

    print(f"\nSweep complete: {done} runs, results in {out_dir}")


if __name__ == "__main__":
    main()
