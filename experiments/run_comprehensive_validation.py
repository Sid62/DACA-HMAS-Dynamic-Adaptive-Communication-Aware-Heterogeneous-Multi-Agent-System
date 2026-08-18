#!/usr/bin/env python3
"""Comprehensive Validation Sweep Runner for DACA-HMAS.

Executes:
- Baselines: B1, B2 (AutoHMA-LLM)
- Ablations: A1, A2, A3, A4
- Previous DACA-HMAS: A5_unopt (all research novelties, unoptimized cost/latency)
- Optimized DACA-HMAS: A5 (all research novelties + cost/latency optimizations)

Across:
- Scenarios: logistics, inspection, search_rescue
- Profiles: stable, gradual, sudden, oscillatory
- Seeds: 0, 1, 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.metrics.evaluation import MetricsCollector
from src.llm.exceptions import ExperimentFailed

CONFIGS_TO_RUN = ["B1", "B2", "A1", "A2", "A3", "A4", "A5_unopt", "A5"]
SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILES = ["stable", "gradual", "sudden", "oscillatory"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full DACA-HMAS Validation Sweep")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per cell (default: 3)")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output-dir", default="experiments/results/val_sweep")

    parser.add_argument("--quick", action="store_true", help="Run reduced matrix for fast verification")
    args = parser.parse_args()

    if args.quick:
        configs = ["B1", "B2", "A5_unopt", "A5"]
        scenarios = ["logistics", "inspection"]
        profiles = ["stable", "oscillatory"]
        seeds = 1
    else:
        configs = CONFIGS_TO_RUN
        scenarios = SCENARIOS
        profiles = PROFILES
        seeds = args.seeds

    collector = MetricsCollector()
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)


    total_runs = len(configs) * len(scenarios) * len(profiles) * seeds
    completed = 0
    start_time = time.time()

    print(f"===============================================================")
    print(f" STARTING COMPREHENSIVE EXPERIMENTAL VALIDATION MATRIX ({total_runs} RUNS)")
    print(f" Configurations : {configs}")
    print(f" Scenarios      : {scenarios}")
    print(f" Profiles       : {profiles}")
    print(f" Seeds          : {seeds}")
    print(f"===============================================================\n")

    for cfg_name in configs:
        for scenario in scenarios:
            for profile in profiles:
                for seed in range(seeds):
                    completed += 1
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
                        print(f"[{completed}/{total_runs}] [FAILED] {cfg_name}/{scenario}/{profile}/s{seed}: {e}")
                        continue
                    except Exception as e:
                        print(f"[{completed}/{total_runs}] [ERROR] {cfg_name}/{scenario}/{profile}/s{seed}: {e}")
                        continue

                    collector.records.append(metrics)
                    res_dict = metrics.to_dict()

                    # Enhance dict with unoptimized vs optimized tag
                    res_dict["is_optimized"] = (cfg_name == "A5")
                    res_dict["is_unoptimized_daca"] = (cfg_name == "A5_unopt")

                    try:
                        single_res_dir = Path(out_dir)
                        single_res_dir.mkdir(parents=True, exist_ok=True)
                        single_res_file = single_res_dir / f"{cfg_name}_{scenario}_{profile}_s{seed}.json"
                        single_res_file.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")
                    except Exception:
                        pass




                    print(
                        f"[{completed:3d}/{total_runs}] {cfg_name:8s} | {scenario:13s} | {profile:11s} | s{seed} "
                        f"| Success: {metrics.success_rate:6.1f}% | Steps: {metrics.steps:3d} "
                        f"| Tokens: {metrics.total_tokens:7d} | Calls: {metrics.total_api_calls:4d} "
                        f"| Latency: {metrics.avg_planning_latency:6.3f}s | Switches: {metrics.switch_count:2d}"
                    )

    all_results = [r.to_dict() for r in collector.records]
    all_results_file = out_dir / "all_results.json"
    with open(all_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    agg = MetricsCollector.aggregate_by_config(collector.records)
    summary_file = out_dir / "aggregate_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n===============================================================")
    print(f" VALIDATION SWEEP COMPLETE! ({completed}/{total_runs} runs)")
    print(f" Total Elapsed Time: {total_time:.2f} seconds")
    print(f" All raw results saved to : {all_results_file}")
    print(f" Aggregate summary saved to: {summary_file}")
    print(f"===============================================================\n")


if __name__ == "__main__":
    main()
