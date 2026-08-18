#!/usr/bin/env python3
"""Targeted A5 Evaluation Runner across Network Profiles & Scenarios."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.metrics.evaluation import MetricsCollector
from src.llm.exceptions import ExperimentFailed

SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILES = ["stable", "gradual", "sudden", "oscillatory"]
SEEDS = 5
MAX_STEPS = 150


def main() -> None:
    collector = MetricsCollector()
    out_dir = ROOT / "experiments/results/a5_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(SCENARIOS) * len(PROFILES) * SEEDS
    completed = 0
    start_time = time.time()

    print(f"===============================================================")
    print(f" TARGETED DACA-HMAS A5 EVALUATION ({total_runs} RUNS)")
    print(f" Scenarios : {SCENARIOS}")
    print(f" Profiles  : {PROFILES}")
    print(f" Seeds     : {SEEDS}")
    print(f"===============================================================\n")

    for scenario in SCENARIOS:
        for profile in PROFILES:
            for seed in range(SEEDS):
                completed += 1
                orch = DACAOrchestrator(
                    scenario=scenario,
                    network_profile=profile,
                    seed=seed,
                    config=CONFIGS["A5"],
                    max_steps=MAX_STEPS,
                )
                try:
                    metrics = orch.run()
                except ExperimentFailed as e:
                    print(f"[{completed:2d}/{total_runs}] [FAILED] A5/{scenario}/{profile}/s{seed}: {e}")
                    continue
                except Exception as e:
                    print(f"[{completed:2d}/{total_runs}] [ERROR] A5/{scenario}/{profile}/s{seed}: {e}")
                    continue

                collector.records.append(metrics)
                res_dict = metrics.to_dict()

                single_res_file = out_dir / f"A5_{scenario}_{profile}_s{seed}.json"
                try:
                    single_res_file.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")
                except Exception:
                    pass

                print(
                    f"[{completed:2d}/{total_runs}] A5 | {scenario:13s} | {profile:11s} | s{seed} "
                    f"| Success: {metrics.success_rate*100 if metrics.success_rate<=1.0 else metrics.success_rate:5.1f}% "
                    f"| Steps: {metrics.steps:3d} | Tokens: {metrics.total_tokens:6d} "
                    f"| Calls: {metrics.total_api_calls:3d} | Comp: {metrics.computation_s:5.2f}s"
                )

    all_results = [r.to_dict() for r in collector.records]
    all_results_file = out_dir / "a5_all_results.json"
    with open(all_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n===============================================================")
    print(f" A5 EVALUATION COMPLETE! ({completed}/{total_runs} runs in {total_time:.2f}s)")
    print(f" Saved to: {all_results_file}")
    print(f"===============================================================\n")


if __name__ == "__main__":
    main()
