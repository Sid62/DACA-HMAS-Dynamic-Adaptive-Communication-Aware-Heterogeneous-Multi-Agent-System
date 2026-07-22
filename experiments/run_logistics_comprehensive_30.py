#!/usr/bin/env python3
"""
Comprehensive Senior AI Research Experiment Runner for Logistics Scenario.
Evaluates:
  1. Standard Logistics Scenario (6 subtasks, 7 agents) at max_steps=30
  2. Scaled Logistics Scenario (30 subtasks, 30 agents [12 UAVs, 10 vehicles, 8 robots]) at max_steps=30
Across configs: B1, B2, A1, A2, A3, A4, A5
Across network profiles: stable, gradual, sudden, oscillatory
Across seeds: 0, 1, 2, 3, 4
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_thresholds
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.metrics.evaluation import MetricsCollector
from src.llm.exceptions import ExperimentFailed

CONFIGS_TO_RUN = ["B1", "B2", "A1", "A2", "A3", "A4", "A5"]
PROFILES = ["stable", "gradual", "sudden", "oscillatory"]
SEEDS = [0, 1, 2, 3, 4]
MAX_STEPS = 30


def run_experiment_suite(suite_name: str, custom_logistics_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f" STARTING SUITE: {suite_name} (Max Steps: {MAX_STEPS})")
    print(f"=======================================================\n")
    
    collector = MetricsCollector()
    total_runs = len(CONFIGS_TO_RUN) * len(PROFILES) * len(SEEDS)
    completed = 0
    results_by_cell: dict[str, list[dict[str, Any]]] = {}

    for cfg_name in CONFIGS_TO_RUN:
        for profile in PROFILES:
            cell_key = f"{cfg_name}_{profile}"
            results_by_cell[cell_key] = []
            
            for seed in SEEDS:
                completed += 1
                thresholds = get_thresholds()
                if custom_logistics_cfg:
                    thresholds.setdefault("scenarios", {}).setdefault("logistics", {}).update(custom_logistics_cfg)
                
                orch = DACAOrchestrator(
                    scenario="logistics",
                    network_profile=profile,
                    seed=seed,
                    config=CONFIGS[cfg_name],
                    thresholds=thresholds,
                    max_steps=MAX_STEPS,
                )
                
                start_t = time.perf_counter()
                try:
                    metrics = orch.run()
                except ExperimentFailed as e:
                    print(f"[{completed}/{total_runs}] [FAILED] {cfg_name} | {profile} | seed={seed}: {e}")
                    continue
                elapsed = time.perf_counter() - start_t
                
                metrics_dict = metrics.to_dict()
                metrics_dict["execution_wall_time_s"] = round(elapsed, 4)
                collector.records.append(metrics)
                results_by_cell[cell_key].append(metrics_dict)

    # Statistical Aggregation
    aggregated_summary: dict[str, Any] = {}
    for cell_key, runs in results_by_cell.items():
        if not runs:
            continue
        successes = [r["success_rate"] for r in runs]
        cfrs = [r["cfr"] for r in runs]
        switches = [r["switch_count"] for r in runs]
        tokens = [r["tokens"] for r in runs]
        comp_times = [r["computation_s"] for r in runs]
        
        cfg_name, profile = cell_key.split("_", 1)
        if cfg_name not in aggregated_summary:
            aggregated_summary[cfg_name] = {}
            
        aggregated_summary[cfg_name][profile] = {
            "num_seeds": len(runs),
            "success_rate_mean": round(float(np.mean(successes)), 2),
            "success_rate_std": round(float(np.std(successes)), 2),
            "cfr_mean": round(float(np.mean(cfrs)), 4),
            "cfr_std": round(float(np.std(cfrs)), 4),
            "switch_count_mean": round(float(np.mean(switches)), 2),
            "total_tokens_mean": round(float(np.mean(tokens)), 1),
            "computation_time_s_mean": round(float(np.mean(comp_times)), 4),
        }

    return {
        "suite_name": suite_name,
        "max_steps": MAX_STEPS,
        "raw_results": results_by_cell,
        "summary": aggregated_summary,
    }


def main() -> None:
    output_dir = ROOT / "experiments" / "results" / "logistics_research_30"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard Logistics Suite (6 subtasks, 7 agents, max_steps=30)
    std_suite = run_experiment_suite(
        suite_name="standard_logistics_max_steps_30",
        custom_logistics_cfg=None
    )
    
    with open(output_dir / "standard_logistics_max30.json", "w", encoding="utf-8") as f:
        json.dump(std_suite, f, indent=2)

    # 2. Scaled Logistics Suite (30 subtasks, 30 agents [12 UAVs, 10 vehicles, 8 robots], max_steps=30)
    scaled_cfg = {
        "num_subtasks": 30,
        "num_uav": 12,
        "num_vehicle": 10,
        "num_robot": 8,
    }
    scaled_suite = run_experiment_suite(
        suite_name="scaled_30_logistics_max_steps_30",
        custom_logistics_cfg=scaled_cfg
    )

    with open(output_dir / "scaled_logistics_max30.json", "w", encoding="utf-8") as f:
        json.dump(scaled_suite, f, indent=2)

    print(f"\n[RESEARCH RUN COMPLETE] All raw and summary results written to {output_dir}")


if __name__ == "__main__":
    main()
