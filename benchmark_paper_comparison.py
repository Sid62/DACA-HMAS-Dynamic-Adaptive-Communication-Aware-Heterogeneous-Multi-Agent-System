#!/usr/bin/env python3
"""Comprehensive Empirical Benchmark & Paper Comparison.

Evaluates cloud_api_calls across 5 seeds (0, 1, 2, 3, 4) for:
  - logistics (Paper Table I: 4.23 API Calls)
  - inspection (Paper Table I: 4.85 API Calls)
  - search_rescue (Paper Table I: 3.41 API Calls)

Evaluates 4 configurations:
  1. A5_fresh: Hierarchical DACA-HMAS under cold-start (uncached, raw cloud LLM calls with selective replanning)
  2. A5_optimized: Full DACA-HMAS with semantic cache + experience store + plan continuity
  3. B1: Centralized Static Baseline (Paper counterpart central planner without adaptive offloading)
  4. B2: Decentralized Baseline (100% Device LLM, 0 Cloud calls)
"""

from __future__ import annotations

import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import DACAConfig, DACAOrchestrator

PAPER_REFERENCE = {
    "logistics": 4.23,
    "inspection": 4.85,
    "search_rescue": 3.41,
}

SEEDS = [0, 1, 2, 3, 4]
SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILES = ["stable", "oscillatory"]

CONFIGS_TO_TEST = {
    "A5_fresh": DACAConfig(
        name="A5_fresh",
        use_optimizations=True,
        experience_reuse=False,
        cache_responses=False,
        semantic_cache=False,
        plan_continuity=True,
    ),
    "A5_optimized": DACAConfig(
        name="A5_optimized",
        use_optimizations=True,
        experience_reuse=True,
        cache_responses=True,
        semantic_cache=True,
        plan_continuity=True,
    ),
    "B1": DACAConfig(
        name="B1",
        use_distance_decomp=False,
        use_coalition_feasibility=False,
        use_cqm=False,
        use_acds=False,
        use_handoff=False,
        use_reallocation=False,
        static_mode=0,
        use_optimizations=False,
    ),
    "B2": DACAConfig(
        name="B2",
        use_distance_decomp=False,
        use_coalition_feasibility=False,
        use_cqm=False,
        use_acds=False,
        use_handoff=False,
        use_reallocation=False,
        static_mode=1,
        use_optimizations=False,
    ),
}


def run_benchmark():
    all_records = []
    summary = {}

    print("=" * 90)
    print("STARTING DACA-HMAS vs. AutoHMA-LLM SCIENTIFIC EQUIVALENCE BENCHMARK")
    print(f"Seeds: {SEEDS}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Profiles: {PROFILES}")
    print(f"Configurations: {list(CONFIGS_TO_TEST.keys())}")
    print("=" * 90)

    for scenario in SCENARIOS:
        summary[scenario] = {}
        for cfg_name, cfg in CONFIGS_TO_TEST.items():
            summary[scenario][cfg_name] = {}
            for profile in PROFILES:
                results_per_seed = []
                for seed in SEEDS:
                    orch = DACAOrchestrator(
                        scenario=scenario,
                        network_profile=profile,
                        seed=seed,
                        config=cfg,
                        max_steps=120,
                    )
                    orch.cloud_llm.config["use_mock"] = True
                    for d_llm in orch.device_llms.values():
                        d_llm.config["use_mock"] = True

                    trap = io.StringIO()
                    with contextlib.redirect_stdout(trap):
                        metrics = orch.run()

                    usage = orch.cloud_llm.usage
                    record = {
                        "scenario": scenario,
                        "config": cfg_name,
                        "profile": profile,
                        "seed": seed,
                        "cloud_api_calls": metrics.cloud_api_calls,
                        "logical_llm_requests": metrics.logical_llm_requests,
                        "cloud_disk_cache_hits": usage.cloud_disk_cache_hits,
                        "semantic_cache_hits": usage.semantic_cache_hits,
                        "experience_store_hits": usage.experience_store_hits,
                        "plan_continuity_reuse": usage.plan_continuity_reuse,
                        "replanning_count": metrics.replanning_count,
                        "success_rate": metrics.success_rate,
                        "steps": metrics.steps,
                    }
                    all_records.append(record)
                    results_per_seed.append(record)
                    print(
                        f"[{scenario.upper():13s} | {cfg_name:12s} | {profile:11s} | Seed {seed}] "
                        f"Cloud Calls={record['cloud_api_calls']:2d} | "
                        f"Logical={record['logical_llm_requests']:2d} | "
                        f"Success={record['success_rate']*100:5.1f}% | "
                        f"Steps={record['steps']:3d}"
                    )

                calls = [r["cloud_api_calls"] for r in results_per_seed]
                logicals = [r["logical_llm_requests"] for r in results_per_seed]
                successes = [r["success_rate"] for r in results_per_seed]
                replan_counts = [r["replanning_count"] for r in results_per_seed]

                mean_calls = statistics.mean(calls)
                std_calls = statistics.stdev(calls) if len(calls) > 1 else 0.0
                median_calls = statistics.median(calls)
                min_calls = min(calls)
                max_calls = max(calls)

                ref_val = PAPER_REFERENCE.get(scenario, 4.0)
                reduction_pct = ((ref_val - mean_calls) / ref_val) * 100.0

                summary[scenario][cfg_name][profile] = {
                    "mean_cloud_api_calls": round(mean_calls, 3),
                    "std_cloud_api_calls": round(std_calls, 3),
                    "median_cloud_api_calls": median_calls,
                    "min_cloud_api_calls": min_calls,
                    "max_cloud_api_calls": max_calls,
                    "mean_logical_llm_requests": round(statistics.mean(logicals), 3),
                    "mean_replanning_count": round(statistics.mean(replan_counts), 3),
                    "mean_success_rate": round(statistics.mean(successes), 3),
                    "paper_reference_api_calls": ref_val,
                    "reduction_vs_paper_pct": round(reduction_pct, 2),
                }

    out_file = ROOT / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": all_records}, f, indent=2)

    print("\n" + "=" * 115)
    print("SCIENTIFIC BENCHMARK SUMMARY: DACA-HMAS vs. AutoHMA-LLM Table I")
    print("=" * 115)
    print(f"{'Scenario':14s} | {'Config':12s} | {'Profile':11s} | {'Cloud Calls (Mean±Std [Min-Max])':35s} | {'Paper Ref':10s} | {'Reduction %':11s} | {'Success Rate':12s}")
    print("-" * 125)
    for scenario in SCENARIOS:
        for cfg_name in CONFIGS_TO_TEST:
            for profile in PROFILES:
                s = summary[scenario][cfg_name][profile]
                call_str = f"{s['mean_cloud_api_calls']:.2f} ± {s['std_cloud_api_calls']:.2f} [{s['min_cloud_api_calls']}-{s['max_cloud_api_calls']}]"
                print(
                    f"{scenario:14s} | {cfg_name:12s} | {profile:11s} | {call_str:35s} | {s['paper_reference_api_calls']:10.2f} | {s['reduction_vs_paper_pct']:10.1f}% | {s['mean_success_rate']*100:10.1f}%"
                )
    print("=" * 125)


if __name__ == "__main__":
    run_benchmark()
