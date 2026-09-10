#!/usr/bin/env python3
"""Paired experiment: compare delta-dispatch optimization against baseline results.

Runs DACA-HMAS A5 config across all scenarios with oscillatory profile,
seeds 1-5, and compares against existing results in experiments/results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.llm.exceptions import ExperimentFailed

SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILE = "oscillatory"
SEEDS = range(1, 6)
CONFIG = "A5"


def run_experiment(scenario: str, seed: int) -> dict:
    """Run single experiment, return dict of key metrics."""
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile=PROFILE,
        seed=seed,
        config=CONFIGS[CONFIG],
        max_steps=300,
    )
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
    try:
        metrics = orch.run()
    except ExperimentFailed as e:
        print(f"  [FAILED] {scenario} seed={seed}: {e}")
        return {}
    d = metrics.to_dict()
    return {
        "success_rate": d.get("success_rate", 0),
        "paper_communication_steps": d.get("paper_communication_steps", 0),
        "dispatch": d.get("communication_step_breakdown", {}).get("dispatch", 0),
        "global_planning": d.get("communication_step_breakdown", {}).get("global_planning", 0),
        "dispatch_skipped_rounds": d.get("dispatch_skipped_rounds", 0),
        "cloud_planning_calls": d.get("cloud_planning_calls", 0),
        "cloud_network_calls": d.get("cloud_network_calls", 0),
        "cloud_total_tokens": d.get("cloud_total_tokens", 0),
        "device_total_tokens": d.get("device_total_tokens", 0),
        "switch_count": d.get("switch_count", 0),
        "replanning_count": d.get("replanning_count", 0),
        "steps": d.get("steps", 0),
        "computation_s": d.get("computation_s", 0),
    }


def load_baseline(scenario: str, seed: int) -> dict:
    """Load baseline result from existing experiments/results/ directory."""
    path = ROOT / "experiments" / "results" / f"A5_{scenario}_{PROFILE}_s{seed}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {
        "success_rate": d.get("success_rate", 0),
        "paper_communication_steps": d.get("paper_communication_steps", 0),
        "dispatch": d.get("communication_step_breakdown", {}).get("dispatch", 0),
        "global_planning": d.get("communication_step_breakdown", {}).get("global_planning", 0),
        "dispatch_skipped_rounds": d.get("dispatch_skipped_rounds", 0),
        "cloud_planning_calls": d.get("cloud_planning_calls", 0),
        "cloud_network_calls": d.get("cloud_network_calls", 0),
        "cloud_total_tokens": d.get("cloud_total_tokens", 0),
        "device_total_tokens": d.get("device_total_tokens", 0),
        "switch_count": d.get("switch_count", 0),
        "replanning_count": d.get("replanning_count", 0),
        "steps": d.get("steps", 0),
        "computation_s": d.get("computation_s", 0),
    }


def main():
    print("=" * 80)
    print("PAIRED EXPERIMENT: Delta Dispatch Optimization")
    print("=" * 80)

    all_results = []

    for scenario in SCENARIOS:
        for seed in SEEDS:
            print(f"\n--- {scenario} seed={seed} ---")
            baseline = load_baseline(scenario, seed)
            delta = run_experiment(scenario, seed)

            if not baseline or not delta:
                print(f"  [SKIP] Missing data for {scenario} seed={seed}")
                continue

            all_results.append({
                "scenario": scenario,
                "seed": seed,
                "baseline": baseline,
                "delta": delta,
            })

    # Print comparison table
    print("\n" + "=" * 120)
    print(f"{'Scenario':<18} {'Seed':>4} | {'Success (B/D)':>14} | {'PCS (B/D)':>10} | {'Dispatch (B/D)':>14} | {'Skipped':>7} | {'CloudAPI (B/D)':>14} | {'CloudTok (B/D)':>16}")
    print("-" * 120)

    total_b_pcs, total_d_pcs = 0, 0
    total_b_disp, total_d_disp = 0, 0
    total_skipped = 0
    n = 0

    for r in all_results:
        b, d = r["baseline"], r["delta"]
        print(
            f"{r['scenario']:<18} {r['seed']:>4} | "
            f"{b['success_rate']:>6.1f}/{d['success_rate']:<6.1f} | "
            f"{b['paper_communication_steps']:>4}/{d['paper_communication_steps']:<4} | "
            f"{b['dispatch']:>6}/{d['dispatch']:<6} | "
            f"{d['dispatch_skipped_rounds']:>7} | "
            f"{b['cloud_network_calls']:>6}/{d['cloud_network_calls']:<6} | "
            f"{b['cloud_total_tokens']:>7}/{d['cloud_total_tokens']:<7}"
        )
        total_b_pcs += b["paper_communication_steps"]
        total_d_pcs += d["paper_communication_steps"]
        total_b_disp += b["dispatch"]
        total_d_disp += d["dispatch"]
        total_skipped += d["dispatch_skipped_rounds"]
        n += 1

    if n > 0:
        print("-" * 120)
        print(f"{'AVERAGE':<18} {'':>4} | {'':>14} | {total_b_pcs/n:>4.1f}/{total_d_pcs/n:<4.1f} | {total_b_disp/n:>6.1f}/{total_d_disp/n:<6.1f} | {total_skipped/n:>7.1f} |")
        print(f"\nDelta Paper Comm Steps: {total_d_pcs/n - total_b_pcs/n:+.1f} ({(total_d_pcs-total_b_pcs)/max(total_b_pcs,1)*100:+.1f}%)")
        print(f"Delta Dispatch:         {total_d_disp/n - total_b_disp/n:+.1f} ({(total_d_disp-total_b_disp)/max(total_b_disp,1)*100:+.1f}%)")
        print(f"Total skipped:          {total_skipped}")

    # Save results
    out_path = ROOT / "experiments" / "results" / "delta_dispatch_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
