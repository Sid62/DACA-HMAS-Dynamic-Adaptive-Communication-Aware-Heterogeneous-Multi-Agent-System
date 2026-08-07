#!/usr/bin/env python3
"""Controlled Ablation Debugger: Isolate which optimization causes mission timeout (steps=150)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import DACAConfig, DACAOrchestrator


def run_single_config(config_name: str, opt_flags: dict, scenario: str = "logistics", seeds: int = 3) -> dict:
    results = []
    for seed in range(seeds):
        cfg = DACAConfig(
            name=config_name,
            use_distance_decomp=opt_flags.get("use_distance_decomp", True),
            use_coalition_feasibility=opt_flags.get("use_coalition_feasibility", True),
            use_cqm=opt_flags.get("use_cqm", True),
            use_acds=opt_flags.get("use_acds", True),
            use_handoff=opt_flags.get("use_handoff", True),
            use_reallocation=opt_flags.get("use_reallocation", True),
            use_hysteresis=opt_flags.get("use_hysteresis", True),
            use_optimizations=opt_flags.get("use_optimizations", True),
        )
        orch = DACAOrchestrator(
            scenario=scenario,
            network_profile="stable",
            seed=seed,
            config=cfg,
            max_steps=150,
        )

        # Apply specific module overrides
        if not opt_flags.get("use_plan_continuity", True):
            orch.continuity_engine = None
            orch.centralized.continuity_engine = None
            orch.decentralized.continuity_engine = None

        if not opt_flags.get("use_cache", True):
            orch.cloud_llm.config["cache_responses"] = False
            for d in orch.device_llms.values():
                d.config["cache_responses"] = False

        if not opt_flags.get("use_delta_transfer", True):
            orch.delta_transfer_manager = None

        if not opt_flags.get("use_plan_repair", True):
            orch.plan_repairer = None

        metrics = orch.run()
        results.append(metrics.to_dict())

    succ = [r["success_rate"] for r in results]
    steps = [r["steps"] for r in results]
    tokens = [r["tokens"] for r in results]
    calls = [r["api_calls"] for r in results]
    comp = [r["computation_s"] for r in results]
    timeouts = sum(1 for r in results if r["steps"] >= 150)

    return {
        "config": config_name,
        "avg_success": float(sum(succ) / len(succ)),
        "avg_steps": float(sum(steps) / len(steps)),
        "avg_tokens": float(sum(tokens) / len(tokens)),
        "avg_api_calls": float(sum(calls) / len(calls)),
        "avg_comp_s": float(sum(comp) / len(comp)),
        "timeouts": timeouts,
        "total_runs": len(results),
    }


def main():
    print("=========================================================")
    print(" RUNNING CONTROLLED ABLATION DEBUGGER")
    print("=========================================================\n")

    test_matrix = {
        "1_Original_A5_Unoptimized": {
            "use_optimizations": False,
            "use_plan_continuity": False,
            "use_cache": False,
            "use_delta_transfer": False,
            "use_plan_repair": False,
        },
        "2_Plus_Plan_Continuity_Only": {
            "use_optimizations": True,
            "use_plan_continuity": True,
            "use_cache": False,
            "use_delta_transfer": False,
            "use_plan_repair": False,
        },
        "3_Plus_Prompt_Cache_Only": {
            "use_optimizations": True,
            "use_plan_continuity": False,
            "use_cache": True,
            "use_delta_transfer": False,
            "use_plan_repair": False,
        },
        "4_Plus_Delta_Transfer_Only": {
            "use_optimizations": True,
            "use_plan_continuity": False,
            "use_cache": False,
            "use_delta_transfer": True,
            "use_plan_repair": False,
        },
        "5_Plus_Plan_Repair_Only": {
            "use_optimizations": True,
            "use_plan_continuity": False,
            "use_cache": False,
            "use_delta_transfer": False,
            "use_plan_repair": True,
        },
        "6_Full_Optimized_A5": {
            "use_optimizations": True,
            "use_plan_continuity": True,
            "use_cache": True,
            "use_delta_transfer": True,
            "use_plan_repair": True,
        },
    }

    ablation_summary = []
    for label, flags in test_matrix.items():
        res = run_single_config(label, flags, scenario="logistics", seeds=3)
        ablation_summary.append(res)
        print(
            f"[{label:30s}] Success: {res['avg_success']:5.1f}% | Steps: {res['avg_steps']:5.1f} "
            f"| Timeouts: {res['timeouts']}/{res['total_runs']} | Calls: {res['avg_api_calls']:4.1f} | Comp: {res['avg_comp_s']:5.2f}s"
        )

    out_file = ROOT / "experiments/results/ablation_debug_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    print(f"\nAblation debug results saved to {out_file}")


if __name__ == "__main__":
    main()
