#!/usr/bin/env python3
"""Comprehensive Statistical Analysis and Report Generator for DACA-HMAS Validation Sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_results(results_path: Path) -> list[dict[str, Any]]:
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def safe_pct(new_val: float, old_val: float, lower_is_better: bool = True) -> float:
    if old_val == 0:
        return 0.0
    diff = (new_val - old_val) / old_val * 100.0
    return -diff if lower_is_better else diff


def generate_full_analysis(results: list[dict[str, Any]]) -> dict[str, Any]:
    configs = sorted(list({r["config"] for r in results}))
    scenarios = sorted(list({r["scenario"] for r in results}))
    profiles = sorted(list({r["profile"] for r in results}))

    # Aggregate by Config
    by_config: dict[str, dict[str, Any]] = {}
    metrics_keys = [
        "success_rate", "steps", "tokens", "api_calls", "memory_mb",
        "computation_s", "switch_count", "peer_messages", "broadcast_count",
        "consensus_rounds", "consensus_latency", "cloud_planning_calls",
        "device_planning_calls", "replanning_count", "local_reallocation_count",
        "cached_plan_reuse_count", "merged_singleton_count", "avg_planning_latency",
        "coalition_change_count", "tfr", "cfr"
    ]

    for cfg in configs:
        cfg_runs = [r for r in results if r["config"] == cfg]
        by_config[cfg] = {"count": len(cfg_runs)}
        for k in metrics_keys:
            vals = [r.get(k, 0) for r in cfg_runs if k in r]
            if vals:
                by_config[cfg][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }

    # Comparison Matrix: AutoHMA (B1/B2 avg) vs Previous DACA (A5_unopt) vs Optimized DACA (A5)
    autohma_b1 = by_config.get("B1", {})
    autohma_b2 = by_config.get("B2", {})
    prev_daca = by_config.get("A5_unopt", {})
    opt_daca = by_config.get("A5", {})

    base_comparison = []
    base_metrics = ["success_rate", "steps", "tokens", "api_calls", "memory_mb", "computation_s"]
    metric_labels = {
        "success_rate": "Mission Success (%)",
        "steps": "Steps to Finish",
        "tokens": "Total LLM Tokens",
        "api_calls": "Total API Calls",
        "memory_mb": "Memory Usage (MB)",
        "computation_s": "Computation Time (s)",
    }

    for m in base_metrics:
        b1_val = autohma_b1.get(m, {}).get("mean", 0.0)
        b2_val = autohma_b2.get(m, {}).get("mean", 0.0)
        auto_val = (b1_val + b2_val) / 2.0 if (b1_val and b2_val) else (b1_val or b2_val)
        prev_val = prev_daca.get(m, {}).get("mean", 0.0)
        opt_val = opt_daca.get(m, {}).get("mean", 0.0)

        lower_better = m != "success_rate"
        diff_prev = opt_val - prev_val
        pct_imp_prev = safe_pct(opt_val, prev_val, lower_is_better=lower_better)
        pct_imp_auto = safe_pct(opt_val, auto_val, lower_is_better=lower_better)

        base_comparison.append({
            "metric": metric_labels[m],
            "autohma": round(auto_val, 2),
            "previous_daca": round(prev_val, 2),
            "optimized_daca": round(opt_val, 2),
            "difference_vs_prev": round(diff_prev, 2),
            "pct_imp_vs_prev": round(pct_imp_prev, 2),
            "pct_imp_vs_autohma": round(pct_imp_auto, 2),
        })

    # Validation Modules Breakdown
    # 1. Plan Continuity
    switches_opt = opt_daca.get("switch_count", {}).get("mean", 0.0)
    plan_reuse_opt = opt_daca.get("cached_plan_reuse_count", {}).get("mean", 0.0)
    replans_prev = prev_daca.get("replanning_count", {}).get("mean", 0.0)
    replans_opt = opt_daca.get("replanning_count", {}).get("mean", 0.0)
    avoided_calls_continuity = max(0.0, replans_prev - replans_opt)
    plan_reuse_pct = (plan_reuse_opt / (plan_reuse_opt + replans_opt) * 100.0) if (plan_reuse_opt + replans_opt) > 0 else 0.0

    plan_continuity_stats = {
        "switch_count": round(switches_opt, 2),
        "cached_plan_reuse_count": round(plan_reuse_opt, 2),
        "plan_reuse_percentage": round(plan_reuse_pct, 2),
        "replans_unoptimized": round(replans_prev, 2),
        "replans_optimized": round(replans_opt, 2),
        "avoided_llm_calls": round(avoided_calls_continuity, 2),
    }

    # 2. Consensus & Communication
    consensus_rounds_prev = prev_daca.get("consensus_rounds", {}).get("mean", 0.0)
    consensus_rounds_opt = opt_daca.get("consensus_rounds", {}).get("mean", 0.0)
    consensus_lat_prev = prev_daca.get("consensus_latency", {}).get("mean", 0.0)
    consensus_lat_opt = opt_daca.get("consensus_latency", {}).get("mean", 0.0)
    peer_msgs_prev = prev_daca.get("peer_messages", {}).get("mean", 0.0)
    peer_msgs_opt = opt_daca.get("peer_messages", {}).get("mean", 0.0)

    consensus_stats = {
        "consensus_rounds_prev": round(consensus_rounds_prev, 2),
        "consensus_rounds_opt": round(consensus_rounds_opt, 2),
        "consensus_latency_prev": round(consensus_lat_prev, 4),
        "consensus_latency_opt": round(consensus_lat_opt, 4),
        "latency_reduction_pct": round(safe_pct(consensus_lat_opt, consensus_lat_prev, lower_is_better=True), 2),
        "peer_messages_prev": round(peer_msgs_prev, 2),
        "peer_messages_opt": round(peer_msgs_opt, 2),
        "peer_message_reduction_pct": round(safe_pct(peer_msgs_opt, peer_msgs_prev, lower_is_better=True), 2),
    }

    # 3. Best / Worst Case Analysis per Scenario
    best_worst_by_scenario = {}
    for sc in scenarios:
        sc_runs = [r for r in results if r["scenario"] == sc]
        sc_cfgs = sorted(list({r["config"] for r in sc_runs}))

        cfg_perf = {}
        for c in sc_cfgs:
            c_items = [r for r in sc_runs if r["config"] == c]
            succ = np.mean([r["success_rate"] for r in c_items])
            tok = np.mean([r["tokens"] for r in c_items])
            calls = np.mean([r["api_calls"] for r in c_items])
            comp = np.mean([r["computation_s"] for r in c_items])
            cfg_perf[c] = {
                "success_rate": float(succ),
                "tokens": float(tok),
                "api_calls": float(calls),
                "computation_s": float(comp),
            }

        highest_succ = max(cfg_perf.items(), key=lambda x: x[1]["success_rate"])[0]
        lowest_tok = min(cfg_perf.items(), key=lambda x: x[1]["tokens"])[0]
        lowest_api = min(cfg_perf.items(), key=lambda x: x[1]["api_calls"])[0]
        lowest_comp = min(cfg_perf.items(), key=lambda x: x[1]["computation_s"])[0]

        best_worst_by_scenario[sc] = {
            "highest_success": highest_succ,
            "highest_success_val": round(cfg_perf[highest_succ]["success_rate"], 2),
            "lowest_token": lowest_tok,
            "lowest_token_val": round(cfg_perf[lowest_tok]["tokens"], 2),
            "lowest_api": lowest_api,
            "lowest_api_val": round(cfg_perf[lowest_api]["api_calls"], 2),
            "lowest_computation": lowest_comp,
            "lowest_computation_val": round(cfg_perf[lowest_comp]["computation_s"], 3),
            "best_overall": "A5" if "A5" in cfg_perf else sc_cfgs[-1],
            "worst_overall": "B1" if "B1" in cfg_perf else sc_cfgs[0],
        }

    return {
        "by_config": by_config,
        "base_comparison": base_comparison,
        "plan_continuity_stats": plan_continuity_stats,
        "consensus_stats": consensus_stats,
        "best_worst_by_scenario": best_worst_by_scenario,
    }


def main():
    res_dir = ROOT / "experiments/results/val_sweep"
    all_res_file = res_dir / "all_results.json"

    if not all_res_file.exists():
        print(f"Results file {all_res_file} not found! Waiting or checking for alternate output directory...")
        return

    results = load_results(all_res_file)
    analysis = generate_full_analysis(results)

    analysis_out = res_dir / "comprehensive_report.json"
    with open(analysis_out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n===============================================================")
    print(f" STATISTICAL REPORT GENERATED: {analysis_out}")
    print(f" Configs Analyzed: {list(analysis['by_config'].keys())}")
    print(f"===============================================================\n")


if __name__ == "__main__":
    main()
