#!/usr/bin/env python3
"""Analysis Script: Compare DACA-HMAS A5 (Best Profile) vs. AutoHMA-LLM Paper Baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# AutoHMA Paper Values from uploaded table image
AUTOHMA_PAPER = {
    "logistics": {
        "success_rate": 85.73,
        "steps": 5.11,
        "tokens": 152.87,
        "api_calls": 4.23,
        "memory_mb": 50.0,
        "computation_s": 8.5,
    },
    "inspection": {
        "success_rate": 85.67,
        "steps": 3.84,
        "tokens": 97.10,
        "api_calls": 2.13,
        "memory_mb": 40.0,
        "computation_s": 7.8,
    },
    "search_rescue": {
        "success_rate": 82.03,
        "steps": 4.30,
        "tokens": 166.69,
        "api_calls": 3.41,
        "memory_mb": 55.0,
        "computation_s": 9.2,
    },
}


def safe_pct(opt_val: float, base_val: float, lower_is_better: bool = True) -> float:
    if base_val == 0:
        return 0.0
    diff = (opt_val - base_val) / base_val * 100.0
    return -diff if lower_is_better else diff


def main():
    res_file = ROOT / "experiments/results/a5_eval/a5_all_results.json"
    if not res_file.exists():
        print(f"File {res_file} does not exist yet. Run run_a5_eval.py first.")
        return

    with open(res_file, encoding="utf-8") as f:
        runs = json.load(f)

    scenarios = sorted(list({r["scenario"] for r in runs}))
    profiles = ["stable", "gradual", "sudden", "oscillatory"]

    best_profiles = {}
    profile_analysis = {}

    for sc in scenarios:
        sc_runs = [r for r in runs if r["scenario"] == sc]
        prof_metrics = {}

        for prof in profiles:
            p_runs = [r for r in sc_runs if r["profile"] == prof]
            if not p_runs:
                continue
            
            # Extract metrics
            succ = np.mean([r.get("success_rate", 0) for r in p_runs])
            # normalize if success_rate is 0.0-1.0
            if succ <= 1.0:
                succ *= 100.0

            st = np.mean([r.get("steps", 0) for r in p_runs])
            tok = np.mean([r.get("tokens", 0) for r in p_runs])
            calls = np.mean([r.get("api_calls", 0) for r in p_runs])
            mem = np.mean([r.get("memory_mb", 0) for r in p_runs])
            comp = np.mean([r.get("computation_s", 0) for r in p_runs])

            prof_metrics[prof] = {
                "success_rate": float(succ),
                "steps": float(st),
                "tokens": float(tok),
                "api_calls": float(calls),
                "memory_mb": float(mem),
                "computation_s": float(comp),
            }

        # Select Best Profile: highest Success Rate; break ties with lower tokens / api / comp
        best_p = max(
            prof_metrics.keys(),
            key=lambda p: (
                prof_metrics[p]["success_rate"],
                -prof_metrics[p]["tokens"],
                -prof_metrics[p]["api_calls"],
                -prof_metrics[p]["computation_s"],
            )
        )
        best_profiles[sc] = best_p
        profile_analysis[sc] = prof_metrics

    # Generate Output Comparison Tables
    comparison_table = {}
    for sc in scenarios:
        bp = best_profiles[sc]
        a5_m = profile_analysis[sc][bp]
        auto_m = AUTOHMA_PAPER[sc]

        comp_rows = {}
        for metric_key in ["success_rate", "steps", "tokens", "api_calls", "memory_mb", "computation_s"]:
            a5_val = a5_m[metric_key]
            auto_val = auto_m[metric_key]
            lower_better = metric_key != "success_rate"
            diff = a5_val - auto_val
            pct = safe_pct(a5_val, auto_val, lower_is_better=lower_better)

            comp_rows[metric_key] = {
                "autohma": round(auto_val, 2),
                "optimized_a5": round(a5_val, 2),
                "difference": round(diff, 2),
                "pct_improvement": round(pct, 2),
            }

        comparison_table[sc] = comp_rows

    out_data = {
        "best_profiles": best_profiles,
        "profile_analysis": profile_analysis,
        "comparison_table": comparison_table,
    }

    out_json = ROOT / "experiments/results/a5_eval/a5_vs_autohma_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved comparison data to {out_json}")


if __name__ == "__main__":
    main()
