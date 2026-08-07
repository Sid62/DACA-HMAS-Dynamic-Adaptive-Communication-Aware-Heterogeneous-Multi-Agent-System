#!/usr/bin/env python3
"""Verification Script: Validate Fixed Optimized DACA-HMAS (A5) vs Original Baseline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator

SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILES = ["stable", "gradual", "sudden", "oscillatory"]
SEEDS = 3


def main():
    print("=========================================================")
    print(" EMPIRICAL VERIFICATION OF PLAN CONTINUITY FIX")
    print("=========================================================\n")

    out_dir = ROOT / "experiments/results/fix_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_configs = ["A5_unopt", "A5"]
    all_verification_records = []

    for cfg_name in test_configs:
        for sc in SCENARIOS:
            for prof in PROFILES:
                for seed in range(SEEDS):
                    orch = DACAOrchestrator(
                        scenario=sc,
                        network_profile=prof,
                        seed=seed,
                        config=CONFIGS[cfg_name],
                        max_steps=150,
                    )
                    metrics = orch.run()
                    res_dict = metrics.to_dict()
                    res_dict["cfg_name"] = cfg_name
                    res_dict["scenario"] = sc
                    res_dict["profile"] = prof
                    res_dict["seed"] = seed
                    all_verification_records.append(res_dict)

                    print(
                        f"[{cfg_name:8s} | {sc:13s} | {prof:11s} | s{seed}] "
                        f"Success: {metrics.success_rate*100 if metrics.success_rate<=1.0 else metrics.success_rate:5.1f}% "
                        f"| Steps: {metrics.steps:3d} | Tokens: {metrics.total_tokens:6d} "
                        f"| Calls: {metrics.total_api_calls:3d} | Comp: {metrics.computation_s:5.2f}s"
                    )

    out_file = out_dir / "verification_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_verification_records, f, indent=2)

    print(f"\nVerification raw data saved to {out_file}")


if __name__ == "__main__":
    main()
