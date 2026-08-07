#!/usr/bin/env python3
"""Empirical Proof & Verification Suite: Instrumenting DACA-HMAS to prove root cause hypotheses."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAConfig, DACAOrchestrator

ARTIFACT_DIR = Path(r"C:\Users\siddh\.gemini\antigravity-ide\brain\c23d3900-40ff-4e7d-9d6e-757044178297")


def run_ablation_matrix():
    scenarios = ["logistics", "inspection", "search_rescue"]
    profiles = ["oscillatory", "stable"]
    seeds = [0, 1, 2, 3, 4]

    configs = {
        "1_Original_A5_Baseline": {
            "use_optimizations": False,
            "use_acds": True,
            "use_coalition_feasibility": True,
            "use_hysteresis": True,
        },
        "2_Fixed_Optimized_A5": {
            "use_optimizations": True,
            "use_acds": True,
            "use_coalition_feasibility": True,
            "use_hysteresis": True,
        },
        "3_No_Switching_Fixed_Central": {
            "use_optimizations": True,
            "use_acds": False,
            "static_mode": 0,
            "use_coalition_feasibility": True,
            "use_hysteresis": True,
        },
        "4_No_Coalition_Adaptation": {
            "use_optimizations": True,
            "use_acds": True,
            "use_coalition_feasibility": False,
            "use_hysteresis": True,
        },
        "5_No_Hysteresis_Narrow_Band": {
            "use_optimizations": True,
            "use_acds": True,
            "use_coalition_feasibility": True,
            "use_hysteresis": False,
        },
    }

    results = {}

    print("=========================================================================")
    print(" RUNNING EMPIRICAL PROOF & ABLATION MATRIX (5 SEEDS x 3 SCENARIOS)")
    print("=========================================================================\n")

    for label, cfg_kwargs in configs.items():
        results[label] = []
        for sc in scenarios:
            for prof in profiles:
                for s in seeds:
                    cfg = DACAConfig(
                        name=label,
                        use_distance_decomp=True,
                        use_coalition_feasibility=cfg_kwargs.get("use_coalition_feasibility", True),
                        use_cqm=True,
                        use_acds=cfg_kwargs.get("use_acds", True),
                        use_handoff=True,
                        use_reallocation=True,
                        use_hysteresis=cfg_kwargs.get("use_hysteresis", True),
                        use_optimizations=cfg_kwargs.get("use_optimizations", True),
                        static_mode=cfg_kwargs.get("static_mode", None),
                    )

                    orch = DACAOrchestrator(
                        scenario=sc,
                        network_profile=prof,
                        seed=s,
                        config=cfg,
                        max_steps=150,
                    )
                    metrics = orch.run()
                    res = metrics.to_dict()
                    results[label].append(res)

                    succ = res["success_rate"]
                    print(
                        f"[{label:30s} | {sc:13s} | {prof:11s} | s{s}] "
                        f"Success: {succ:5.1f}% | Steps: {res['steps']:3d} | Switches: {res['switch_count']:2d} "
                        f"| PeerMsgs: {res['peer_messages']:3d} | Comp: {res['computation_s']:5.2f}s"
                    )

    out_dir = ROOT / "experiments" / "results" / "empirical_proof"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ablation_empirical_proof.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


    print(f"\nEmpirical Proof Ablation Data saved to: {out_file}")
    return results


if __name__ == "__main__":
    run_ablation_matrix()
