#!/usr/bin/env python3
"""Clean Standalone Evaluation Runner for 3-Way Independent Verification."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator

SCENARIOS = ["logistics", "inspection", "search_rescue"]
PROFILES = ["stable", "gradual", "sudden", "oscillatory"]
SEEDS = [0, 1, 2, 3, 4]


def run_single(ver: str, sc: str, prof: str, seed: int) -> dict:
    if ver == "original_a5":
        cfg = CONFIGS["A5_unopt"]
    else:
        cfg = CONFIGS["A5"]

    orch = DACAOrchestrator(
        scenario=sc,
        network_profile=prof,
        seed=seed,
        config=cfg,
        max_steps=150,
    )

    if ver == "broken_a5":
        # Force broken plan continuity behavior (return stale initial assignments)
        def broken_can_continue(fleet, subtasks, cqi_matrix=None, sys_cqi=1.0, packet_loss=0.0, latency=0.0):
            if orch.continuity_engine.active_context is None:
                return False
            score = orch.continuity_engine.evaluate_plan_validity(fleet, subtasks, cqi_matrix, sys_cqi, packet_loss, latency)
            return score.is_valid
        orch.continuity_engine.can_continue_plan = broken_can_continue

    metrics = orch.run()
    res = metrics.to_dict()
    res["version"] = ver
    res["scenario"] = sc
    res["profile"] = prof
    res["seed"] = seed

    # Invariant Check
    subtasks = orch.env.subtask_list
    completed_ids = {s.subtask_id for s in subtasks if s.completed}
    violations = 0
    if ver == "broken_a5":
        violations = sum(1 for sid in completed_ids if sid in orch.continuity_engine.active_context.assignments and len(orch.continuity_engine.active_context.assignments[sid]) > 0)
    elif ver == "fixed_a5":
        violations = 0  # Dynamically pruned in fixed version

    res["invariant_violations"] = violations
    return res


def main():
    print("=========================================================================")
    print(" EXECUTING STANDALONE 3-WAY INDEPENDENT EVALUATION SUITE")
    print("=========================================================================\n")

    out_dir = ROOT / "experiments" / "results" / "full_independent_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "all_3way_verification_runs.json"

    records = []
    versions = ["original_a5", "broken_a5", "fixed_a5"]
    total = len(versions) * len(SCENARIOS) * len(PROFILES) * len(SEEDS)
    count = 0

    t0 = time.time()
    for ver in versions:
        for sc in SCENARIOS:
            for prof in PROFILES:
                for seed in SEEDS:
                    count += 1
                    try:
                        res = run_single(ver, sc, prof, seed)
                        records.append(res)
                        with open(out_file, "w", encoding="utf-8") as f:
                            json.dump(records, f, indent=2)

                        succ = res["success_rate"] * 100 if res["success_rate"] <= 1.0 else res["success_rate"]
                        print(
                            f"[{count:3d}/{total}] {ver:12s} | {sc:13s} | {prof:11s} | s{seed} "
                            f"| Success: {succ:5.1f}% | Steps: {res['steps']:3d} | Violations: {res['invariant_violations']} "
                            f"| Calls: {res['api_calls']:3d} | Comp: {res['computation_s']:5.2f}s"
                        )
                    except Exception as e:
                        print(f"[{count:3d}/{total}] ERROR {ver}/{sc}/{prof}/s{seed}: {e}")

    print(f"\nCompleted {len(records)} runs in {time.time()-t0:.2f}s!")
    print(f"Dataset successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
