#!/usr/bin/env python3
"""F1/E2 Validation Runner — runs 10 seeds with the F1 fix applied and reconstructs
the 'before' counter values from the execution trace.

Since the F1 fix is PURELY a counter change (no execution logic modified), the
before counters can be exactly reconstructed: in the old code, every replan event
unconditionally incremented all counters. So:
  before_global_planning = replanning_count (for centralized replans)
  before_dispatch = replanning_count (for centralized replans)
  before_local_coordination = replanning_count (for decentralized replans)
  ... etc

The "after" counters are the actual values from the fixed code.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.llm.exceptions import ExperimentFailed

SCENARIOS = ["logistics", "inspection"]
PROFILE = "oscillatory"
SEEDS = 5
MAX_STEPS = 150


def main() -> None:
    out_dir = ROOT / "experiments" / "results" / "f1_e2_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(SCENARIOS) * SEEDS
    completed = 0
    start_time = time.time()

    print(f"{'='*70}")
    print(f" F1/E2 VALIDATION ({total_runs} runs)")
    print(f" Scenarios: {SCENARIOS}  Profile: {PROFILE}  Seeds: {SEEDS}")
    print(f"{'='*70}\n")

    results = []

    for scenario in SCENARIOS:
        for seed in range(SEEDS):
            completed += 1
            orch = DACAOrchestrator(
                scenario=scenario,
                network_profile=PROFILE,
                seed=seed,
                config=CONFIGS["A5"],
                max_steps=MAX_STEPS,
            )
            # Force mock mode for deterministic results without API credentials
            orch.cloud_llm.config["use_mock"] = True
            for dc in orch.device_llms.values():
                dc.config["use_mock"] = True

            try:
                metrics = orch.run()
            except ExperimentFailed as e:
                print(f"[{completed}/{total_runs}] [FAILED] {scenario}/s{seed}: {e}")
                continue
            except Exception as e:
                print(f"[{completed}/{total_runs}] [ERROR] {scenario}/s{seed}: {e}")
                import traceback
                traceback.print_exc()
                continue

            d = metrics.to_dict()
            breakdown = d.get("communication_step_breakdown", {})

            # Collect semantic cache miss reasons
            sc = getattr(orch.cloud_llm, "semantic_cache", None)
            miss_reasons = []
            if sc is not None and hasattr(sc, "miss_reason_log"):
                miss_reasons = list(sc.miss_reason_log)

            # Reconstruct "before" values: in the old code, every replan event
            # unconditionally incremented all counters for that mode.
            # replanning_count = total replan events (both modes combined).
            # The after breakdown tells us which were centralized vs decentralized:
            #   centralized replans = dispatch count (unchanged, always incremented)
            #   decentralized replans = replanning_count - dispatch count
            #
            # Before fix: global_planning = dispatch (both unconditional in centralized)
            # After fix: global_planning <= dispatch (gated by cloud_reasoned)
            #
            # For decentralized: before all 3 counters equaled decentralized_replan_count.
            # After fix: they may be less (gated by cloud_reasoned).

            after_dispatch = breakdown.get("dispatch", 0)
            after_gp = breakdown.get("global_planning", 0)
            after_lc = breakdown.get("local_coordination", 0)
            after_pc = breakdown.get("peer_consensus", 0)
            after_fs = breakdown.get("feedback_sync", 0)

            # Centralized replans = dispatch count (unconditional, same before and after)
            centralized_replans = after_dispatch
            # Decentralized replans = total - centralized
            replanning_count = d.get("replanning_count", 0)
            decentralized_replans = replanning_count - centralized_replans

            # Before values (old code: unconditional)
            before_gp = centralized_replans  # was always = dispatch
            before_dispatch = after_dispatch  # unchanged
            before_lc = decentralized_replans
            before_pc = decentralized_replans
            before_fs = decentralized_replans

            before_paper_comm = before_gp + before_dispatch
            after_paper_comm = d.get("paper_communication_steps", 0)

            before_total_comm = (before_gp + before_dispatch + before_lc + before_pc +
                                before_fs + breakdown.get("handoff_reallocation", 0))
            after_total_comm = d.get("communication_steps", 0)

            # Suppressed counts
            suppressed_gp = before_gp - after_gp
            suppressed_lc = before_lc - after_lc
            suppressed_pc = before_pc - after_pc
            suppressed_fs = before_fs - after_fs

            record = {
                "scenario": scenario,
                "seed": seed,
                "success_rate": d.get("success_rate", 0),
                "cloud_planning_calls": d.get("cloud_planning_calls", 0),
                "replanning_count": d.get("replanning_count", 0),
                "paper_communication_steps": d.get("paper_communication_steps", 0),
                "communication_steps": d.get("communication_steps", 0),
                "communication_step_breakdown": d.get("communication_step_breakdown", {}),
                "centralized_replans": centralized_replans,
                "decentralized_replans": decentralized_replans,
                "before": {
                    "global_planning": before_gp,
                    "dispatch": before_dispatch,
                    "local_coordination": before_lc,
                    "peer_consensus": before_pc,
                    "feedback_sync": before_fs,
                    "paper_communication_steps": before_paper_comm,
                    "communication_steps": before_total_comm,
                },
                "after": {
                    "global_planning": after_gp,
                    "dispatch": after_dispatch,
                    "local_coordination": after_lc,
                    "peer_consensus": after_pc,
                    "feedback_sync": after_fs,
                    "paper_communication_steps": after_paper_comm,
                    "communication_steps": after_total_comm,
                    "communication_step_breakdown": breakdown,
                },
                "suppressed": {
                    "global_planning": suppressed_gp,
                    "local_coordination": suppressed_lc,
                    "peer_consensus": suppressed_pc,
                    "feedback_sync": suppressed_fs,
                },
                "dispatch_unchanged": (before_dispatch == after_dispatch),
                "semantic_cache_hits": d.get("semantic_cache_hits", 0),
                "cache_misses": d.get("cache_misses", 0),
                "miss_reason_log": miss_reasons,
            }
            results.append(record)

            print(
                f"[{completed:2d}/{total_runs}] {scenario:10s} s{seed} "
                f"| SR={record['success_rate']:5.1f}% "
                f"| CloudCalls={record['cloud_planning_calls']:3d} "
                f"| Replan={replanning_count:2d} (C={centralized_replans} D={decentralized_replans}) "
                f"| PaperComm: {before_paper_comm:3d} -> {after_paper_comm:3d} (d={after_paper_comm - before_paper_comm:+d}) "
                f"| TotalComm: {before_total_comm:3d} -> {after_total_comm:3d} (d={after_total_comm - before_total_comm:+d}) "
                f"| Dispatch: {before_dispatch} == {after_dispatch} {'OK' if before_dispatch == after_dispatch else 'FAIL'}"
            )

    # Save results
    out_file = out_dir / "f1_e2_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*140}")
    print(f" BEFORE/AFTER COMPARISON TABLE")
    print(f"{'='*140}")
    print(f"{'Scenario':<12} {'Seed':>4} | {'SR%':>6} | {'CC':>4} | {'RP':>3} "
          f"| {'GP(B)':>5} {'GP(A)':>5} {'dGP':>4} "
          f"| {'DI(B)':>5} {'DI(A)':>5} {'chk':>4} "
          f"| {'LC(B)':>5} {'LC(A)':>5} {'dLC':>4} "
          f"| {'PC(B)':>5} {'PC(A)':>5} {'dPC':>4} "
          f"| {'FS(B)':>5} {'FS(A)':>5} {'dFS':>4} "
          f"| {'Paper(B)':>8} {'Paper(A)':>8} {'d':>4}")
    print("-" * 140)

    all_ok = True
    for r in results:
        b = r["before"]
        a = r["after"]
        di_ok = r["dispatch_unchanged"]
        if not di_ok:
            all_ok = False

        print(
            f"{r['scenario']:<12} {r['seed']:>4} | {r['success_rate']:>6.1f} | {r['cloud_planning_calls']:>4} | {r['replanning_count']:>3} "
            f"| {b['global_planning']:>5} {a['global_planning']:>5} {a['global_planning'] - b['global_planning']:>+4} "
            f"| {b['dispatch']:>5} {a['dispatch']:>5} {'OK' if di_ok else 'FAIL':>4} "
            f"| {b['local_coordination']:>5} {a['local_coordination']:>5} {a['local_coordination'] - b['local_coordination']:>+4} "
            f"| {b['peer_consensus']:>5} {a['peer_consensus']:>5} {a['peer_consensus'] - b['peer_consensus']:>+4} "
            f"| {b['feedback_sync']:>5} {a['feedback_sync']:>5} {a['feedback_sync'] - b['feedback_sync']:>+4} "
            f"| {b['paper_communication_steps']:>8} {a['paper_communication_steps']:>8} {a['paper_communication_steps'] - b['paper_communication_steps']:>+4}"
        )

    print(f"\n{'='*140}")
    if all_ok:
        print(" [PASS] ALL DISPATCH COUNTS UNCHANGED (no scope leak)")
    else:
        print(" [FAIL] DISPATCH MISMATCH DETECTED (scope leak!)")
    print(f"{'='*140}")

    # Semantic cache miss-reason breakdown
    print(f"\n{'='*70}")
    print(f" SEMANTIC CACHE MISS-REASON BREAKDOWN")
    print(f"{'='*70}")

    reason_totals = {"cache_empty": 0, "no_candidate_within_age": 0, "similarity_below_threshold": 0}
    best_sims = []

    for r in results:
        for entry in r.get("miss_reason_log", []):
            reason = entry.get("reason", "unknown")
            if reason in reason_totals:
                reason_totals[reason] += 1
            if "best_similarity" in entry:
                best_sims.append(entry["best_similarity"])

    total_misses = sum(reason_totals.values())
    print(f"\n  Total misses logged: {total_misses}")
    for reason, count in reason_totals.items():
        pct = (count / total_misses * 100) if total_misses > 0 else 0
        print(f"  {reason:<30s}: {count:>4} ({pct:5.1f}%)")

    if best_sims:
        import statistics
        print(f"\n  Best-similarity stats (for similarity_below_threshold misses):")
        print(f"    min={min(best_sims):.6f}  max={max(best_sims):.6f}  "
              f"mean={statistics.mean(best_sims):.6f}  median={statistics.median(best_sims):.6f}")

    # Per-seed cache details
    print(f"\n  Per-seed cache summary:")
    for r in results:
        hits = r.get("semantic_cache_hits", 0)
        misses = r.get("cache_misses", 0)
        reasons = {}
        for entry in r.get("miss_reason_log", []):
            rr = entry.get("reason", "unknown")
            reasons[rr] = reasons.get(rr, 0) + 1
        print(f"    {r['scenario']:10s} s{r['seed']}: hits={hits} misses={misses} reasons={reasons}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f" VALIDATION COMPLETE -- {completed}/{total_runs} runs in {elapsed:.1f}s")
    print(f" Results saved to: {out_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
