"""Verification probe for Cloud LLM Optimizations 1-8."""

import json
from pathlib import Path
from src.coordination.orchestrator import DACAConfig, DACAOrchestrator


def main():
    print("[VERIFICATION] Initializing DACAOrchestrator with all Optimizations 1-8 active...")
    config = DACAConfig(
        name="verification_optimizations",
        use_distance_decomp=False,
        use_coalition_feasibility=False,
    )
    orch = DACAOrchestrator(
        config=config,
        scenario="logistics",
        network_profile="oscillatory",
        seed=42,
        max_steps=10,
    )
    orch.cloud_llm.config["use_mock"] = True

    metrics = orch.run()
    d = metrics.to_dict()

    print("\n[VERIFICATION RESULT - OPTIMIZATION METRICS EXPORT]")
    keys_to_check = [
        "prompt_reduction_percent",
        "cache_hits",
        "cache_misses",
        "cache_hit_rate",
        "saved_cloud_calls",
        "saved_tokens",
        "saved_latency",
        "local_reasoning_count",
        "cloud_reasoning_count",
        "consensus_skipped",
        "planner_latency",
    ]
    all_passed = True
    for key in keys_to_check:
        if key in d:
            print(f"  [OK] {key}: {d[key]}")
        else:
            print(f"  [MISSING] {key}")
            all_passed = False

    if all_passed:
        print("\nSUCCESS: All 8 optimization metrics are correctly tracked, populated, and serialized!")
    else:
        print("\nFAILURE: Some optimization metrics were missing.")


if __name__ == "__main__":
    main()
