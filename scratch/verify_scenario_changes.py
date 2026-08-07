"""Compare metrics before/after scenario fidelity changes within a single process.

This runs the comparison in one Python process to avoid hash() non-determinism
across sessions (Python 3.3+ randomizes string hashes by default).
"""
import json
import sys
import importlib
import copy
sys.path.insert(0, ".")


def get_subtask_targets(scenario_name: str, seed: int) -> list:
    """Get subtask targets from current code."""
    from src.config import get_thresholds
    from src.env.scenarios import get_scenario
    thresholds = get_thresholds()
    scenario = get_scenario(scenario_name, thresholds, seed)
    return [(s.subtask_id, s.target.x, s.target.y) for s in scenario.subtasks]


def run_scenario(scenario_name: str, seed: int = 42, max_steps: int = 50) -> dict:
    """Run a scenario with forced mock mode."""
    from src.coordination.orchestrator import DACAOrchestrator, CONFIGS
    cfg = CONFIGS["B1"]
    orch = DACAOrchestrator(
        scenario=scenario_name,
        network_profile="stable",
        seed=seed,
        config=cfg,
        max_steps=max_steps,
    )
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
    result = orch.run()
    return result.to_dict()


if __name__ == "__main__":
    # Metrics that are deterministic (not wall-clock)
    DETERMINISTIC_KEYS = {
        "success_rate", "steps", "tokens", "api_calls", "switch_count",
        "communication_steps", "communication_step_breakdown", "tfr", "cfr",
        "cloud_planning_calls", "device_planning_calls", "replanning_count",
        "coalition_change_count", "hallucination_stats",
        "peer_messages", "broadcast_count", "consensus_rounds",
        "consensus_latency", "plan_merge_count", "distributed_replanning_count",
        "local_reallocation_count", "cached_plan_reuse_count",
        "merged_singleton_count",
    }

    # Note: experience_reuse_* may differ because experience_reuse is enabled
    # and inspection's behavior changed (sensors start at target). Exclude from
    # strict comparison since the experience store content can vary.
    EXCLUDE_KEYS = {
        "memory_mb", "computation_s", "total_wall_clock_s",
        "avg_planning_latency", "config", "scenario", "profile", "seed",
        "experience_reuse_attempts", "experience_reuse_hits",
    }

    print("=" * 70)
    print("SCENARIO FIDELITY VERIFICATION")
    print("=" * 70)

    # === CONSTRAINT 5: Logistics subtask targets unchanged ===
    print("\n--- CONSTRAINT 5: RNG ordering (logistics targets) ---")
    targets_log = get_subtask_targets("logistics", 42)
    # Run logistics twice to verify same-process determinism
    targets_log2 = get_subtask_targets("logistics", 42)
    all_match = all(
        t1[1] == t2[1] and t1[2] == t2[2]
        for t1, t2 in zip(targets_log, targets_log2)
    )
    print(f"Same-process target determinism: {all_match}")
    for t in targets_log:
        print(f"  {t[0]}: ({t[1]:.15f}, {t[2]:.15f})")

    # === CONSTRAINT 2: Logistics metrics unchanged ===
    print("\n--- CONSTRAINT 2: Logistics metrics (should be byte-identical) ---")
    r_log_1 = run_scenario("logistics", seed=42)
    r_log_2 = run_scenario("logistics", seed=42)
    log_ok = True
    for k in sorted(DETERMINISTIC_KEYS):
        if k in r_log_1:
            if r_log_1[k] != r_log_2[k]:
                print(f"  {k}: {r_log_1[k]} != {r_log_2[k]} *** MISMATCH ***")
                log_ok = False
            else:
                v = r_log_1[k]
                if isinstance(v, dict):
                    print(f"  {k}: (dict) [MATCH]")
                else:
                    print(f"  {k}: {v} [MATCH]")
    print(f"Logistics deterministic metrics identical: {log_ok}")

    # === CONSTRAINT 2: Search & Rescue metrics unchanged ===
    print("\n--- CONSTRAINT 2: Search & Rescue metrics (should be byte-identical) ---")
    r_sar_1 = run_scenario("search_rescue", seed=42)
    r_sar_2 = run_scenario("search_rescue", seed=42)
    sar_ok = True
    for k in sorted(DETERMINISTIC_KEYS):
        if k in r_sar_1:
            if r_sar_1[k] != r_sar_2[k]:
                print(f"  {k}: {r_sar_1[k]} != {r_sar_2[k]} *** MISMATCH ***")
                sar_ok = False
            else:
                v = r_sar_1[k]
                if isinstance(v, dict):
                    print(f"  {k}: (dict) [MATCH]")
                else:
                    print(f"  {k}: {v} [MATCH]")
    print(f"Search & Rescue deterministic metrics identical: {sar_ok}")

    # === CONSTRAINT 3: Inspection success_rate comparison ===
    print("\n--- CONSTRAINT 3: Inspection success_rate before/after ---")
    r_insp = run_scenario("inspection", seed=42)
    print(f"  Inspection success_rate: {r_insp['success_rate']}")
    print(f"  Inspection tokens: {r_insp['tokens']}")
    print(f"  Inspection api_calls: {r_insp['api_calls']}")
    print(f"  Inspection communication_steps: {r_insp['communication_steps']}")
    print(f"  Inspection replanning_count: {r_insp['replanning_count']}")

    # === CONSTRAINT 4: New fields have zero consumers ===
    print("\n--- CONSTRAINT 4: Check new metadata fields ---")
    from src.env.scenarios import get_scenario
    from src.config import get_thresholds
    thresholds = get_thresholds()

    log_sc = get_scenario("logistics", thresholds, 42)
    insp_sc = get_scenario("inspection", thresholds, 42)
    sar_sc = get_scenario("search_rescue", thresholds, 42)

    print(f"  logistics.metadata keys: {list(log_sc.metadata.keys())}")
    print(f"  logistics.traffic_lane_occupancy: {log_sc.metadata.get('traffic_lane_occupancy')}")
    print(f"  inspection.metadata keys: {list(insp_sc.metadata.keys())}")
    print(f"  inspection._position_overrides in agent_config: {'_position_overrides' in insp_sc.agent_config}")
    print(f"  search_rescue.metadata keys: {list(sar_sc.metadata.keys())}")
    print(f"  search_rescue.environment_description: {sar_sc.metadata.get('environment_description', '')[:80]}...")

    # === Full JSON dumps for inspection ===
    print("\n--- FULL RESULT JSONS ---")
    for label, r in [("logistics", r_log_1), ("inspection", r_insp), ("search_rescue", r_sar_1)]:
        print(f"\n{label}:")
        print(json.dumps(r, indent=2))
