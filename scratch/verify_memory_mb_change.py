"""Verify memory_mb change: run same seed twice + different seed, compare all metrics.

Forces use_mock=true so device LLM complete() actually reaches the memory_mb
assignment without needing a real vLLM/Ollama server.
"""
import json
import sys
sys.path.insert(0, ".")

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS
from src.config import get_llm_config


def run_seed(seed: int, config_name: str = "B1", max_steps: int = 50) -> dict:
    """Run a scenario with forced mock mode so device LLM complete() executes."""
    cfg = CONFIGS[config_name]
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=seed,
        config=cfg,
        max_steps=max_steps,
    )
    # Force mock mode so we actually exercise complete() without a server
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
    result = orch.run()
    return result.to_dict()


if __name__ == "__main__":
    print("=" * 70)
    print("VERIFICATION: memory_mb dynamic measurement")
    print("=" * 70)

    print("\n[1] Running seed=42, config=B1...")
    r1 = run_seed(42, "B1")
    print(f"    memory_mb = {r1['memory_mb']}")

    print("\n[2] Running seed=42, config=B1 (second run, same seed)...")
    r2 = run_seed(42, "B1")
    print(f"    memory_mb = {r2['memory_mb']}")

    print("\n[3] Running seed=99, config=B1...")
    r3 = run_seed(99, "B1")
    print(f"    memory_mb = {r3['memory_mb']}")

    # ---- Constraint 4: key set unchanged ----
    print("\n" + "=" * 70)
    print("CONSTRAINT 4: JSON key set unchanged")
    print("=" * 70)
    print(f"Keys: {sorted(r1.keys())}")
    keys_match = set(r1.keys()) == set(r2.keys()) == set(r3.keys())
    print(f"All three runs have identical key set: {keys_match}")

    # ---- Constraint 1: same-seed identity for all metrics EXCEPT memory_mb ----
    print("\n" + "=" * 70)
    print("CONSTRAINT 1: same-seed (42 vs 42) all non-memory_mb metrics identical")
    print("=" * 70)
    exclude = {"memory_mb"}
    all_match = True
    for k in sorted(r1.keys()):
        v1, v2 = r1[k], r2[k]
        if k in exclude:
            print(f"  {k}: run1={v1}, run2={v2}  [EXCLUDED — expected to differ]")
            continue
        if v1 != v2:
            print(f"  {k}: {v1} != {v2}  *** MISMATCH ***")
            all_match = False
        else:
            # For brevity, only print dicts in shortened form
            if isinstance(v1, dict):
                print(f"  {k}: (dict, {len(v1)} keys)  [MATCH]")
            else:
                print(f"  {k}: {v1}  [MATCH]")
    print(f"\nAll non-memory_mb metrics identical: {all_match}")

    # ---- Constraint 2: memory_mb is no longer constant ----
    print("\n" + "=" * 70)
    print("CONSTRAINT 2: memory_mb is dynamic (not constant)")
    print("=" * 70)
    vals = [r1["memory_mb"], r2["memory_mb"], r3["memory_mb"]]
    print(f"memory_mb values: {vals}")
    print(f"  Not constant 4096.0: {all(v != 4096.0 for v in vals)}")
    print(f"  Not constant 8192.0: {all(v != 8192.0 for v in vals)}")
    print(f"  Values are positive: {all(v > 0 for v in vals)}")
    # Note: in a single process, RSS readings at different moments may be similar
    # but should not be the old hardcoded constants

    # ---- Dump full JSON for inspection ----
    print("\n" + "=" * 70)
    print("FULL OUTPUT JSONS")
    print("=" * 70)
    for label, r in [("seed42_run1", r1), ("seed42_run2", r2), ("seed99", r3)]:
        print(f"\n--- {label} ---")
        print(json.dumps(r, indent=2))
