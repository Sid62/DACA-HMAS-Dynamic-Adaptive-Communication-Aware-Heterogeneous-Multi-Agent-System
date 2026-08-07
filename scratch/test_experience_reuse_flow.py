"""Script to verify end-to-end Subtask Experience Reuse across repeated runs using distance-based decomposition."""

import json
from pathlib import Path
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.memory.experience_store import SubtaskExperienceStore

store_path = Path("test_run_experience_store.json")
if store_path.exists():
    store_path.unlink()

print("--- RUN 1 (Populating Experience Store) ---")
orch1 = DACAOrchestrator(
    scenario="logistics",
    network_profile="stable",
    seed=42,
    config=CONFIGS["A1"],  # A1 uses distance decomposition (no LLM network calls required)
    max_steps=100,
)
orch1.experience_store = SubtaskExperienceStore(store_path=str(store_path), enabled=True)
orch1.centralized.experience_store = orch1.experience_store
orch1.decentralized.experience_store = orch1.experience_store

m1 = orch1.run()
d1 = m1.to_dict()
print(f"Run 1 Metrics: Success={d1['success_rate']}%, Hits={d1['experience_reuse_hits']}, Attempts={d1['experience_reuse_attempts']}")
print(f"Store exists after run 1: {store_path.exists()}")

print("\n--- RUN 2 (Reusing Stored Experience) ---")
orch2 = DACAOrchestrator(
    scenario="logistics",
    network_profile="stable",
    seed=42,
    config=CONFIGS["A1"],  # Same scenario & config
    max_steps=100,
)
orch2.experience_store = SubtaskExperienceStore(store_path=str(store_path), enabled=True)
orch2.centralized.experience_store = orch2.experience_store
orch2.decentralized.experience_store = orch2.experience_store

m2 = orch2.run()
d2 = m2.to_dict()
print(f"Run 2 Metrics: Success={d2['success_rate']}%, Hits={d2['experience_reuse_hits']}, Attempts={d2['experience_reuse_attempts']}")

# Cleanup
if store_path.exists():
    store_path.unlink()

print("\n--- VERIFICATION CHECKS ---")
assert d2['experience_reuse_attempts'] > 0, "Run 2 must have experience reuse attempts!"
assert d2['experience_reuse_hits'] > 0, "Run 2 must have at least 1 experience reuse hit!"
assert d2['success_rate'] >= d1['success_rate'], "Run 2 success rate must be >= Run 1 success rate!"
print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
