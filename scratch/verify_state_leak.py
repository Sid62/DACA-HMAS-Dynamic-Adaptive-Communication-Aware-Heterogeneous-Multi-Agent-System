"""Empirical State Leak Verification Test for DACA-HMAS."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator


def test_5_seed_state_leak():
    print("=== EMPIRICAL STATE LEAK VERIFICATION LOG ===")
    for seed in range(1, 6):
        orch = DACAOrchestrator(
            scenario="logistics",
            network_profile="oscillatory",
            seed=seed,
            config=CONFIGS["A5"],
            max_steps=10,
        )
        cloud = orch.cloud_llm
        print(f"SEED {seed} INIT -> CloudLLMClient Object ID: {id(cloud)} | _last_coalitions ID: {id(cloud._last_coalitions)} | len(_last_coalitions): {len(cloud._last_coalitions)} | total_events: {cloud.hallucination_stats['total_events']}")
        
        orch.run()

        print(f"SEED {seed} END  -> CloudLLMClient Object ID: {id(cloud)} | _last_coalitions ID: {id(cloud._last_coalitions)} | len(_last_coalitions): {len(cloud._last_coalitions)} | total_events: {cloud.hallucination_stats['total_events']}\n")


if __name__ == "__main__":
    test_5_seed_state_leak()
