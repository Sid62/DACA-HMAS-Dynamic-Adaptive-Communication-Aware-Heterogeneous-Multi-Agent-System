"""Debug Search & Rescue Seed 1 and Seed 4 to trace why success rate dropped."""

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator

def debug_seed(seed):
    print(f"\n=================== DEBUGGING SEARCH_RESCUE SEED {seed} ===================")
    orch = DACAOrchestrator(
        scenario="search_rescue",
        network_profile="oscillatory",
        seed=seed,
        config=CONFIGS["A5"],
        max_steps=200,
    )
    metrics = orch.run().to_dict()
    print("SUCCESS RATE:", metrics["success_rate"])
    print("STEPS:", metrics["steps"])
    print("HALLUCINATION STATS:", metrics["hallucination_stats"])
    print("COMM STEPS:", metrics["communication_steps"])
    print("COMM BREAKDOWN:", metrics["communication_step_breakdown"])

if __name__ == "__main__":
    debug_seed(1)
    debug_seed(4)
