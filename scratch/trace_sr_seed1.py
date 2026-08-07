"""Trace Search & Rescue Seed 1 to check raw LLM responses and coalition outputs."""

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
import json

def trace_sr_seed1():
    orch = DACAOrchestrator(
        scenario="search_rescue",
        network_profile="oscillatory",
        seed=1,
        config=CONFIGS["A5"],
        max_steps=200,
    )

    # Monitor Cloud LLM complete calls
    orig_complete = orch.cloud_llm.complete
    def monitored_complete(prompt, system="", caller=""):
        res = orig_complete(prompt, system=system, caller=caller)
        if caller == "form_coalitions":
            print(f"\n[FORM_COALITIONS STEP {orch.cloud_llm.current_step}]")
            print("RAW RESPONSE:", res[:300])
        return res

    orch.cloud_llm.complete = monitored_complete
    metrics = orch.run().to_dict()
    print("\nFINAL METRICS:")
    print("Success Rate:", metrics["success_rate"])
    print("Hallucination Stats:", metrics["hallucination_stats"])

if __name__ == "__main__":
    trace_sr_seed1()
