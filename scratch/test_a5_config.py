import json
import sys
sys.path.insert(0, ".")

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS

for scenario in ["logistics", "inspection", "search_rescue"]:
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile="stable",
        seed=42,
        config=CONFIGS["A5"],
        max_steps=50,
    )
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True
    m = orch.run().to_dict()
    print(f"A5 {scenario}: success_rate={m['success_rate']}, steps={m['steps']}, tokens={m['tokens']}, api_calls={m['api_calls']}, comm_steps={m['communication_steps']}, replans={m['replanning_count']}")
