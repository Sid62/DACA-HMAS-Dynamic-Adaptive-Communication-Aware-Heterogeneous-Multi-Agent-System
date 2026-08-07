"""Verification script for upgraded DACA-HMAS IEEE Transactions instrumentation."""

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS

def main():
    print("--- Running Instrumentation Verification Probe ---")
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=42,
        config=CONFIGS["A5"],
        max_steps=30,
    )
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True

    metrics = orch.run()
    print("TYPE OF METRICS:", type(metrics))
    print("METRICS FIELDS:", [f for f in dir(metrics) if not f.startswith('_')])
    d = metrics.to_dict()

    print("\n--- Exported JSON Metric Keys Check ---")
    required_keys = [
        "config", "scenario", "profile", "seed", "success_rate", "steps",
        "tokens", "cloud_prompt_tokens", "cloud_completion_tokens", "cloud_total_tokens",
        "device_prompt_tokens", "device_completion_tokens", "device_total_tokens",
        "total_prompt_tokens", "total_completion_tokens", "cloud_retry_tokens", "device_retry_tokens",
        "api_calls", "cloud_planning_calls", "device_planning_calls", "successful_calls",
        "failed_calls", "retried_calls", "cache_hits", "local_non_llm_operations",
        "memory_mb", "process_peak_rss_mb", "process_mean_rss_mb", "gpu_peak_memory_mb", "gpu_mean_memory_mb",
        "computation_s", "cloud_inference_time_s", "device_inference_time_s", "cqi_evaluation_time_s",
        "coalition_computation_time_s", "architecture_switching_time_s", "state_handoff_time_s",
        "coalition_repair_time_s", "consensus_time_s", "planning_time_s", "network_waiting_time_s",
        "simulation_computation_time_s", "total_wall_clock_s", "tfr", "cfr", "switch_count",
        "peer_messages", "broadcast_count", "consensus_rounds", "consensus_latency",
        "plan_merge_count", "distributed_replanning_count", "replanning_count", "local_reallocation_count",
        "cached_plan_reuse_count", "merged_singleton_count", "avg_planning_latency",
        "planning_latency_p50", "planning_latency_p95", "planning_latency_p99",
        "planning_latency_min", "planning_latency_max", "planning_latency_std",
        "cloud_to_device_messages", "device_to_cloud_messages", "handoff_messages",
        "coalition_messages", "repair_messages", "cloud_bytes", "peer_bytes", "broadcast_bytes",
        "total_bytes", "communication_steps", "paper_communication_steps"
    ]

    missing = [k for k in required_keys if k not in d]
    if missing:
        print(f"FAILED: Missing keys: {missing}")
    else:
        print(f"PASSED: All {len(required_keys)} required metrics present in exported JSON.")

    print("\n--- Key Value Summary ---")
    for k in ["success_rate", "steps", "cloud_total_tokens", "device_total_tokens", "total_tokens",
              "cloud_planning_calls", "device_planning_calls", "total_bytes", "process_peak_rss_mb",
              "cqi_evaluation_time_s", "planning_time_s", "simulation_computation_time_s",
              "total_wall_clock_s", "planning_latency_p50", "planning_latency_p95"]:
        print(f"  {k}: {d.get(k)}")

    # Verify duplicate record append bug fix
    assert len(orch.metrics.records) == 1, f"Expected 1 record in MetricsCollector, found {len(orch.metrics.records)}"
    print("PASSED: MetricsCollector records count = 1 (no duplicate append).")

    print("\n--- Probe Complete: SUCCESS ---")

if __name__ == "__main__":
    main()
