"""Verification script for Fix 2: API call counter miscounting (R8).

Tests:
1. Clean call: 1 network call, 0 cache hits, 0 failures.
2. Cache-hit call: 0 network calls (delta), 1 cache hit (delta), 0 failures. Also verifies record_call_category was invoked.
3. Retry call (fails twice, succeeds on 3rd attempt): 3 network calls, 0 cache hits, 2 failed attempts.
4. Serialized output: Verifies cloud_network_calls, cloud_disk_cache_hits, and cloud_failed_attempts appear in ExperimentMetrics.to_dict() JSON.
"""

import sys
import os
import tempfile
from pathlib import Path

# Ensure local workspace root is at sys.path[0]
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.llm.cloud_llm_client import CloudLLMClient
from src.metrics.evaluation import ExperimentMetrics, MetricsCollector

def main():
    print("--- Running Verification for Fix 2 (R8 Counters) ---")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        client = CloudLLMClient(
            config={
                "use_mock": True,
                "cache_responses": True,
                "cache_dir": str(tmp_path / "cache"),
                "cloud": {"provider": "mock", "model": "mock-model"}
            }
        )
        client.max_retries = 3

        # -------------------------------------------------------------
        # Scenario 1: Clean cloud call
        # -------------------------------------------------------------
        print("\n--- Scenario 1: Clean Call ---")
        client.usage.reset()
        resp1 = client.complete("Prompt 1: Initial task decomposition", caller="initial_planning")
        
        c_net1 = client.usage.cloud_network_calls
        c_hit1 = client.usage.cloud_disk_cache_hits
        c_fail1 = client.usage.cloud_failed_attempts
        
        print(f"Counters: cloud_network_calls={c_net1}, cloud_disk_cache_hits={c_hit1}, cloud_failed_attempts={c_fail1}")
        assert c_net1 == 1, f"Expected 1 network call, got {c_net1}"
        assert c_hit1 == 0, f"Expected 0 cache hits, got {c_hit1}"
        assert c_fail1 == 0, f"Expected 0 failures, got {c_fail1}"
        assert client.usage.initial_planning_calls == 1, "Expected initial_planning_calls category to be 1"
        print("[PASS] Scenario 1 counters match expected (1 network, 0 hit, 0 fail)")

        # -------------------------------------------------------------
        # Scenario 2: Cache hit call
        # -------------------------------------------------------------
        print("\n--- Scenario 2: Disk Cache Hit ---")
        # Do not reset usage so we measure cumulative / deltas
        prev_net = client.usage.cloud_network_calls
        prev_hits = client.usage.cloud_disk_cache_hits
        prev_fails = client.usage.cloud_failed_attempts
        prev_replan = client.usage.cqi_replan_calls

        resp2 = client.complete("Prompt 1: Initial task decomposition", caller="cqi_replan")

        delta_net = client.usage.cloud_network_calls - prev_net
        delta_hits = client.usage.cloud_disk_cache_hits - prev_hits
        delta_fails = client.usage.cloud_failed_attempts - prev_fails
        delta_replan = client.usage.cqi_replan_calls - prev_replan

        print(f"Deltas: cloud_network_calls={delta_net}, cloud_disk_cache_hits={delta_hits}, cloud_failed_attempts={delta_fails}")
        assert resp2 == resp1, "Cached response did not match original"
        assert delta_net == 0, f"Expected 0 network call delta for cache hit, got {delta_net}"
        assert delta_hits == 1, f"Expected 1 cache hit delta, got {delta_hits}"
        assert delta_fails == 0, f"Expected 0 failure delta, got {delta_fails}"
        assert delta_replan == 1, f"Expected record_call_category to increment cqi_replan_calls by 1, got {delta_replan}"
        print("[PASS] Scenario 2 counters match expected (0 network, 1 hit, 0 fail, category attributed)")

        # -------------------------------------------------------------
        # Scenario 3: Retried call failing twice before succeeding
        # -------------------------------------------------------------
        print("\n--- Scenario 3: Fails twice before succeeding ---")
        client.usage.reset()
        client.config["use_mock"] = False
        client.backoff_base = 0.001  # fast backoff for test

        attempt_counter = [0]
        def mock_api_call(prompt, system):
            attempt_counter[0] += 1
            if attempt_counter[0] < 3:
                raise RuntimeError(f"Simulated transient error attempt {attempt_counter[0]}")
            return "Successful response after 2 failures", 10, 10, 20

        client._api_call = mock_api_call

        resp3 = client.complete("Prompt 3: Retry test prompt", caller="retry_test")
        
        c_net3 = client.usage.cloud_network_calls
        c_hit3 = client.usage.cloud_disk_cache_hits
        c_fail3 = client.usage.cloud_failed_attempts

        print(f"Counters: cloud_network_calls={c_net3}, cloud_disk_cache_hits={c_hit3}, cloud_failed_attempts={c_fail3}")
        print("Note on accounting rule: Only the 2 failed HTTP attempts increment cloud_failed_attempts. The 3rd attempt succeeded, so it is not a failure.")
        assert resp3 == "Successful response after 2 failures"
        assert c_net3 == 3, f"Expected 3 network calls (2 failed + 1 success), got {c_net3}"
        assert c_hit3 == 0, f"Expected 0 cache hits, got {c_hit3}"
        assert c_fail3 == 2, f"Expected 2 failed attempts, got {c_fail3}"
        print("[PASS] Scenario 3 counters match expected (3 network calls: 2 failed + 1 success)")

        # -------------------------------------------------------------
        # Scenario 4: JSON Metrics Serialization Verification
        # -------------------------------------------------------------
        print("\n--- Scenario 4: JSON Metrics Serialization ---")
        collector = MetricsCollector()
        metrics = collector.finalize(
            success_rate=1.0,
            steps=10,
            cloud_tokens=100,
            cloud_api_calls=client.usage.cloud_api_calls,
            device_tokens=0,
            device_api_calls=0,
            device_memory_mb=10.0,
            computation_s=1.0,
            total_wall_clock_s=2.0,
            tfr_history=[1.0],
            cfr_history=[1.0],
            switch_count=0,
            config_name="test_config",
            scenario="logistics",
            network_profile="stable",
            seed=42,
            cloud_network_calls=client.usage.cloud_network_calls,
            cloud_disk_cache_hits=client.usage.cloud_disk_cache_hits,
            cloud_failed_attempts=client.usage.cloud_failed_attempts,
        )

        metrics_dict = metrics.to_dict()
        print(f"Serialized metrics dict keys: {sorted(list(metrics_dict.keys()))}")
        
        assert "cloud_network_calls" in metrics_dict, "cloud_network_calls missing from to_dict()"
        assert "cloud_disk_cache_hits" in metrics_dict, "cloud_disk_cache_hits missing from to_dict()"
        assert "cloud_failed_attempts" in metrics_dict, "cloud_failed_attempts missing from to_dict()"
        
        assert metrics_dict["cloud_network_calls"] == 3
        assert metrics_dict["cloud_disk_cache_hits"] == 0
        assert metrics_dict["cloud_failed_attempts"] == 2
        
        print("[PASS] All 3 new counters properly serialized into final metrics JSON!")

    print("\n[ALL FIX 2 VERIFICATION TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    main()
