import json
import sys
import time
import os
import io
import psutil

sys.path.insert(0, ".")

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS
from src.llm.device_llm_client import DeviceLLMClient, aggregate_device_usage

class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout

def test_metric1_memory():
    print("\n=======================================================")
    print("=== PROBING METRIC 1: memory_mb ===")
    print("=======================================================")
    
    # 1. Run short run (max_steps=10)
    with SuppressOutput():
        orch1 = DACAOrchestrator(scenario="inspection", network_profile="stable", seed=42, config=CONFIGS["A5"], max_steps=10)
        orch1.cloud_llm.config["use_mock"] = True
        for dc in orch1.device_llms.values():
            dc.config["use_mock"] = True
        m1 = orch1.run()
        res1 = m1.to_dict()
    
    # 2. Run longer run (max_steps=100)
    with SuppressOutput():
        orch2 = DACAOrchestrator(scenario="inspection", network_profile="stable", seed=42, config=CONFIGS["A5"], max_steps=100)
        orch2.cloud_llm.config["use_mock"] = True
        for dc in orch2.device_llms.values():
            dc.config["use_mock"] = True
        m2 = orch2.run()
        res2 = m2.to_dict()
    
    print(f"Run 1 (max_steps=10): api_calls={m1.total_api_calls}, memory_mb (raw sum)={m1.device_memory_mb:.2f} MB, in JSON dict={res1['memory_mb']} MB")
    print(f"Run 2 (max_steps=100): api_calls={m2.total_api_calls}, memory_mb (raw sum)={m2.device_memory_mb:.2f} MB, in JSON dict={res2['memory_mb']} MB")
    
    # 3. Test Cache Hit behavior on DeviceLLMClient
    print("\n-- Testing Cache Hit vs Miss memory_mb update in DeviceLLMClient --")
    dummy_config = {"use_mock": True, "cache_dir": "scratch/cache_test"}
    os.makedirs("scratch/cache_test", exist_ok=True)
    d_client = DeviceLLMClient(node_id="test_node", config=dummy_config)
    
    # Force initial memory_mb to 0.0
    d_client.usage.memory_mb = 0.0
    print(f"Before any complete(): usage.memory_mb = {d_client.usage.memory_mb}")
    
    # First call (cache miss)
    with SuppressOutput():
        resp1 = d_client.complete(prompt="Hello device prompt 1", caller="probe_miss")
    mem_after_miss = d_client.usage.memory_mb
    print(f"After Cache MISS: usage.memory_mb = {mem_after_miss:.4f} MB")
    
    # Manually mutate usage.memory_mb to sentinel value 999.999
    d_client.usage.memory_mb = 999.999
    with SuppressOutput():
        resp2 = d_client.complete(prompt="Hello device prompt 1", caller="probe_hit") # Repeated prompt triggers cache hit
    mem_after_hit = d_client.usage.memory_mb
    print(f"After Cache HIT (prompt repeated): usage.memory_mb = {mem_after_hit:.4f} MB (Sentinel 999.999 unchanged because cache hit skipped RSS update!)")

def test_metric2_computation():
    print("\n=======================================================")
    print("=== PROBING METRIC 2: computation_s and total_wall_clock_s ===")
    print("=======================================================")
    
    scenarios_configs = [
        ("inspection", "stable", "A5", 20),
        ("logistics", "oscillatory", "A5", 50),
        ("search_rescue", "sudden", "A2", 80),
    ]
    
    for i, (sc, net, cfg_key, steps) in enumerate(scenarios_configs, 1):
        with SuppressOutput():
            orch = DACAOrchestrator(scenario=sc, network_profile=net, seed=42+i, config=CONFIGS[cfg_key], max_steps=steps)
            orch.cloud_llm.config["use_mock"] = True
            for dc in orch.device_llms.values():
                dc.config["use_mock"] = True
            m = orch.run()
            res = m.to_dict()
        
        comp = m.computation_s
        wall = m.total_wall_clock_s
        cloud_calls = m.cloud_api_calls
        dev_calls = m.device_api_calls
        total_calls = m.total_api_calls
        cloud_wait = orch.cloud_llm.usage.llm_wait_s
        device_wait = sum(dc.usage.llm_wait_s for dc in orch.device_llms.values())
        total_wait = cloud_wait + device_wait
        
        diff = wall - comp
        print(f"\nRun {i} ({sc}, {net}, {cfg_key}, steps={steps}):")
        print(f"  total_wall_clock_s (JSON) = {res['total_wall_clock_s']:.3f} s (raw: {wall:.6f} s)")
        print(f"  computation_s (JSON)      = {res['computation_s']:.3f} s (raw: {comp:.6f} s)")
        print(f"  total_llm_wait_s          = {total_wait:.6f} s (cloud={cloud_wait:.6f}s, device={device_wait:.6f}s)")
        print(f"  gap (wall - computation)   = {diff:.6f} s")
        print(f"  cloud_api_calls={cloud_calls}, device_api_calls={dev_calls}, total_api_calls={total_calls}")
        print(f"  Condition total_wall_clock_s >= computation_s: {wall >= comp}")
        print(f"  Did clamp fire (elapsed < total_llm_wait_s)? {'YES' if comp == 0.0 and diff > wall else 'NO'}")

def test_metric3_communication():
    print("\n=======================================================")
    print("=== PROBING METRIC 3: communication_steps and breakdown ===")
    print("=======================================================")
    
    # Run 1: Mode-switching run (logistics, oscillatory profile, A5 config)
    with SuppressOutput():
        orch1 = DACAOrchestrator(scenario="logistics", network_profile="oscillatory", seed=101, config=CONFIGS["A5"], max_steps=100)
        orch1.cloud_llm.config["use_mock"] = True
        for dc in orch1.device_llms.values():
            dc.config["use_mock"] = True
        m1 = orch1.run()
        res1 = m1.to_dict()
    
    # Run 2: Decentralized run (search_rescue, sudden profile, A2 config)
    with SuppressOutput():
        orch2 = DACAOrchestrator(scenario="search_rescue", network_profile="sudden", seed=202, config=CONFIGS["A2"], max_steps=60)
        orch2.cloud_llm.config["use_mock"] = True
        for dc in orch2.device_llms.values():
            dc.config["use_mock"] = True
        m2 = orch2.run()
        res2 = m2.to_dict()
        
    print("\n--- RUN 1 JSON Excerpt (Mode-switching Run, switch_count > 0) ---")
    print(json.dumps(res1, indent=2))
    
    breakdown1 = m1.communication_step_breakdown
    total_comm1 = m1.communication_steps
    sum_breakdown1 = sum(breakdown1.values())
    
    print(f"\nRun 1 Metrics Breakdown:")
    print(f"  switch_count: {m1.switch_count}")
    print(f"  communication_steps (total): {total_comm1}")
    print(f"  communication_step_breakdown: {breakdown1}")
    print(f"  Sum of breakdown values: {sum_breakdown1}")
    print(f"  Arithmetic equality (total == sum): {total_comm1 == sum_breakdown1}")
    print(f"  global_planning == dispatch: {breakdown1.get('global_planning')} == {breakdown1.get('dispatch')} ({breakdown1.get('global_planning') == breakdown1.get('dispatch')})")
    print(f"  local_coordination == peer_consensus == feedback_sync: {breakdown1.get('local_coordination')} == {breakdown1.get('peer_consensus')} == {breakdown1.get('feedback_sync')} ({breakdown1.get('local_coordination') == breakdown1.get('peer_consensus') == breakdown1.get('feedback_sync')})")
    total_llm1 = m1.cloud_api_calls + m1.device_api_calls
    ratio1 = total_comm1 / max(1, total_llm1)
    print(f"  LLM Invocations (cloud={m1.cloud_api_calls} + device={m1.device_api_calls}) = {total_llm1}")
    print(f"  Ratio of comm_steps / LLM calls = {total_comm1} / {total_llm1} = {ratio1:.4f}")

    print("\n--- RUN 2 Metrics Breakdown (Decentralized Run) ---")
    breakdown2 = m2.communication_step_breakdown
    total_comm2 = m2.communication_steps
    sum_breakdown2 = sum(breakdown2.values())
    print(f"  switch_count: {m2.switch_count}")
    print(f"  communication_steps (total): {total_comm2}")
    print(f"  communication_step_breakdown: {breakdown2}")
    print(f"  Sum of breakdown values: {sum_breakdown2}")
    print(f"  Arithmetic equality (total == sum): {total_comm2 == sum_breakdown2}")
    print(f"  global_planning == dispatch: {breakdown2.get('global_planning')} == {breakdown2.get('dispatch')} ({breakdown2.get('global_planning') == breakdown2.get('dispatch')})")
    print(f"  local_coordination == peer_consensus == feedback_sync: {breakdown2.get('local_coordination')} == {breakdown2.get('peer_consensus')} == {breakdown2.get('feedback_sync')} ({breakdown2.get('local_coordination') == breakdown2.get('peer_consensus') == breakdown2.get('feedback_sync')})")
    total_llm2 = m2.cloud_api_calls + m2.device_api_calls
    ratio2 = total_comm2 / max(1, total_llm2)
    print(f"  LLM Invocations (cloud={m2.cloud_api_calls} + device={m2.device_api_calls}) = {total_llm2}")
    print(f"  Ratio of comm_steps / LLM calls = {total_comm2} / {total_llm2} = {ratio2:.4f}")

if __name__ == "__main__":
    test_metric1_memory()
    test_metric2_computation()
    test_metric3_communication()
